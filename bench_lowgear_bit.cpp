/*
* READ FIRST
*
* How to compile?
*   make bench_offline
*
* How to benchmark for two parties?
*   Assume Party 0's IP is x.x.x.x
*   Party 0: ./bench_lowgear_bit.x -N 2 -l 64 -h x.x.x.x -p 0
*   Party 1: ./bench_lowgear_bit.x -N 2 -l 64 -h x.x.x.x -p 1
*   (to support floating points with sufficient space, we need to set -l to a higher value)
*
* How to benchmark for three parties?
*   Party 0: ./bench_lowgear_bit.x -N 3 -l 64 -h x.x.x.x -p 0
*   Party 1: ./bench_lowgear_bit.x -N 3 -l 64 -h x.x.x.x -p 1
*   Party 2: ./bench_lowgear_bit.x -N 3 -l 64 -h x.x.x.x -p 2
*
* How to change the batch size?
*   Add parameter -b 40 to make the batch size 40 in all the commands above (must be added for each party)
*
*/

#include <signal.h>
#include <cstdio>
#include <stdexcept>
#include "./ThreadPool.h"
#include <future>
#include <chrono>
#include <omp.h>
#include <ctime>
#include <iostream>
using namespace std;
using std::future;
using std::cout;
using std::max;
using std::cerr;
using std::endl;
using std::flush;

#include "FHEOffline/Multiplier.h"
#include "FHEOffline/DataSetup.h"
#include "FHEOffline/DistKeyGen.h"
#include "FHEOffline/EncCommit.h"
#include "FHEOffline/Producer.h"
#include "Networking/Server.h"
#include "FHE/NTL-Subs.h"
#include "Tools/ezOptionParser.h"
#include "Tools/mkpath.h"
#include "Math/Setup.h"
#include "Tools/time-func.h"

#define FD FFT_Data

map<string, map<int, std::chrono::time_point<std::chrono::system_clock>>> start_timer_map;
map<string, map<int, std::chrono::time_point<std::chrono::system_clock>>> end_timer_map;

template<typename T>
void joinNclean(vector<future<T>>& res) {
	/*
	* This function is stolen from Xiao Wang's AG-MPC.
	*/
	for(auto &v: res) v.get();
	res.clear();
}

void find_FHE_param(FHE_Params &params, FD& FieldD, const Player& P, int plainlength){
	/*
	* Generate FHE parameter without extra slack
	* Party 0 samples the parameter, other parties accept the parameter.
	*/
	PRNG G;
	G.ReSeed();

	octetStream o;
	if (P.my_num() == 0){
		start_timer_map["generating FHE parameters (setup)"][0] = std::chrono::system_clock::now();
		generate_semi_setup(plainlength, 40, params, FieldD, false);
		end_timer_map["generating FHE parameters (setup)"][0] = std::chrono::system_clock::now();

		params.pack(o);
		FieldD.pack(o);

		start_timer_map["sending FHE parameters (setup)"][0] = std::chrono::system_clock::now();
		P.send_all(o);
		end_timer_map["sending FHE parameters (setup)"][0] = std::chrono::system_clock::now();
	}
	else
	{
		P.receive_player(0, o);
		params.unpack(o);
		FieldD.unpack(o);
		FieldD.init_field();
	}
}

void FHE_keygen(vector<FHE_PK>& pks, FHE_SK* &psk, const Player& P, int plainlength, const Names &N, FHE_Params &params, FD &FieldD){
	find_FHE_param(params, FieldD, P, plainlength);

	/*
	* Generate my public key and private key
	*/
	pks.resize(N.num_players(), {params, 0});
	for (auto& x : pks)
	x = FHE_PK(params, FieldD.get_prime());
	auto pk = FHE_PK(params, FieldD.get_prime());
	auto sk = new FHE_SK(pk);
	psk = sk;

	start_timer_map["generating FHE keys (setup)"][0] = std::chrono::system_clock::now();
	PRNG G;
	G.ReSeed();
	KeyGen(pk, (*sk), G);
	end_timer_map["generating FHE keys (setup)"][0] = std::chrono::system_clock::now();

	/*
	* Broadcast my key to others
	*/
	vector<octetStream> os(N.num_players());
	pk.pack(os[N.my_num()]);
	start_timer_map["receiving FHE keys (setup)"][0] = std::chrono::system_clock::now();
	P.Broadcast_Receive(os);
	end_timer_map["receiving FHE keys (setup)"][0] = std::chrono::system_clock::now();

	for(int i = 0; i < N.num_players(); i++){
		pks[i].unpack(os[i]);
	}

	for (int i = 0; i < N.num_players(); i++)
	cout << "Player " << i << " has pk " << pks[i].a().get(0).get_constant().get_limb(0) << " ..." << endl;

	os.clear();
}

void network_setup(Names& N, const int nplayers, const int my_num, const string &hostname){
	Server::start_networking(N, my_num, nplayers, hostname, 12345);
}

void generate_one_batch(PlaintextVector<FD> &a, const int batch_size, const FHE_Params & params, const FD &FieldD, const PlainPlayer &P, vector<FHE_PK>& pks, FHE_SK *psk, ThreadPool * pool){
	PRNG G;
	G.ReSeed();

	PlaintextVector<FD> c;

	AddableVector<Ciphertext> Ca;
	int num_players = P.num_players();
	int my_num = P.my_num();

	/*
	* Step 1: Generate a_i, b_i, c_i randomly
	* Here, c_i has the initial value, which is a_i * b_i
	*/
	start_timer_map["sampling randomized a, initialing c = aa"][0] = std::chrono::system_clock::now();
	{
		a.resize(batch_size, FieldD);
		c.resize(batch_size, FieldD);
		Ca.resize(batch_size, params);
	}

	{
		a.allocate_slots(FieldD.get_prime());
		c.allocate_slots((bigint)FieldD.get_prime() << 64);
	}

	{
		a.randomize(G);
		c.mul(a, a);
	}
	end_timer_map["sampling randomized a, initialing c = aa"][0] = std::chrono::system_clock::now();
	printf("sampling randomized a, initialing c = aa done.\n");

	/*
	* Step 2: Encrypt a_i and prepare to send it out
	*/
	PRNG G_array[omp_get_max_threads()];
	for(int i = 0; i < omp_get_max_threads(); i++){
		G_array[i].ReSeed();
	}

	start_timer_map["encrypting a"][0] = std::chrono::system_clock::now();
	Random_Coins rc(params);

	#pragma omp parallel for
	for(int i = 0; i < batch_size; i++){
		int num = omp_get_thread_num();
		Random_Coins rc2(params);
		rc2.generate(G_array[num]);
		pks[my_num].encrypt(Ca[i], a[i], rc2);
	}
	end_timer_map["encrypting a"][0] = std::chrono::system_clock::now();
	printf("encrypting a done.\n");

	octetStream CaStream;
	int CaSize = Ca[0].report_size(USED);
	CaStream.resize_precise(batch_size * CaSize);
	CaStream.reset_write_head();
	for(int i = 0; i < batch_size; i++){
		Ca[i].pack(CaStream);
	}

	/*
	* Step 3: FFT a
	*/
	start_timer_map["FFTing a"][0] = std::chrono::system_clock::now();
	AddableVector<Rq_Element> a_mod_q;
	{
		a_mod_q.resize(batch_size, {params, evaluation, evaluation});
		/* here, evaluation is a type, defined in FHE/Ring_Element.h */
	}
	for (int i = 0; i < batch_size; i++){
		a_mod_q.at(i).from_vec(a.at(i).get_poly());
	}
	end_timer_map["FFTing a"][0] = std::chrono::system_clock::now();
	printf("FFTing a done.\n");

	/*
	* Step 4: Receive others' a_i in ciphertext, multiply it with a_i, and return it back.
	*/
	PlaintextVector<FD> ed[num_players];
	for(int i = 0; i < P.num_players(); i++){
		ed[i].resize(batch_size, FieldD);
		ed[i].allocate_slots(FieldD.get_prime());
		ed[i].assign_zero();
	}

	/* test decrypt */
	Plaintext_<FD> product_share(FieldD);
	product_share.allocate_slots(FieldD.get_prime());

	vector<octetStream> CaStream_others;
	CaStream_others.resize(num_players);

	vector<future<void>> res;
	for(int j = 1; j < num_players; j++){
		int party = (my_num + j) % num_players;
		int party_send = (my_num + num_players - j) % num_players;
		res.push_back(pool->enqueue([party, party_send, batch_size, CaSize, &P, &CaStream, &CaStream_others]() {
			PRNG G;
			G.ReSeed();

			CaStream_others[party].resize_precise(batch_size * CaSize);
			CaStream_others[party].reset_write_head();

			printf("receiving encrypted a from party %d.\n", party);
			{
				start_timer_map["receiving encrypted a"][party] = std::chrono::system_clock::now();
				P.comm_stats["Exchanging"].add(CaStream);
				CaStream.exchange(P.socket(party_send), P.socket(party), CaStream_others[party]);
				P.sent += CaStream.get_length();
				end_timer_map["receiving encrypted a"][party] = std::chrono::system_clock::now();
			}
			printf("receiving encrypted a from party %d done.\n", party);
		}));
	}
	joinNclean(res);

	for(int j = 1; j < num_players; j++){
		int party = (my_num + j) % num_players;
		int party_send = (my_num + num_players - j) % num_players;

		res.push_back(pool->enqueue([party, party_send, batch_size, CaSize, &P, &CaStream_others, &c, &a_mod_q, &FieldD, &params, &pks, &ed, &psk]() {
			AddableVector<Ciphertext> Ca_other;
			Ca_other.resize(batch_size, params);
			for(int i = 0; i < batch_size; i++){
				Ca_other[i].unpack(CaStream_others[party]);
			}

			AddableVector<Ciphertext> tmp1, tmp2;
			tmp1.resize(batch_size, params);
			tmp2.resize(batch_size, params);

			PlaintextVector<FD> product_share(omp_get_max_threads(), FieldD);
			Random_Coins rc(params);
			product_share.allocate_slots(params.p0() << 64);

			bigint B = 6 * params.get_R();
			B *= FieldD.get_prime();
			B <<= 40;

			printf("multiplying by a and add randomness for party %d.\n", party);
			start_timer_map["multiplying by a and add randomness"][party] = std::chrono::system_clock::now();

			PRNG G_array[omp_get_max_threads()];
			for(int i = 0; i < omp_get_max_threads(); i++){
				G_array[i].ReSeed();
			}

			#pragma omp parallel for
			for(int i = 0; i < batch_size; i++){
				tmp1[i].mul(Ca_other[i], a_mod_q[i]);

				int num = omp_get_thread_num();
				product_share[num].randomize(G_array[num]);

				Random_Coins rc2(params);
				rc2.generateUniform(G_array[num], 0, B, B);

				ed[party][i] -= product_share[num];

				pks[party].encrypt(tmp2[i], product_share[num], rc2);
				tmp1[i] += tmp2[i];
			}
			end_timer_map["multiplying by a and add randomness"][party] = std::chrono::system_clock::now();
			printf("multiplying by a and add randomness for party %d done.\n", party);

			CaStream_others[party].reset_write_head();
			for(int i = 0; i < batch_size; i++){
				tmp1[i].pack(CaStream_others[party]);
			}

			printf("sending the final result with party %d.\n", party);
			start_timer_map["sending the final result"][party] = std::chrono::system_clock::now();
			{
				P.comm_stats["Exchanging"].add(CaStream_others[party]);
				CaStream_others[party].exchange(P.socket(party), P.socket(party_send));
				P.sent += CaStream_others[party].get_length();
			}
			end_timer_map["sending the final result"][party] = std::chrono::system_clock::now();
			printf("sending the final result with party %d done.\n", party);

			for(int i = 0; i < batch_size; i++){
				tmp1[i].unpack(CaStream_others[party]);
			}

			printf("decrypting the final result with party %d.\n", party);
			start_timer_map["decrypting the final result"][party] = std::chrono::system_clock::now();
			for(int i = 0; i < batch_size; i++){
				(*psk).decrypt_any(product_share[0], tmp1[i]);
				ed[party][i] += product_share[0];
			}
			end_timer_map["decrypting the final result"][party] = std::chrono::system_clock::now();
			printf("decrypting the final result with party %d done.\n", party);
		}));
	}
	joinNclean(res);

	printf("calculating the final c.\n");
	start_timer_map["calculating the final c"][0] = std::chrono::system_clock::now();
	for(int i = 0; i < batch_size; i++){
		for(int j = 0; j < num_players; j++){
			if(j != my_num){
				c[i] += ed[j][i];
			}
		}
	}
	end_timer_map["calculating the final c"][0] = std::chrono::system_clock::now();
	printf("calculating the final c done.\n");

	if(my_num == 0){
			PlaintextVector<FD> c_others[num_players];

			printf("party 0: receiving shares of c.\n");
			start_timer_map["receiving shares of c"][0] = std::chrono::system_clock::now();

			vector<future<void>> res;
			for(int j = 1; j < num_players; j++){
				int party = j;
				res.push_back(pool->enqueue([party, &P, &c, batch_size, &FieldD, &c_others]() {
					octetStream final_cStream;
					final_cStream.resize_precise(batch_size * c[0].report_size(USED));
					final_cStream.reset_write_head();

					printf("receiving shares of c from party %d.\n", party);
					{
						auto sock_recv = P.socket(party);
						final_cStream.Receive(sock_recv);
					}
					printf("receiving encrypted a from party %d done.\n", party);

					c_others[party].resize(batch_size, FieldD);
					c_others[party].allocate_slots((bigint)FieldD.get_prime() << 64);

					for(int i = 0; i < batch_size; i++){
						c_others[party][i].unpack(final_cStream);
					}
				}));
			}
			joinNclean(res);

			end_timer_map["receiving shares of c"][0] = std::chrono::system_clock::now();
			printf("party 0: receiving shares of c done.\n");

			printf("party 0: adding all shares of c.\n");
			start_timer_map["adding all shares of c"][0] = std::chrono::system_clock::now();
			#pragma omp parallel for
			for(int i = 0; i < batch_size; i++){
				for(int j = 1; j < num_players; j++){
					c[i] = c[i] + c_others[j][i];
				}
			}
			printf("party 0: adding all shares of c done.\n");
			end_timer_map["adding all shares of c"][0] = std::chrono::system_clock::now();


			printf("party 0: sending out the sum of c.\n");
			start_timer_map["sending out the sum of c"][0] = std::chrono::system_clock::now();

			octetStream c_sum_Stream;
			c_sum_Stream.resize_precise(batch_size * c[0].report_size(USED));
			c_sum_Stream.reset_write_head();
			for(int i = 0; i < batch_size; i++){
				c[i].pack(c_sum_Stream);
			}

			vector<future<void>> res2;
			for(int j = 1; j < num_players; j++){
				int party = j;
				res2.push_back(pool->enqueue([party, &P, &c_sum_Stream]() {
					auto socket_send = P.socket(party);
					c_sum_Stream.Send(socket_send);
				}));
			}
			joinNclean(res2);

			printf("party 0: sending out the sum of c done.\n");
			end_timer_map["sending out the sum of c"][0] = std::chrono::system_clock::now();
	}else{
		printf("sending out this party's share of c=aa.\n");
		start_timer_map["sending out this party's share of c=aa"][0] = std::chrono::system_clock::now();
		octetStream final_cStream;
		final_cStream.resize_precise(batch_size * c[0].report_size(USED));
		final_cStream.reset_write_head();

		for(int i = 0; i < batch_size; i++){
			c[i].pack(final_cStream);
		}

		auto sock_send = P.socket(0);
		final_cStream.Send(sock_send);

		printf("sending out this party's share of c=aa, done\n");
		end_timer_map["sending out this party's share of c=aa"][0] = std::chrono::system_clock::now();

		printf("receiving the sum of c.\n");
		start_timer_map["receiving the sum of c"][0] = std::chrono::system_clock::now();
		final_cStream.reset_write_head();
		auto sock_recv = P.socket(0);
		final_cStream.Receive(sock_recv);

		for(int i = 0; i < batch_size; i++){
			c[i].unpack(final_cStream);
		}
		printf("receiving the sum of c done.\n");
		end_timer_map["receiving the sum of c"][0] = std::chrono::system_clock::now();
	}

	/*
	* Step 4: Assuming that no sum of a is a zero (only possible in semi-honest setting)
	* Turn c=a*a to its square root inverse
	*/
	start_timer_map["set c = inv sqrt root of c"][0] = std::chrono::system_clock::now();
	printf("set c = inv sqrt root of c.\n");
	int num_slots = c[0].num_slots();
	#pragma omp parallel for
	for(int i = 0; i < batch_size; i++){
		for(int j = 0; j < num_slots; j++){
			gfp temp = c[i].element(j).sqrRoot();
			temp.invert();
			c[i].set_element(j, temp);
		}
	}
	end_timer_map["set c = inv sqrt root of c"][0] = std::chrono::system_clock::now();
	printf("set c = inv sqrt root of c done.\n");

	/*
	* Step 5: turn every a into ((c * a) + 1) / 2
	*/
	start_timer_map["set a = ((c * a) + 1) / 2"][0] = std::chrono::system_clock::now();
	printf("set a = ((c * a) + 1) / 2.\n");
	gfp two_inv, one;
	to_gfp(two_inv, (a[0].get_field().get_prime() + 1) / 2);
	one.assign_one();
	#pragma omp parallel for
	for(int i = 0; i < batch_size; i++){
		for(int j = 0; j < num_slots; j++){
			gfp a_tmp = a[i].element(j);
			gfp c_tmp = c[i].element(j);

			a_tmp = a_tmp * c_tmp;
			if(my_num == 0){
				a_tmp = a_tmp + one;
			}
			a_tmp = a_tmp * two_inv;

			a[i].set_element(j, a_tmp);
		}
	}
	end_timer_map["set a = ((c * a) + 1) / 2"][0] = std::chrono::system_clock::now();
	printf("set a = ((c * a) + 1) / 2 done.\n");

	printf("synchronizing all parties to end.\n");
	vector<octetStream> os(P.num_players());
	bool sync = true;
	os[P.my_num()].reset_write_head();
        os[P.my_num()].store_int(sync, 1);
        P.Broadcast_Receive(os);
	printf("synchronizing all parties to end done.\n");
}

void check_first_result(PlaintextVector<FD> &a, const FHE_Params & params, const FD &FieldD, const PlainPlayer &P, ThreadPool * pool){
	(void)(params);

	int num_players = P.num_players();
	int my_num = P.my_num();

	if(my_num != 0){
		octetStream aStream;
		aStream.resize_precise(a[0].report_size(USED));
		aStream.reset_write_head();

		a[0].pack(aStream);

		P.comm_stats["Sending directly"].add(aStream);
		auto socket_send = P.socket(0);
		aStream.Send(socket_send);
		P.sent += aStream.get_length();
	}else{
		PlaintextVector<FD> a_other(num_players, FieldD);

		for(int i = 0; i < P.num_players(); i++){
			a_other[i].allocate_slots(FieldD.get_prime());
		}

		a_other[0] = a[0];

		int rowsize = a[0].report_size(USED);

		printf("Ready to receive data.\n");

		vector<future<void>> res;
		for(int j = 1; j < num_players; j++){
			int party = j;
			res.push_back(pool->enqueue([party, rowsize, &P, &a_other](){
				octetStream aStream_other;
				aStream_other.resize_precise(rowsize);
				aStream_other.reset_write_head();

				auto socket_recv = P.socket(party);
				aStream_other.Receive(socket_recv);

				a_other[party].unpack(aStream_other);
			}));
		}
		joinNclean(res);

		printf("obtain all data from different parties.\n");

		for(int i = 0; i < num_players; i++){
			printf("Party %d\n", i);

			a_other[i].print_evaluation(2, "a");
		}

		Plaintext_<FD> a_sum(FieldD);

		a_sum.allocate_slots(FieldD.get_prime());
		a_sum = a_other[0];

		for(int i = 1; i < num_players; i++){
			a_sum += a_other[i];
		}

		printf("\n");

		a_sum.print_evaluation(2, "a_sum ");
	}
}

int main(int argc, const char** argv)
{
	// Simulation parameters
	ez::ezOptionParser opt;
	opt.add(
		"2", // Default.
		0, // Required?
		1, // Number of args expected.
		0, // Delimiter if expecting multiple args.
		"Number of parties.", // Help description.
		"-N", // Flag token.
		"--nparties" // Flag token.
	);
	opt.add(
		"64", // Default.
		0, // Required?
		1, // Number of args expected.
		0, // Delimiter if expecting multiple args.
		"Plaintext length.", // Help description.
		"-l", // Flag token.
		"--plainlength" // Flag token.
	);
	opt.add(
		"", // Default.
		1, // Required?
		1, // Number of args expected.
		0, // Delimiter if expecting multiple args.
		"IP address of the host.", // Help description.
		"-h", // Flag token.
		"--hostname" // Flag token.
	);
	opt.add(
		"", // Default.
		1, // Required?
		1, // Number of args expected.
		0, // Delimiter if expecting multiple args.
		"Party ID (starting from 0).", // Help description.
		"-p", // Flag token.
		"--party" // Flag token.
	);
	opt.add(
		"40", // Default.
		1, // Required?
		1, // Number of args expected.
		0, // Delimiter if expecting multiple args.
		"Batch size.", // Help description.
		"-b", // Flag token.
		"--batch" // Flag token.
	);
	opt.parse(argc, argv);

	// Input the simulation parameters
	int nplayers = 2;
	int plainlength = 64;
	int my_num = 0;
	int batch_size = 40;

	string hostname;

	opt.get("-N")->getInt(nplayers);
	opt.get("-l")->getInt(plainlength);
	opt.get("-h")->getString(hostname);
	opt.get("-p")->getInt(my_num);
	opt.get("-b")->getInt(batch_size);

	Names N;
	network_setup(N, nplayers, my_num, hostname);

	PlainPlayer P(N, 0xffff << 16);

	vector<FHE_PK> pks;
	FHE_SK *psk;
	FHE_Params params(0);
	FD FieldD;

	FHE_keygen(pks, psk, P, plainlength, N, params, FieldD);

	/*
	* The current ThreadPool size 8 is smaller than the number of parties.
	*/
	ThreadPool pool(64);
	PlaintextVector<FD> a;
	PlaintextVector<FD> c;

	std::chrono::time_point<std::chrono::system_clock> start_time, end_time;
	start_time = std::chrono::system_clock::now();
	generate_one_batch(a, batch_size, params, FieldD, P, pks, psk, &pool);
	end_time = std::chrono::system_clock::now();

	check_first_result(a, params, FieldD, P, &pool);

	cerr << "Time " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()) / 1000.0 << " seconds " << endl;
	cerr << "Generated " << (a[0].num_slots()) * batch_size << " triplets." << endl;

	cerr << endl;
	cerr << "Rate: " << (a[0].num_slots()) * batch_size / ((std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()) / 1000.0) << " tuples/second" << endl;

	cerr << endl;
	for (auto & x : start_timer_map)
	{
		if(x.second.size() == 1){
			string name = x.first + ": ";
			name = name.append(60 - name.length(), ' ');
			cerr << name << (std::chrono::duration_cast<std::chrono::milliseconds>(end_timer_map[x.first][0] - x.second[0]).count()) / 1000.0 << " second" << endl;
		}else{
			double sum = 0.0;
			int count = 0;
			for (auto & y : x.second){
				sum += (std::chrono::duration_cast<std::chrono::milliseconds>(end_timer_map[x.first][y.first] - y.second).count()) / 1000.0;
				count ++;
			}

			string name = x.first + ": ";
			name = name.append(60 - name.length(), ' ');

			cerr << name << sum/count << " second (average)" << endl;
		}
	}

	cerr << endl;
	cerr << "The time below could be incorrect." << endl;
}
