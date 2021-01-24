/*
* READ FIRST
*
* How to compile?
*   make bench_offline
*
* How to benchmark for two parties?
*   Assume Party 0's IP is x.x.x.x
*   Party 0: ./bench_lowgear_triple.x -N 2 -l 64 -h x.x.x.x -p 0
*   Party 1: ./bench_lowgear_triple.x -N 2 -l 64 -h x.x.x.x -p 1
*   (to support floating points with sufficient space, we need to set -l to a higher value)
*
* How to benchmark for three parties?
*   Party 0: ./bench_lowgear_triple.x -N 3 -l 64 -h x.x.x.x -p 0
*   Party 1: ./bench_lowgear_triple.x -N 3 -l 64 -h x.x.x.x -p 1
*   Party 2: ./bench_lowgear_triple.x -N 3 -l 64 -h x.x.x.x -p 2
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

void generate_one_batch(PlaintextVector<FD> &a, PlaintextVector<FD> &b, PlaintextVector<FD> &c, const int batch_size, const FHE_Params & params, const FD &FieldD, const PlainPlayer &P, vector<FHE_PK>& pks, FHE_SK *psk, ThreadPool * pool){
	PRNG G;
	G.ReSeed();

	AddableVector<Ciphertext> Ca;
	int num_players = P.num_players();
	int my_num = P.my_num();

	/*
	* Step 1: Generate a_i, b_i, c_i randomly
	* Here, c_i has the initial value, which is a_i * b_i
	*/
	start_timer_map["sampling randomized a/b, initialing c = ab"][0] = std::chrono::system_clock::now();
	{
		a.resize(batch_size, FieldD);
		b.resize(batch_size, FieldD);
		c.resize(batch_size, FieldD);
		Ca.resize(batch_size, params);
	}

	{
		a.allocate_slots(FieldD.get_prime());
		b.allocate_slots(FieldD.get_prime());
		c.allocate_slots((bigint)FieldD.get_prime() << 64);
	}

	PRNG G_array[omp_get_max_threads()];
	for(int i = 0; i < omp_get_max_threads(); i++){
		G_array[i].ReSeed();
	}

	{
		#pragma omp parallel for
		for(int i = 0; i < batch_size; i++){
			int num = omp_get_thread_num();
			a[i].randomize(G_array[num]);
			b[i].randomize(G_array[num]);
			c[i].mul(a[i], b[i]);
		}
	}
	end_timer_map["sampling randomized a/b, initialing c = ab"][0] = std::chrono::system_clock::now();
	printf("sampling randomized a/b, initialing c = ab done.\n");

	/*
	* Step 2: Encrypt a_i and prepare to send it out
	*/

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
	* Step 3: FFT b
	*/
	start_timer_map["FFTing b"][0] = std::chrono::system_clock::now();
	AddableVector<Rq_Element> b_mod_q;
	{
		b_mod_q.resize(batch_size, {params, evaluation, evaluation});
		/* here, evaluation is a type, defined in FHE/Ring_Element.h */
	}
	#pragma omp parallel for
	for (int i = 0; i < batch_size; i++){
		b_mod_q.at(i).from_vec(b.at(i).get_poly());
	}
	end_timer_map["FFTing b"][0] = std::chrono::system_clock::now();
	printf("FFTing b done.\n");

	/*
	* Step 4: Receive others' a_i in ciphertext, multiply it with b_i, and return it back.
	*/
	PlaintextVector<FD> ed[num_players];
	for(int i = 0; i < P.num_players(); i++){
		ed[i].resize(batch_size, FieldD);
		ed[i].allocate_slots(FieldD.get_prime());
		ed[i].assign_zero();
	}

	vector<octetStream> CaStream_others_send;
	CaStream_others_send.resize(num_players);

	vector<octetStream> CaStream_others_recv;
	CaStream_others_recv.resize(num_players);

	vector<future<void>> res;

	printf("exchanging a.\n");
	start_timer_map["exchanging a"][0] = std::chrono::system_clock::now();
	for(int j = 1; j < num_players; j++){
		int party = (my_num + j) % num_players;
		res.push_back(pool->enqueue([party, batch_size, CaSize, &P, &CaStream_others_recv]() {
			CaStream_others_recv[party].resize_precise(batch_size * CaSize);
			CaStream_others_recv[party].reset_write_head();

			auto sock_recv = P.socket(party);
			CaStream_others_recv[party].Receive(sock_recv);
		}));
	}

	for(int j = 1; j < num_players; j++){
		int party_send = (my_num + num_players - j) % num_players;
		res.push_back(pool->enqueue([party_send, batch_size, CaSize, &P, &CaStream]() {
			auto sock_send = P.socket(party_send);
			CaStream.Send(sock_send);
		}));
	}
	joinNclean(res);
	printf("exchanging a done.\n");
	end_timer_map["exchanging a"][0] = std::chrono::system_clock::now();

	printf("computing a times b.\n");
	start_timer_map["computing a times b"][0] = std::chrono::system_clock::now();
	for(int j = 1; j < num_players; j++){
		int party = (my_num + j) % num_players;
		res.push_back(pool->enqueue([party, batch_size, CaSize, &P, &CaStream_others_recv, &CaStream_others_send, &c, &b_mod_q, &FieldD, &params, &pks, &ed, &psk]() {
			AddableVector<Ciphertext> Ca_other;
			Ca_other.resize(batch_size, params);
			for(int i = 0; i < batch_size; i++){
				Ca_other[i].unpack(CaStream_others_recv[party]);
			}

			PlaintextVector<FD> product_share(omp_get_max_threads(), FieldD);
			Random_Coins rc(params);
			product_share.allocate_slots(params.p0() << 64);

			bigint B = 6 * params.get_R();
			B *= FieldD.get_prime();
			B <<= 40;

			PRNG G_array[omp_get_max_threads()];
			for(int i = 0; i < omp_get_max_threads(); i++){
				G_array[i].ReSeed();
			}

			AddableVector<Ciphertext> tmp1, tmp2;
			tmp1.resize(batch_size, params);
			tmp2.resize(batch_size, params);

			#pragma omp parallel for
			for(int i = 0; i < batch_size; i++){
				int num = omp_get_thread_num();

				tmp1[i].mul(Ca_other[i], b_mod_q[i]);
				product_share[num].randomize(G_array[num]);

				Random_Coins rc2(params);
				rc2.generateUniform(G_array[num], 0, B, B);

				ed[party][i] -= product_share[num];

				pks[party].encrypt(tmp2[i], product_share[num], rc2);
				tmp1[i] += tmp2[i];
			}
			CaStream_others_send[party].resize_precise(batch_size * CaSize);
			CaStream_others_send[party].reset_write_head();
			for(int i = 0; i < batch_size; i++){
				tmp1[i].pack(CaStream_others_send[party]);
			}
		}));
	}
	joinNclean(res);
	printf("computing a times b done.\n");
	end_timer_map["computing a times b"][0] = std::chrono::system_clock::now();

	printf("exchanging the final result.\n");
	start_timer_map["exchanging the final result"][0] = std::chrono::system_clock::now();
	for(int j = 1; j < num_players; j++){
		int party = (my_num + j) % num_players;
		res.push_back(pool->enqueue([party, &P, &CaStream_others_recv]() {
			CaStream_others_recv[party].reset_write_head();
			auto sock_recv = P.socket(party);
			CaStream_others_recv[party].Receive(sock_recv);
		}));
	}
	for(int j = 1; j < num_players; j++){
		int party = (my_num + j) % num_players;
		res.push_back(pool->enqueue([party, &P, &CaStream_others_send]() {
			auto sock_send = P.socket(party);
			CaStream_others_send[party].Send(sock_send);
		}));
	}
	joinNclean(res);
	printf("exchanging the final result done.\n");
	end_timer_map["exchanging the final result"][0] = std::chrono::system_clock::now();

	AddableVector<Ciphertext> tmp1[omp_get_max_threads()];
	for(int i = 0; i < omp_get_max_threads(); i++){
		tmp1[i].resize(batch_size, params);
	}

	PlaintextVector<FD> product_share(omp_get_max_threads(), FieldD);
	product_share.allocate_slots(params.p0() << 64);

	printf("decrypting the final result.\n");
	start_timer_map["decrypting the final result"][0] = std::chrono::system_clock::now();
	#pragma omp parallel for
	for(int j = 1; j < num_players; j++){
		int party = (my_num + j) % num_players;

		int num = omp_get_thread_num();
		for(int i = 0; i < batch_size; i++){
			tmp1[num][i].unpack(CaStream_others_recv[party]);
		}
		for(int i = 0; i < batch_size; i++){
			(*psk).decrypt_any(product_share[num], tmp1[num][i]);
			ed[party][i] += product_share[num];
		}
	}
	joinNclean(res);
	printf("decrypting the final result done.\n");
	end_timer_map["decrypting the final result"][0] = std::chrono::system_clock::now();

	printf("calculating the final c.\n");
	start_timer_map["calculating the final c"][0] = std::chrono::system_clock::now();
	#pragma omp parallel for
	for(int i = 0; i < batch_size; i++){
		for(int j = 0; j < num_players; j++){
			if(j != my_num){
				c[i] += ed[j][i];
			}
		}
	}
	end_timer_map["calculating the final c"][0] = std::chrono::system_clock::now();
	printf("calculating the final c done.\n");

	printf("synchronizing all parties to end.\n");
	vector<octetStream> os(P.num_players());
	bool sync = true;
	os[P.my_num()].reset_write_head();
  os[P.my_num()].store_int(sync, 1);
  P.Broadcast_Receive(os);
	printf("synchronizing all parties to end done.\n");
}

void check_first_result(PlaintextVector<FD> &a, PlaintextVector<FD> &b, PlaintextVector<FD> &c, const FHE_Params & params, const FD &FieldD, const PlainPlayer &P, ThreadPool * pool){
	(void)(params);

	int num_players = P.num_players();
	int my_num = P.my_num();

	if(my_num != 0){
		octetStream abcStream;
		abcStream.resize_precise(a[0].report_size(USED) + b[0].report_size(USED) + c[0].report_size(USED));
		abcStream.reset_write_head();

		a[0].pack(abcStream);
		b[0].pack(abcStream);
		c[0].pack(abcStream);

		P.comm_stats["Sending directly"].add(abcStream);
		auto socket_send = P.socket(0);
		abcStream.Send(socket_send);
		P.sent += abcStream.get_length();
	}else{
		PlaintextVector<FD> a_other(num_players, FieldD);
		PlaintextVector<FD> b_other(num_players, FieldD);
		PlaintextVector<FD> c_other(num_players, FieldD);

		for(int i = 0; i < P.num_players(); i++){
			a_other[i].allocate_slots(FieldD.get_prime());
			b_other[i].allocate_slots(FieldD.get_prime());
			c_other[i].allocate_slots((bigint)FieldD.get_prime() << 64);
		}

		a_other[0] = a[0];
		b_other[0] = b[0];
		c_other[0] = c[0];

		int rowsize = a[0].report_size(USED) + b[0].report_size(USED) + c[0].report_size(USED);

		printf("Ready to receive data.\n");

		vector<future<void>> res;
		for(int j = 1; j < num_players; j++){
			int party = j;
			res.push_back(pool->enqueue([party, rowsize, &P, &a_other, &b_other, &c_other](){
				octetStream abcStream_other;
				abcStream_other.resize_precise(rowsize);
				abcStream_other.reset_write_head();

				auto socket_recv = P.socket(party);
				abcStream_other.Receive(socket_recv);

				a_other[party].unpack(abcStream_other);
				b_other[party].unpack(abcStream_other);
				c_other[party].unpack(abcStream_other);
			}));
		}
		joinNclean(res);

		printf("obtain all data from different parties.\n");

		for(int i = 0; i < num_players; i++){
			printf("Party %d\n", i);

			a_other[i].print_evaluation(1, "a");
			b_other[i].print_evaluation(1, "b");
			c_other[i].print_evaluation(1, "c");
		}

		Plaintext_<FD> a_sum(FieldD);
		Plaintext_<FD> b_sum(FieldD);
		Plaintext_<FD> c_sum(FieldD);
		Plaintext_<FD> cc_sum(FieldD);

		a_sum.allocate_slots(FieldD.get_prime());
		b_sum.allocate_slots(FieldD.get_prime());
		c_sum.allocate_slots((bigint)FieldD.get_prime() << 64);
		cc_sum.allocate_slots((bigint)FieldD.get_prime() << 64);

		a_sum = a_other[0];
		b_sum = b_other[0];
		c_sum = c_other[0];

		for(int i = 1; i < num_players; i++){
			a_sum += a_other[i];
			b_sum += b_other[i];
			c_sum += c_other[i];
		}

		printf("\n");

		a_sum.print_evaluation(2, "a_sum ");
		b_sum.print_evaluation(2, "b_sum ");
		c_sum.print_evaluation(2, "c_sum ");

		cc_sum.mul(a_sum, b_sum);
		cc_sum.print_evaluation(2, "cc_sum");
	}
}

void write_triples_to_file(PlaintextVector<FD>& a, PlaintextVector<FD>& b, PlaintextVector<FD>& c, int batch_size, int nplayers, FD& FieldD, int my_num) {
        cout << "Reading in MAC Key: " << endl;
        gfp mkey;
        mkey.init_field(FieldD.get_prime());
        for (int i = 0; i < nplayers; i++) {
                string mac_file_name = "/home/ubuntu/cerebro/crypto_backend/SCALE-MAMBA/Data/MKey-" + to_string(i) + ".key";
                cout << "MAC FILE NAME: " << mac_file_name << endl;
                ifstream mac_file;
                mac_file.open(mac_file_name);
                gfp mac_share;
                mac_share.init_field(FieldD.get_prime());
                mac_file >> mac_share;
                mkey = mkey + mac_share;
                mac_file.close();
        }
        cout << "MAC Key " << mkey << endl;
        string file_name = "/home/ubuntu/triple_" + to_string(my_num) + ".txt";
        ofstream file;
        file.open(file_name, ios::out | ios::app);
        cout << "Outputting triples to file" << endl;
        octetStream o;
        unsigned int num_slots = a[0].num_slots();
        uint8_t party_num = my_num;
        for (int i = 0; i < batch_size; i++) {
                for (unsigned int j = 0; j < num_slots; j++) {
                        file << party_num;
                        gfp share_a = a[i].element(0);
                        share_a.output(file, false);
                        //cout << "Share a: " << share_a << endl;
                        gfp mac_share_a = share_a * mkey;
                        //cout << "Mac share a: " << mac_share_a << endl;
                        mac_share_a.output(file, false);

                        file << party_num;
                        gfp share_b = b[i].element(0);
                        share_b.output(file, false);
                        //cout << "Share b: " << share_b << endl;
                        gfp mac_share_b = share_b * mkey;
                        //cout << "Mac share b: " << mac_share_b << endl;
                        mac_share_b.output(file, false);

                        file << party_num;
                        gfp share_c = c[i].element(0);
                        share_c.output(file, false);
                        //cout << "Share c: " << share_c << endl;
                        gfp mac_share_c = share_c * mkey;
                        mac_share_c.output(file, false);

                }

        }
        o.output(file);
        file.close();
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
	opt.add(
		"1", // Default.
		1, // Required?
		1, // Number of args expected.
		0, // Delimiter if expecting multiple args.
		"Number of triples", // Help description.
		"-t", // Flag token.
		"--triple" // Flag token.
	);
        opt.parse(argc, argv);

	// Input the simulation parameters
	int nplayers = 2;
	int plainlength = 64;
	int my_num = 0;
	int batch_size = 40;
	int num_triples = 1;
	string hostname;

	opt.get("-N")->getInt(nplayers);
	opt.get("-l")->getInt(plainlength);
	opt.get("-h")->getString(hostname);
	opt.get("-p")->getInt(my_num);
	opt.get("-b")->getInt(batch_size);
	opt.get("-t")->getInt(num_triples);

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
	PlaintextVector<FD> b;
	PlaintextVector<FD> c;
	generate_one_batch(a, b, c, batch_size, params, FieldD, P, pks, psk, &pool);
	int num_rounds = num_triples / (a[0].num_slots() * batch_size) + 1;
	std::chrono::time_point<std::chrono::system_clock> start_time, end_time;
	long total_time = 0;
	for (int i = 0; i < num_rounds; i++) {
		start_time = std::chrono::system_clock::now();
		generate_one_batch(a, b, c, batch_size, params, FieldD, P, pks, psk, &pool);
		end_time = std::chrono::system_clock::now();
		total_time += (std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count());
	}

	check_first_result(a, b, c, params, FieldD, P, &pool);

	// cerr << "Time " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()) / 1000.0 << " seconds " << endl;
	cerr << "Time " << total_time / 1000.0 << " seconds " << endl;
	cerr << "Generated " << num_rounds * (a[0].num_slots()) * batch_size << " triplets." << endl;

	cerr << endl;
	// cerr << "Rate: " << (a[0].num_slots()) * batch_size * num_rounds / ((std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()) / 1000.0) << " triplets/second" << endl;
	cerr << "Rate: " << (a[0].num_slots() * batch_size * num_rounds) / (total_time / 1000.0) << " triplets/second" << endl;

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
