/*
* READ FIRST
*
* How to compile?
*   make bench_offline
*
* How to benchmark for two parties?
*   Assume Party 0's IP is x.x.x.x
*   Party 0: ./bench_lowgear_matvec.x -N 2 -l 64 -h x.x.x.x -p 0
*   Party 1: ./bench_lowgear_matvec.x -N 2 -l 64 -h x.x.x.x -p 1
*   (to support floating points with sufficient space, we need to set -l to a higher value)
*
* How to benchmark for three parties?
*   Party 0: ./bench_lowgear_matvec.x -N 3 -l 64 -h x.x.x.x -p 0
*   Party 1: ./bench_lowgear_matvec.x -N 3 -l 64 -h x.x.x.x -p 1
*   Party 2: ./bench_lowgear_matvec.x -N 3 -l 64 -h x.x.x.x -p 2
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

template <class FD>
using PlaintextMatrix = AddableMatrix< Plaintext_<FD> >;

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

void generate_one_batch(PlaintextMatrix<FD> &a, PlaintextVector<FD> &b, PlaintextVector<FD> &c, int row, int column, const FHE_Params & params, const FD &FieldD, const PlainPlayer &P, vector<FHE_PK>& pks, FHE_SK *psk, ThreadPool * pool){
	PRNG G;
	G.ReSeed();

	AddableVector<Ciphertext> Cb;

	int num_players = P.num_players();
	int my_num = P.my_num();

	printf("row = %d, column = %d.\n", row, column);

	/*
	* Step 1: Generate a_i and b_i randomly
	*/
	printf("sampling randomized a/b.\n");
	start_timer_map["sampling randomized a/b"][0] = std::chrono::system_clock::now();
	{
		a.resize(row);
		for(int i = 0; i < row; i++){
			a[i].resize(column, FieldD);
		}
		b.resize(column, FieldD);
		c.resize(row, FieldD);
		Cb.resize(column, params);
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

	#pragma omp parallel for
	for(int i = 0; i < row; i++){
		int num = omp_get_thread_num();
		a[i].randomize(G_array[num]);
	}

	#pragma omp parallel for
	for(int i = 0; i < column; i++){
		int num = omp_get_thread_num();
		b.randomize(G_array[num]);
	}
	end_timer_map["sampling randomized a/b"][0] = std::chrono::system_clock::now();
	printf("sampling randomized a/b done.\n");

	/*
	* Step 1-3: Generate c_i randomly
	* Here, c_i has the initial value, which is a_i * b_i
	*/
	printf("initialing c = ab.\n");
	start_timer_map["initialing c = ab"][0] = std::chrono::system_clock::now();
	{
		PlaintextVector<FD> tmp;
		tmp.resize(omp_get_max_threads(), FieldD);
		for(int i = 0; i < omp_get_max_threads(); i++){
			tmp[i].allocate_slots(FieldD.get_prime());
		}

		PlaintextMatrix<FD> tmp_b;
		tmp_b.resize(omp_get_max_threads());
		for(int i = 0; i < omp_get_max_threads(); i++){
			tmp_b[i].resize(column, FieldD);
			tmp_b[i].allocate_slots(FieldD.get_prime());
			tmp_b[i] = b;
		}

		#pragma omp parallel for
		for(int i = 0; i < row; i++){
			int num = omp_get_thread_num();
			c[i].mul(a[i][0], tmp_b[num][0]);

			for(int j = 1; j < column; j++){
				tmp[num].mul(a[i][j], tmp_b[num][j]);
				c[i] += tmp[num];
			}
		}
	}
	end_timer_map["initialing c = ab"][0] = std::chrono::system_clock::now();
	printf("initialing c = ab done.\n");

	/*
	* Step 2: Encrypt b_i and prepare to send it out
	*/

	printf("encrypting b.\n");
	start_timer_map["encrypting b"][0] = std::chrono::system_clock::now();
	Random_Coins rc(params);

	#pragma omp parallel for
	for(int i = 0; i < column; i++){
		int num = omp_get_thread_num();
		Random_Coins rc2(params);
		rc2.generate(G_array[num]);
		pks[my_num].encrypt(Cb[i], b[i], rc2);
	}
	end_timer_map["encrypting b"][0] = std::chrono::system_clock::now();
	printf("encrypting b done.\n");

	octetStream CbStream;
	int CbSize = Cb[0].report_size(USED);
	CbStream.resize_precise(column * CbSize);
	CbStream.reset_write_head();
	for(int i = 0; i < column; i++){
		Cb[i].pack(CbStream);
	}

	/*
	* Step 3: FFT a
	*/
	start_timer_map["FFTing a"][0] = std::chrono::system_clock::now();
	AddableMatrix<Rq_Element> a_mod_q;
	{
		a_mod_q.resize(row);
		for(int i = 0; i < row; i++){
			a_mod_q[i].resize(column, {params, evaluation, evaluation});
		}
		/* here, evaluation is a type, defined in FHE/Ring_Element.h */
	}

	#pragma omp parallel for
	for (int i = 0; i < row; i++){
		for(int j = 0; j < column; j++){
			a_mod_q[i][j].from_vec(a[i][j].get_poly());
		}
	}
	end_timer_map["FFTing a"][0] = std::chrono::system_clock::now();
	printf("FFTing a done.\n");

	/*
	* Step 4: Receive others' b_i in ciphertext, multiply it with a_ij, and return it back.
	*/
	PlaintextVector<FD> ed[num_players];
	for(int i = 0; i < P.num_players(); i++){
		ed[i].resize(row, FieldD);
		ed[i].allocate_slots(FieldD.get_prime());
		ed[i].assign_zero();
	}

	/* test decrypt */
	Plaintext_<FD> product_share(FieldD);
	product_share.allocate_slots(FieldD.get_prime());

	vector<octetStream> CbStream_others;
	CbStream_others.resize(num_players);

	vector<future<void>> res;
	for(int j = 1; j < num_players; j++){
		int party = (my_num + j) % num_players;
		int party_send = (my_num + num_players - j) % num_players;
		res.push_back(pool->enqueue([party, party_send, column, CbSize, &P, &CbStream, &CbStream_others]() {
			PRNG G;
			G.ReSeed();

			CbStream_others[party].resize_precise(column * CbSize);
			CbStream_others[party].reset_write_head();

			printf("receiving encrypted b from party %d.\n", party);
			{
				start_timer_map["receiving encrypted b"][party] = std::chrono::system_clock::now();
				P.comm_stats["Exchanging"].add(CbStream);
				CbStream.exchange(P.socket(party_send), P.socket(party), CbStream_others[party]);
				P.sent += CbStream.get_length();
				end_timer_map["receiving encrypted b"][party] = std::chrono::system_clock::now();
			}
			printf("receiving encrypted b from party %d done.\n", party);
		}));
	}
	joinNclean(res);

	for(int j = 1; j < num_players; j++){
		int party = (my_num + j) % num_players;
		int party_send = (my_num + num_players - j) % num_players;

		res.push_back(pool->enqueue([party, party_send, row, column, CbSize, &P, &CbStream_others, &c, &a_mod_q, &FieldD, &params, &pks, &ed, &psk]() {
			AddableVector<Ciphertext> Cb_other;
			Cb_other.resize(column, params);
			for(int i = 0; i < column; i++){
				Cb_other[i].unpack(CbStream_others[party]);
			}

			AddableVector<Ciphertext> tmp1, tmp2;
			tmp1.resize(row, params);
			tmp2.resize(row, params);

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
			for(int i = 0; i < row; i++){
				tmp1[i].mul(Cb_other[0], a_mod_q[i][0]);

				for(int j = 1; j < column; j++){
					tmp2[i].mul(Cb_other[j], a_mod_q[i][j]);
					tmp1[i] += tmp2[i];
				}

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

			octetStream CabStream;
			CabStream.resize_precise(row * CbSize);
			CabStream.reset_write_head();
			for(int i = 0; i < row; i++){
				tmp1[i].pack(CabStream);
			}

			printf("sending the final result with party %d.\n", party);
			start_timer_map["sending the final result"][party] = std::chrono::system_clock::now();
			{
				P.comm_stats["Exchanging"].add(CabStream);
				CabStream.exchange(P.socket(party), P.socket(party_send));
				P.sent += CabStream.get_length();
			}
			end_timer_map["sending the final result"][party] = std::chrono::system_clock::now();
			printf("sending the final result with party %d done.\n", party);

			for(int i = 0; i < row; i++){
				tmp1[i].unpack(CabStream);
			}

			printf("decrypting the final result with party %d.\n", party);
			start_timer_map["decrypting the final result"][party] = std::chrono::system_clock::now();
			for(int i = 0; i < row; i++){
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
	for(int i = 0; i < row; i++){
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

void check_first_result(PlaintextMatrix<FD> &a, PlaintextVector<FD> &b, PlaintextVector<FD> &c, int row, int column, const FHE_Params & params, const FD &FieldD, const PlainPlayer &P, ThreadPool * pool){
	(void)(params);

	int num_players = P.num_players();
	int my_num = P.my_num();

	/*
	* Print two rows.
	*/

	if(my_num != 0){
		octetStream abcStream;
		abcStream.resize_precise(a[0].report_size(USED) * 2 * column + b[0].report_size(USED) * column + c[0].report_size(USED) * 2);
		abcStream.reset_write_head();

		for(int i = 0; i < 2; i++){
			for(int j = 0; j < column; j++){
				a[i][j].pack(abcStream);
			}
		}

		for(int j = 0; j < column; j++){
			b[j].pack(abcStream);
		}

		for(int i = 0; i < 2; i++){
			c[i].pack(abcStream);
		}

		P.comm_stats["Sending directly"].add(abcStream);
		auto socket_send = P.socket(0);
		abcStream.Send(socket_send);
		P.sent += abcStream.get_length();
	}else{
		PlaintextMatrix<FD> a_other[num_players];
		PlaintextVector<FD> b_other[num_players];
		PlaintextVector<FD> c_other[num_players];

		for(int i = 0; i < P.num_players(); i++){
			a_other[i].resize(2);
			for(int k = 0; k < 2; k++){
				a_other[i][k].resize(column, FieldD);
			}

			b_other[i].resize(column, FieldD);
			c_other[i].resize(2, FieldD);

			a_other[i].allocate_slots(FieldD.get_prime());
			b_other[i].allocate_slots(FieldD.get_prime());
			c_other[i].allocate_slots((bigint)FieldD.get_prime() << 64);
		}

		for(int i = 0; i < 2; i++){
			a_other[0][i] = a[i];
		}
		for(int i = 0; i < column; i++){
			b_other[0][i] = b[i];
		}
		for(int i = 0; i < 2; i++){
			c_other[0][i] = c[i];
		}

		int test_matrix_size = a[0][0].report_size(USED) * 2 * column + b[0].report_size(USED) * column + c[0].report_size(USED) * 2;

		printf("Ready to receive data.\n");

		vector<future<void>> res;
		for(int j = 1; j < num_players; j++){
			int party = j;
			res.push_back(pool->enqueue([party, test_matrix_size, column, &P, &a_other, &b_other, &c_other](){
				octetStream abcStream_other;
				abcStream_other.resize_precise(test_matrix_size);
				abcStream_other.reset_write_head();

				auto socket_recv = P.socket(party);
				abcStream_other.Receive(socket_recv);

				for(int i = 0; i < 2; i++){
					for(int k = 0; k < column; k++){
						a_other[party][i][k].unpack(abcStream_other);
					}
				}
				for(int i = 0; i < column; i++){
					b_other[party][i].unpack(abcStream_other);
				}
				for(int i = 0; i < 2; i++){
					c_other[party][i].unpack(abcStream_other);
				}
			}));
		}
		joinNclean(res);

		printf("obtain all data from different parties.\n");

		PlaintextMatrix<FD> a_sum;
		PlaintextVector<FD> b_sum;
		PlaintextVector<FD> c_sum;
		PlaintextVector<FD> cc_sum;
		Plaintext_<FD> tmp(FieldD);

		{
			a_sum.resize(2);
			for(int i = 0; i < 2; i++){
				a_sum[i].resize(column, FieldD);
				a_sum[i].allocate_slots(FieldD.get_prime());
			}
		}
		{
			b_sum.resize(column, FieldD);
			b_sum.allocate_slots(FieldD.get_prime());
		}
		{
			c_sum.resize(row, FieldD);
			c_sum.allocate_slots((bigint)FieldD.get_prime() << 64);
		}
		{
			cc_sum.resize(row, FieldD);
			cc_sum.allocate_slots((bigint)FieldD.get_prime() << 64);
		}
		{
			tmp.allocate_slots((bigint)FieldD.get_prime() << 64);
		}

		a_sum = a_other[0];
		b_sum = b_other[0];
		c_sum = c_other[0];

		for(int i = 1; i < num_players; i++){
			a_sum += a_other[i];
			b_sum += b_other[i];
			c_sum += c_other[i];
		}

		printf("\n");

		for(int i = 0; i < 2; i++){
			c_sum[i].print_evaluation(2, "c_sum ");
		}

		for(int i = 0; i < 2; i++){
			cc_sum[i].mul(a_sum[i][0], b_sum[0]);

			for(int j = 1; j < column; j++){
				tmp.mul(a_sum[i][j], b_sum[j]);
				cc_sum[i] += tmp;
			}
		}

		for(int i = 0; i < 2; i++){
    	cc_sum[i].print_evaluation(2, "cc_sum ");
    }
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
		"16", // Default.
		0, // Required?
		1, // Number of args expected.
		0, // Delimiter if expecting multiple args.
		"Number of rows in the matrix.", // Help description.
		"-r", // Flag token.
		"--row" // Flag token.
	);
	opt.add(
		"10", // Default.
		0, // Required?
		1, // Number of args expected.
		0, // Delimiter if expecting multiple args.
		"Number of columns in the matrix.", // Help description.
		"-c", // Flag token.
		"--column" // Flag token.
	);
	opt.parse(argc, argv);

	// Input the simulation parameters
	int nplayers = 2;
	int plainlength = 64;
	int my_num = 0;
	int row = 16;
	int column = 10;

	string hostname;

	opt.get("-N")->getInt(nplayers);
	opt.get("-l")->getInt(plainlength);
	opt.get("-h")->getString(hostname);
	opt.get("-p")->getInt(my_num);
	opt.get("-r")->getInt(row);
	opt.get("-c")->getInt(column);

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
	PlaintextMatrix<FD> a;
	PlaintextVector<FD> b;
	PlaintextVector<FD> c;

	std::chrono::time_point<std::chrono::system_clock> start_time, end_time;
	start_time = std::chrono::system_clock::now();
	generate_one_batch(a, b, c, row, column, params, FieldD, P, pks, psk, &pool);
	end_time = std::chrono::system_clock::now();

	check_first_result(a, b, c, row, column, params, FieldD, P, &pool);

	cerr << "Time " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()) / 1000.0 << " seconds " << endl;
	cerr << "Generated " << a[0][0].num_slots() << " matrices of size "  << row  << " * " << column << endl;

	cerr << endl;
	cerr << "Rate: " << a[0][0].num_slots() / ((std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()) / 1000.0) << " matrices/second" << endl;

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
