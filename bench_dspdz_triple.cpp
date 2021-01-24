/*
* READ FIRST
*
* How to compile?
*   make bench_offline
*
* How to benchmark for two parties?
*   Assume Party 0's IP is x.x.x.x
*   Party 0: ./bench_dspdz_triple.x -N 2 -l 64 -h x.x.x.x -p 0
*   Party 1: ./bench_dspdz_triple.x -N 2 -l 64 -h x.x.x.x -p 1
*   (to support floating points with sufficient space, we need to set -l to a higher value)
*
* How to benchmark for three parties?
*   Party 0: ./bench_dspdz_triple.x -N 3 -l 64 -h x.x.x.x -p 0
*   Party 1: ./bench_dspdz_triple.x -N 3 -l 64 -h x.x.x.x -p 1
*   Party 2: ./bench_dspdz_triple.x -N 3 -l 64 -h x.x.x.x -p 2
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
#include <ctime>
#include <omp.h>
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
		generate_setup(P.num_players(), plainlength, 40, params, FieldD, 0, false);
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

void FHE_keygen(FHE_PK* &ppk, FHE_SK* &psk, const Player& P, int plainlength, FHE_Params &params, FD &FieldD){
	find_FHE_param(params, FieldD, P, plainlength);

	/*
	* Run the covertly secure protocol (with sec = 1) to generate the public key and private key shares
	* This part is going to take some time
	*/
	auto pk = new FHE_PK(params, FieldD.get_prime());
	auto sk = new FHE_SK(*pk);
	ppk = pk;
	psk = sk;

	start_timer_map["generating FHE keys (setup)"][0] = std::chrono::system_clock::now();
	Run_Gen_Protocol(*ppk, *psk, P, 1, false);
	end_timer_map["generating FHE keys (setup)"][0] = std::chrono::system_clock::now();
}

void network_setup(Names& N, const int nplayers, const int my_num, const string &hostname){
	cout << "My num: " << 12345 + my_num << endl;
	Server::start_networking(N, my_num, nplayers, hostname, 12345);
}

void generate_one_batch(PlaintextVector<FD> &a, PlaintextVector<FD> &b, PlaintextVector<FD> &c, const int batch_size, const FHE_Params & params, const FD &FieldD, const PlainPlayer &P, FHE_PK& pk, FHE_SK *psk, ThreadPool * pool, vector<int> distributed_coordinator){
	PRNG G;
	G.ReSeed();

	PlaintextVector<FD> f;

	AddableVector<Ciphertext> Ca;
	AddableVector<Ciphertext> Cb;
	AddableVector<Ciphertext> Cc;
	AddableVector<Ciphertext> Cf;
	int num_players = P.num_players();
	int my_num = P.my_num();

	/*
	* Step 1: Calculate the strategy
	*/
	vector<int> distributed_coordinator_batch_index_start;
	vector<int> distributed_coordinator_batch_index_end;

	{
		distributed_coordinator_batch_index_start.resize(num_players);
		distributed_coordinator_batch_index_end.resize(num_players);

		int current_start = 0;
		for(int p = 0; p < num_players; p++){
			int this_batch_size = distributed_coordinator[p];

			distributed_coordinator_batch_index_start[p] = current_start;
			current_start += this_batch_size;
			distributed_coordinator_batch_index_end[p] = current_start;
		}

		for(int p = 0; p < num_players; p++){
			int this_batch_size = distributed_coordinator[p];

			if(this_batch_size != 0){
				printf("Party %d is a distributed coordinator who is in charge of the batch indexed from %d to %d.\n",
					p,
					distributed_coordinator_batch_index_start[p],
					distributed_coordinator_batch_index_end[p]);
			}
		}
	}

	/*
	* Step 2: Generate a_i, b_i, f_i randomly
	*/
	printf("sampling randomized a/b/f.\n");
	start_timer_map["sampling randomized a/b/f"][0] = std::chrono::system_clock::now();
	{
		a.resize(batch_size, FieldD);
		b.resize(batch_size, FieldD);
		c.resize(batch_size, FieldD);
		f.resize(batch_size, FieldD);
		Ca.resize(batch_size, params);
		Cb.resize(batch_size, params);
		Cc.resize(batch_size, params);
		Cf.resize(batch_size, params);
	}

	{
		a.allocate_slots(FieldD.get_prime());
		b.allocate_slots(FieldD.get_prime());
		c.allocate_slots((bigint)FieldD.get_prime() << 64);
		f.allocate_slots(FieldD.get_prime());
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
			f[i].randomize(G_array[num]);
		}
	}
	end_timer_map["sampling randomized a/b/f"][0] = std::chrono::system_clock::now();
	printf("sampling randomized a/b/f done.\n");

	/*
	* Step 3: Encrypt a_i/b_i/f_i and prepare to send it out to the first party
	*/
	printf("encrypting a/b/f.\n");
	start_timer_map["encrypting a/b/f"][0] = std::chrono::system_clock::now();

	Random_Coins rc(params);

	#pragma omp parallel for
	for(int i = 0; i < batch_size; i++){
		Random_Coins rc2(params);
		int num = omp_get_thread_num();
		rc2.generate(G_array[num]);
		pk.encrypt(Ca[i], a[i], rc2);
		rc2.generate(G_array[num]);
		pk.encrypt(Cb[i], b[i], rc2);
		rc2.generate(G_array[num]);
		pk.encrypt(Cf[i], f[i], rc2);
	}
	end_timer_map["encrypting a/b/f"][0] = std::chrono::system_clock::now();
	printf("encrypting a/b/f done.\n");

	/*
	* Step 4: Prepare the stream for each distributed coordinator
	*/
	octetStream CaStream[num_players];
	octetStream CbStream[num_players];
	octetStream CfStream[num_players];
	int CaSize = Ca[0].report_size(USED);
	int CbSize = Cb[0].report_size(USED);
	int CfSize = Cf[0].report_size(USED);

	#pragma omp parallel for
	for(int p = 0; p < num_players; p++){
		if(p != my_num){
			if(distributed_coordinator[p] == 0){
				continue;
			}

			int start = distributed_coordinator_batch_index_start[p];
			int end = distributed_coordinator_batch_index_end[p];
			int this_batch_size = end - start;

			CaStream[p].resize_precise(this_batch_size * CaSize);
			CaStream[p].reset_write_head();
			for(int i = start; i < end; i++){
					Ca[i].pack(CaStream[p]);
			}

			CbStream[p].resize_precise(this_batch_size * CbSize);
			CbStream[p].reset_write_head();
			for(int i = start; i < end; i++){
					Cb[i].pack(CbStream[p]);
			}

			CfStream[p].resize_precise(this_batch_size * CfSize);
			CfStream[p].reset_write_head();
			for(int i = start; i < end; i++){
					Cf[i].pack(CfStream[p]);
			}
		}
	}

	/*
	* Step 3:
	* For the coordinator, create the threads
	*/
	start_timer_map["transfering a/b/f"][0] = std::chrono::system_clock::now();

	vector<future<void>> res;
	AddableVector<Ciphertext> Ca_others[num_players];
	AddableVector<Ciphertext> Cb_others[num_players];
	AddableVector<Ciphertext> Cf_others[num_players];

	octetStream CaStream_other[num_players];
	octetStream CbStream_other[num_players];
	octetStream CfStream_other[num_players];

	if(distributed_coordinator[my_num] != 0){
		int my_batch_size = distributed_coordinator[my_num];
		for(int p = 0; p < num_players; p++) {
			Ca_others[p].resize(my_batch_size, params);
			Cb_others[p].resize(my_batch_size, params);
			Cf_others[p].resize(my_batch_size, params);
		}

		int start = distributed_coordinator_batch_index_start[my_num];
		int end = distributed_coordinator_batch_index_end[my_num];
		for(int i = start, j = 0; i < end; i++, j++){
			Ca_others[my_num][j] = Ca[i];
		}
		for(int i = start, j = 0; i < end; i++, j++){
			Cb_others[my_num][j] = Cb[i];
		}
		for(int i = start, j = 0; i < end; i++, j++){
			Cf_others[my_num][j] = Cf[i];
		}

		for(int party = 0; party < num_players; party++){
			if(party == my_num){
				continue;
			}
			res.push_back(pool->enqueue([party, &P, my_batch_size, &CaStream_other, &CbStream_other, &CfStream_other, CaSize, CbSize, CfSize]() {
				CaStream_other[party].resize_precise(my_batch_size * CaSize);
				CaStream_other[party].reset_write_head();

				CbStream_other[party].resize_precise(my_batch_size * CbSize);
				CbStream_other[party].reset_write_head();

				CfStream_other[party].resize_precise(my_batch_size * CfSize);
				CfStream_other[party].reset_write_head();

				auto socket_recv = P.socket(party);
				CaStream_other[party].Receive(socket_recv);
				CbStream_other[party].Receive(socket_recv);
				CfStream_other[party].Receive(socket_recv);
			}));
		}
	}

	/*
	* Step 4:
	* Everyone sends the data out to the distributed coordinator
	*/

	{
		for(int party = 0; party < num_players; party++){
			if(party != my_num && distributed_coordinator[party] != 0){
				res.push_back(pool->enqueue([party, &P, &CaStream, &CbStream, &CfStream]() {
					auto socket_send = P.socket(party);
					CaStream[party].Send(socket_send);
					CbStream[party].Send(socket_send);
					CfStream[party].Send(socket_send);
				}));
			}
		}
	}

	Ca.clear();
	Cb.clear();
	Cf.clear();
	Ca.shrink_to_fit();
	Cb.shrink_to_fit();
	Cf.shrink_to_fit();

	joinNclean(res);
	end_timer_map["transfering a/b/f"][0] = std::chrono::system_clock::now();

	/*
	* Step 6:
	* Prepare to receive c + f
	*/
	start_timer_map["transfering c+f"][0] = std::chrono::system_clock::now();

	octetStream CcfStream[num_players];
	for(int p = 0; p < num_players; p++){
		if(p != my_num && distributed_coordinator[p] != 0){
			int start = distributed_coordinator_batch_index_start[p];
			int end = distributed_coordinator_batch_index_end[p];
			int this_batch_size = end - start;

			res.push_back(pool->enqueue([p, &P, &CcfStream, &Cc, start, end, this_batch_size]() {
				CcfStream[p].resize_precise(this_batch_size * Cc[0].report_size(USED));
				CcfStream[p].reset_write_head();

				printf("receiving masked c+f from %d.\n", p);
				auto socket_recv = P.socket(p);
				CcfStream[p].Receive(socket_recv);
				printf("receiving masked c+f from %d done.\n", p);

				for(int i = start; i < end; i++){
					Cc[i].unpack(CcfStream[p]);
				}
			}));
		}
	}
	// do not join here.

	/*
	* Step 5:
	* Distributed coordinators unpack, add, and mul the results
	* then, send the encrypted c+f to each other party
	*/

	octetStream CcfStream_send;
	if(distributed_coordinator[my_num] != 0){
		int start = distributed_coordinator_batch_index_start[my_num];
		int end = distributed_coordinator_batch_index_end[my_num];
		int this_batch_size = end - start;

		#pragma omp parallel for
		for(int party = 0; party < num_players; party++){
			if(party != my_num){
				for(int i = 0; i < this_batch_size; i++){
					Ca_others[party][i].unpack(CaStream_other[party]);
				}
				for(int i = 0; i < this_batch_size; i++){
					Cb_others[party][i].unpack(CbStream_other[party]);
				}
				for(int i = 0; i < this_batch_size; i++){
					Cf_others[party][i].unpack(CfStream_other[party]);
				}
			}
		}

		printf("adding encrypted a/b/f.\n");
		#pragma omp parallel for
		for(int i = 0; i < this_batch_size; i++){
			for(int j = 1; j < num_players; j++){
				add(Ca_others[0][i], Ca_others[0][i], Ca_others[j][i]);
				add(Cb_others[0][i], Cb_others[0][i], Cb_others[j][i]);
				add(Cf_others[0][i], Cf_others[0][i], Cf_others[j][i]);
			}
		}
		printf("adding encrypted a/b/f done.\n");

		printf("multiplying encrypted a/b.\n");
		#pragma omp parallel for
		for(int i = 0; i < this_batch_size; i++){
			mul(Cc[start + i], Ca_others[0][i], Cb_others[0][i], pk);
		}
		printf("multiplying encrypted a/b done.\n");

		printf("masking c with f.\n");
		#pragma omp parallel for
		for(int i = 0; i < this_batch_size; i++){
			if(Cc[start + i].level()==0){
				Cf_others[0][i].Scale(FieldD.get_prime());
			}
			add(Cc[start + i], Cf_others[0][i], Cc[start + i]);
		}
		printf("masking c with f done.\n");

		CcfStream_send.resize_precise(this_batch_size * Cc[0].report_size(USED));
		CcfStream_send.reset_write_head();

		for(int i = start; i < end; i++){
			Cc[i].pack(CcfStream_send);
		}

		printf("sending out masked c+f.\n");
		for(int p = 0; p < num_players; p++){
			if(p != my_num){
				res.push_back(pool->enqueue([p, &P, &CcfStream_send]() {
						auto sock_send = P.socket(p);
						CcfStream_send.Send(sock_send);
				}));
			}
		}
		printf("sending out masked c+f done.\n");
	}
	joinNclean(res);
	end_timer_map["transfering c+f"][0] = std::chrono::system_clock::now();

	/*
	* Step 6: distributed decryption to create vv from Cc
	*/
	AddableMatrix<bigint> vv;
	vv.resize(batch_size, pk.get_params().phi_m());
	bigint limit = pk.get_params().Q() << 64;
	vv.allocate_slots(limit);

	printf("making distributed decryption.\n");
	start_timer_map["making distributed decryption"][0] = std::chrono::system_clock::now();
	#pragma omp parallel for
	for(int i = 0; i < batch_size; i++){
		(*psk).dist_decrypt_1(vv[i], Cc[i], my_num, num_players);
	}
	printf("making distributed decryption done.\n");
	end_timer_map["making distributed decryption"][0] = std::chrono::system_clock::now();

	/*
	* Step 7: exchange the vv
	* first the distributed coordinators add threads to receive vv
	* second all parties compute and send out vv
	*/
	vector<AddableMatrix<bigint>> vv_others(num_players);
	vector<octetStream> vvStream_others(num_players);

	start_timer_map["exchanging vv"][0] = std::chrono::system_clock::now();
	printf("exchanging vv.\n");
	if(distributed_coordinator[my_num] != 0){
		vv_others.resize(num_players);
		int my_batch_size = distributed_coordinator_batch_index_end[my_num] - distributed_coordinator_batch_index_start[my_num];
		for(int p = 0; p < num_players; p++){
			vv_others[p].resize(my_batch_size, pk.get_params().phi_m());
			vv_others[p].allocate_slots(limit);
		}

		for(int p = 0; p < num_players; p++){
			if(p != my_num){
				vvStream_others[p].resize_precise(my_batch_size * vv[0].report_size(USED));
				vvStream_others[p].reset_write_head();

				res.push_back(pool->enqueue([p, &P, &vvStream_others]() {
					printf("receiving vv for distributed decryption from party %d.\n", p);
					auto socket_recv = P.socket(p);
					vvStream_others[p].Receive(socket_recv);
					printf("receiving vv for distributed decryption from party %d, done.\n", p);
				}));
			}
		}
	}
	/* do not join here */

	vector<octetStream> vvStream(num_players);

	for(int p = 0; p < num_players; p++){
		if(distributed_coordinator[p] != 0 && p != my_num){
			int start = distributed_coordinator_batch_index_start[p];
			int end = distributed_coordinator_batch_index_end[p];
			int this_batch_size = end - start;

			vvStream[p].resize_precise(this_batch_size * vv[0].report_size(USED));
			vvStream[p].reset_write_head();

			res.push_back(pool->enqueue([p, &P, &vv, start, end, &vvStream]() {
					for(int i = start; i < end; i++){
						vv[i].pack(vvStream[p]);
					}
					printf("sending vv for distributed decryption to party %d.\n", p);
					auto socket_send = P.socket(p);
					vvStream[p].Send(socket_send);
					printf("sending vv for distributed decryption to party %d done.\n", p);
			}));
		}
	}
	joinNclean(res);
	printf("exchanging vv done.\n");
	end_timer_map["exchanging vv"][0] = std::chrono::system_clock::now();

	/*
	* Step 8: understanding vv
	* if not my coordinated part, c is the negate f.
	* if my coordinated part, run dist decrypt 2 on vv1 (unpacked from vvStream_others).
	*/
	bool my_distributed_flag = distributed_coordinator[my_num] != 0;
	int my_start = distributed_coordinator_batch_index_start[my_num];
	int my_end = distributed_coordinator_batch_index_end[my_num];
	int my_batch_size = my_end - my_start;

	// unpack first
	if(my_distributed_flag){
		#pragma omp parallel for
		for (int p = 0; p < num_players; p++) {
			if(p != my_num){
				for(int i = 0; i < my_batch_size; i++){
					vv_others[p][i].unpack(vvStream_others[p]);
				}
			}
		}
	}

	printf("finalizing the distributed decryption.\n");
	#pragma omp parallel for
	for(int i = 0; i < batch_size; i++){
		if(my_distributed_flag && i >= my_start && i < my_end){
			for(int p = 0; p < num_players; p++){
				if(p != my_num){
					(*psk).dist_decrypt_2(vv[i], vv_others[p][i - my_start]);
				}
			}

			bigint mod = params.p0();
			c[i].set_poly_mod(vv[i], mod);
			sub(c[i], c[i], f[i]);
		}else{
			c[i] = f[i];
			c[i].negate();
		}
	}
	printf("finalizing the distributed decryption done.\n");

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
			//cout << "Mac share c: " << mac_share_c << endl;
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
		"-1", // Default.
		1, // Required?
		1, // Number of args expected.
		0, // Delimiter if expecting multiple args.
		"Last party's workload", // Help description.
		"-d", // Flag token.
		"--distributed" // Flag token.
	);
	opt.add(
		"1", // Default.
		1, // Required?
		1, // Number of args expected.
		0, // Delimiter if expecting multiple args.
		"Number of triples", // Help description
		"-t", // Flag token.
		"--triples" // Flag Token.
	);
	opt.parse(argc, argv);

	// Input the simulation parameters
	int nplayers = 2;
	int plainlength = 64;
	int my_num = 0;
	int batch_size = 40;
	int last_party = -1;
	int num_triples = -1;
	string hostname;

	opt.get("-N")->getInt(nplayers);
	opt.get("-l")->getInt(plainlength);
	opt.get("-h")->getString(hostname);
	opt.get("-p")->getInt(my_num);
	opt.get("-b")->getInt(batch_size);
	opt.get("-d")->getInt(last_party);
	opt.get("-t")->getInt(num_triples);
	Names N;
	network_setup(N, nplayers, my_num, hostname);

	PlainPlayer P(N, 0xffff << 16);

	FHE_PK *ppk;
	FHE_SK *psk;
	FHE_Params params;
	FD FieldD;

	FHE_keygen(ppk, psk, P, plainlength, params, FieldD);

	vector<int> distributed_coordinator(nplayers);
	if(last_party == -1){
		/* split the workload evenly */
		int ideal_each = (int)ceil(batch_size * 1.0 / nplayers);
		int remaining = batch_size;

		for(int p = 0; p < nplayers; p++){
	                if(remaining != 0){
	                        if(remaining > ideal_each){
	                                distributed_coordinator[p] = ideal_each;
	                                remaining = remaining - ideal_each;
	                        }else{
	                                distributed_coordinator[p] = remaining;
	                                remaining = 0;
        	                }
        	        }else{
        		        distributed_coordinator[p] = 0;
                	}
        	}
	}else{
		/* let the last party just do the assigned job */
		distributed_coordinator[nplayers - 1] = last_party;
		int ideal_each = (int)ceil((batch_size - last_party) * 1.0 / (nplayers - 1));
	       	int remaining = batch_size;

		for(int p = 0; p < nplayers - 1; p++){
                        if(remaining != 0){
                                if(remaining > ideal_each){
                                        distributed_coordinator[p] = ideal_each;
                                        remaining = remaining - ideal_each;
                                }else{
                                        distributed_coordinator[p] = remaining;
                                        remaining = 0;
                                }
                        }else{
                                distributed_coordinator[p] = 0;
                        }
                }
	}

	ThreadPool pool(64);
	PlaintextVector<FD> a;
	PlaintextVector<FD> b;
	PlaintextVector<FD> c;
	generate_one_batch(a, b, c, batch_size, params, FieldD, P, *ppk, psk, &pool, distributed_coordinator);
	int num_rounds = num_triples / (a[0].num_slots() * batch_size) + 1;
	std::chrono::time_point<std::chrono::system_clock> start_time, end_time;
	long total_time = 0;
	for (int i = 0; i < num_rounds; i++) {
		start_time = std::chrono::system_clock::now();
		generate_one_batch(a, b, c, batch_size, params, FieldD, P, *ppk, psk, &pool, distributed_coordinator);
		end_time = std::chrono::system_clock::now();
		total_time += (std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()) / 1000.0;  
		write_triples_to_file(a, b, c, batch_size, nplayers, FieldD, my_num);
	}

	check_first_result(a, b, c, params, FieldD, P, &pool);

	//cerr << "Time " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()) / 1000.0 << " seconds " << endl;
	//cerr << "Generated " << (a[0].num_slots()) * batch_size << " triplets." << endl;
	cerr << "Time " << total_time << " seconds " << endl;
	cerr << "Generated " << a[0].num_slots() * batch_size * num_rounds << " triples. " << endl;
	cerr << endl;
	cerr << "Rate: " << a[0].num_slots() * batch_size * num_rounds / total_time << " triples/second " << endl;
	// cerr << "Rate: " << (a[0].num_slots()) * batch_size * num_rounds / ((std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()) / 1000.0) << " triplets/second" << endl;

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


	
	/*
	ifstream infile;
	infile.open(file_name, ios::in);
	cout << "Input triples from file" << endl;
	
	for (int i = 0; i < 10; i++) {
		gfp temp;
		temp.init_field(FieldD.get_prime());
		temp.input(infile, false);
		cout << temp << endl;
	}
	*/
	cout << "Prime: " << FieldD.get_prime() << endl;
	
}
