/*
* READ FIRST
*
* How to compile?
*   make bench_offline
*
* How to benchmark for four parties?
*   Assume Party 0's IP is x.x.x.x
*   Party 0: ./bench_dspdz_triple_two_layer.x -N 4 -l 170 -h x.x.x.x -p 0
*   Party 1: ./bench_dspdz_triple_two_layer.x -N 4 -l 170 -h x.x.x.x -p 1
*   Party 2: ./bench_dspdz_triple_two_layer.x -N 4 -l 170 -h x.x.x.x -p 2
*   Party 3: ./bench_dspdz_triple_two_layer.x -N 4 -l 170 -h x.x.x.x -p 3
*   (to support floating points with sufficient space, we need to set -l to a higher value)
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
		generate_setup(P.num_players(), plainlength, 40, params, FieldD, 0, false);

		params.pack(o);
		FieldD.pack(o);

		P.send_all(o);
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

	Run_Gen_Protocol(*ppk, *psk, P, 1, false);
}

void network_setup(Names& N, const int nplayers, const int my_num, const string &hostname){
	Server::start_networking(N, my_num, nplayers, hostname, 12345);
}

void generate_one_batch(PlaintextVector<FD> &a, PlaintextVector<FD> &b, PlaintextVector<FD> &c, const int batch_size, const FHE_Params & params, const FD &FieldD, const PlainPlayer &P, FHE_PK& pk, FHE_SK *psk, ThreadPool * pool){
	PRNG G;
	G.ReSeed();

	PlaintextVector<FD> f;

	AddableVector<Ciphertext> Ca;
	AddableVector<Ciphertext> Cb;
	AddableVector<Ciphertext> Cc;
	AddableVector<Ciphertext> Cf;
	int num_players = P.num_players();
	int my_num = P.my_num();

	printf("Step 1: sampling randomized a/b/f.\n");
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
	printf("Step 1: sampling randomized a/b/f done.\n");

	printf("Step 2: encrypting a/b/f.\n");
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
	printf("Step 2: encrypting a/b/f.\n");

	printf("Step 3: exchanging a/b/f with parties on the same side.\n");
	vector<octetStream> CaStream_from_my_side(num_players);
	vector<octetStream> CbStream_from_my_side(num_players);
	vector<octetStream> CfStream_from_my_side(num_players);
	int CaSize = Ca[0].report_size(USED);
	int CbSize = Cb[0].report_size(USED);
	int CfSize = Cf[0].report_size(USED);

	vector<future<void>> res;

	if(my_num < 3){
		for(int p = 0; p < 3; p++){
			if(p == my_num) continue;

			res.push_back(pool->enqueue([p, &P, my_num, &CaStream_from_my_side, &CbStream_from_my_side, &CfStream_from_my_side, CaSize, CbSize, CfSize]() {
				CaStream_from_my_side[p].resize_precise(16 * CaSize);
				CaStream_from_my_side[p].reset_write_head();

				CbStream_from_my_side[p].resize_precise(16 * CbSize);
				CbStream_from_my_side[p].reset_write_head();

				CfStream_from_my_side[p].resize_precise(16 * CfSize);
				CfStream_from_my_side[p].reset_write_head();

				printf("-- receiving from party %d\n", p);
				auto socket_recv = P.socket(p);
				CaStream_from_my_side[p].Receive(socket_recv);
				CbStream_from_my_side[p].Receive(socket_recv);
				CfStream_from_my_side[p].Receive(socket_recv);
				printf("-- received from party %d\n", p);
			}));
		}
	}else{
		for(int p = 3; p < 4; p++){
			if(p == my_num) continue;

			res.push_back(pool->enqueue([p, &P, my_num, &CaStream_from_my_side, &CbStream_from_my_side, &CfStream_from_my_side, CaSize, CbSize, CfSize]() {
				CaStream_from_my_side[p].resize_precise(48 * CaSize);
				CaStream_from_my_side[p].reset_write_head();

				CbStream_from_my_side[p].resize_precise(48 * CbSize);
				CbStream_from_my_side[p].reset_write_head();

				CfStream_from_my_side[p].resize_precise(48 * CfSize);
				CfStream_from_my_side[p].reset_write_head();

				printf("-- receiving from party %d\n", p);
				auto socket_recv = P.socket(p);
				CaStream_from_my_side[p].Receive(socket_recv);
				CbStream_from_my_side[p].Receive(socket_recv);
				CfStream_from_my_side[p].Receive(socket_recv);
				printf("-- received from party %d\n", p);
			}));
		}
	}

	vector<octetStream> CaStream_to_my_side(num_players);
	vector<octetStream> CbStream_to_my_side(num_players);
	vector<octetStream> CfStream_to_my_side(num_players);

	if(my_num < 3){
		for(int p = 0; p < 3; p++){
			if(p == my_num) continue;

			res.push_back(pool->enqueue([p, &P, my_num, &CaStream_to_my_side, &CbStream_to_my_side, &CfStream_to_my_side, &Ca, &Cb, &Cf, CaSize, CbSize, CfSize]() {
				CaStream_to_my_side[p].resize_precise(16 * CaSize);
				CaStream_to_my_side[p].reset_write_head();

				CbStream_to_my_side[p].resize_precise(16 * CbSize);
				CbStream_to_my_side[p].reset_write_head();

				CfStream_to_my_side[p].resize_precise(16 * CfSize);
				CfStream_to_my_side[p].reset_write_head();

				int start = p * 16;
				int end = start + 16;

				for(int i = start; i < end; i++){
					Ca[i].pack(CaStream_to_my_side[p]);
					Cb[i].pack(CbStream_to_my_side[p]);
					Cf[i].pack(CfStream_to_my_side[p]);
				}

				printf("-- sending to party %d\n", p);
				auto socket_send = P.socket(p);
				CaStream_to_my_side[p].Send(socket_send);
				CbStream_to_my_side[p].Send(socket_send);
				CfStream_to_my_side[p].Send(socket_send);
				printf("-- sent to party %d\n", p);
			}));
		}
	}else{
		for(int p = 3; p < 4; p++){
			if(p == my_num) continue;

			res.push_back(pool->enqueue([p, &P, my_num, &CaStream_to_my_side, &CbStream_to_my_side, &CfStream_to_my_side, &Ca, &Cb, &Cf, CaSize, CbSize, CfSize]() {
				CaStream_to_my_side[p].resize_precise(48 * CaSize);
				CaStream_to_my_side[p].reset_write_head();

				CbStream_to_my_side[p].resize_precise(48 * CbSize);
				CbStream_to_my_side[p].reset_write_head();

				CfStream_to_my_side[p].resize_precise(48 * CfSize);
				CfStream_to_my_side[p].reset_write_head();

				int start = 3 * 16 + (p - 3) * 48;
				int end = start + 48;

				for(int i = start; i < end; i++){
					Ca[i].pack(CaStream_to_my_side[p]);
					Cb[i].pack(CbStream_to_my_side[p]);
					Cf[i].pack(CfStream_to_my_side[p]);
				}

				printf("-- sending to party %d\n", p);
				auto socket_send = P.socket(p);
				CaStream_to_my_side[p].Send(socket_send);
				CbStream_to_my_side[p].Send(socket_send);
				CfStream_to_my_side[p].Send(socket_send);
				printf("-- sent to party %d\n", p);
			}));
		}
	}
	joinNclean(res);

	printf("Step 3: exchanging a/b/f with parties on the same side done.\n");

	int my_shadow_job_start;
	if(my_num < 3){
		my_shadow_job_start = 3 * 16 + my_num * 16;
	}else{
		my_shadow_job_start = (my_num - 3) * 48;
	}

	vector<octetStream> CaStream_from_my_side_for_shadowing(num_players);
	vector<octetStream> CbStream_from_my_side_for_shadowing(num_players);
	vector<octetStream> CfStream_from_my_side_for_shadowing(num_players);

	printf("Step 4: sending a/b/f to shadows of the other side.\n");
	if(my_num < 3){
		for(int p = 0; p < 3; p++){
			if(p == my_num) continue;

			res.push_back(pool->enqueue([p, &P, my_num, &CaStream_from_my_side_for_shadowing, &CbStream_from_my_side_for_shadowing, &CfStream_from_my_side_for_shadowing, CaSize, CbSize, CfSize]() {
				CaStream_from_my_side_for_shadowing[p].resize_precise(16 * CaSize);
				CaStream_from_my_side_for_shadowing[p].reset_write_head();

				CbStream_from_my_side_for_shadowing[p].resize_precise(16 * CbSize);
				CbStream_from_my_side_for_shadowing[p].reset_write_head();

				CfStream_from_my_side_for_shadowing[p].resize_precise(16 * CfSize);
				CfStream_from_my_side_for_shadowing[p].reset_write_head();

				printf("-- receiving from party %d\n", p);
				auto socket_recv = P.socket(p);
				CaStream_from_my_side_for_shadowing[p].Receive(socket_recv);
				CbStream_from_my_side_for_shadowing[p].Receive(socket_recv);
				CfStream_from_my_side_for_shadowing[p].Receive(socket_recv);
				printf("-- received from party %d\n", p);
			}));
		}
	}else{
		for(int p = 3; p < 4; p++){
			if(p == my_num) continue;

			res.push_back(pool->enqueue([p, &P, my_num, &CaStream_from_my_side_for_shadowing, &CbStream_from_my_side_for_shadowing, &CfStream_from_my_side_for_shadowing, CaSize, CbSize, CfSize]() {
				CaStream_from_my_side_for_shadowing[p].resize_precise(48 * CaSize);
				CaStream_from_my_side_for_shadowing[p].reset_write_head();

				CbStream_from_my_side_for_shadowing[p].resize_precise(48 * CbSize);
				CbStream_from_my_side_for_shadowing[p].reset_write_head();

				CfStream_from_my_side_for_shadowing[p].resize_precise(48 * CfSize);
				CfStream_from_my_side_for_shadowing[p].reset_write_head();

				printf("-- receiving from party %d\n", p);
				auto socket_recv = P.socket(p);
				CaStream_from_my_side_for_shadowing[p].Receive(socket_recv);
				CbStream_from_my_side_for_shadowing[p].Receive(socket_recv);
				CfStream_from_my_side_for_shadowing[p].Receive(socket_recv);
				printf("-- received from party %d\n", p);
			}));
		}
	}

	vector<octetStream> CaStream_to_my_side_for_shadowing(num_players);
	vector<octetStream> CbStream_to_my_side_for_shadowing(num_players);
	vector<octetStream> CfStream_to_my_side_for_shadowing(num_players);

	if(my_num < 3){
		for(int p = 0; p < 3; p++){
			if(p == my_num) continue;

			res.push_back(pool->enqueue([p, &P, &Ca, &Cb, &Cf, my_num, &CaStream_to_my_side_for_shadowing,
				&CbStream_to_my_side_for_shadowing, &CfStream_to_my_side_for_shadowing, CaSize, CbSize, CfSize]() {
				CaStream_to_my_side_for_shadowing[p].resize_precise(16 * CaSize);
				CaStream_to_my_side_for_shadowing[p].reset_write_head();

				CbStream_to_my_side_for_shadowing[p].resize_precise(16 * CbSize);
				CbStream_to_my_side_for_shadowing[p].reset_write_head();

				CfStream_to_my_side_for_shadowing[p].resize_precise(16 * CfSize);
				CfStream_to_my_side_for_shadowing[p].reset_write_head();

				int start = p * 16;
				int end = start + 16;

				for(int i = start; i < end; i++){
					Ca[i].pack(CaStream_to_my_side_for_shadowing[p]);
					Cb[i].pack(CbStream_to_my_side_for_shadowing[p]);
					Cf[i].pack(CfStream_to_my_side_for_shadowing[p]);
				}

				printf("-- sending to party %d\n", p);
				auto socket_send = P.socket(p);
				CaStream_to_my_side_for_shadowing[p].Send(socket_send);
				CbStream_to_my_side_for_shadowing[p].Send(socket_send);
				CfStream_to_my_side_for_shadowing[p].Send(socket_send);
				printf("-- sent to party %d\n", p);
			}));
		}
	}else{
		for(int p = 3; p < 4; p++){
			if(p == my_num) continue;

			res.push_back(pool->enqueue([p, &P, &Ca, &Cb, &Cf, my_num, &CaStream_to_my_side_for_shadowing,
				 &CbStream_to_my_side_for_shadowing, &CfStream_to_my_side_for_shadowing, CaSize, CbSize, CfSize]() {
				CaStream_to_my_side_for_shadowing[p].resize_precise(48 * CaSize);
				CaStream_to_my_side_for_shadowing[p].reset_write_head();

				CbStream_to_my_side_for_shadowing[p].resize_precise(48 * CbSize);
				CbStream_to_my_side_for_shadowing[p].reset_write_head();

				CfStream_to_my_side_for_shadowing[p].resize_precise(48 * CfSize);
				CfStream_to_my_side_for_shadowing[p].reset_write_head();

				int start = (p - 3) * 48;
				int end = start + 48;

				for(int i = start; i < end; i++){
					Ca[i].pack(CaStream_to_my_side_for_shadowing[p]);
					Cb[i].pack(CbStream_to_my_side_for_shadowing[p]);
					Cf[i].pack(CfStream_to_my_side_for_shadowing[p]);
				}

				printf("-- sending to party %d\n", p);
				auto socket_send = P.socket(p);
				CaStream_to_my_side_for_shadowing[p].Send(socket_send);
				CbStream_to_my_side_for_shadowing[p].Send(socket_send);
				CfStream_to_my_side_for_shadowing[p].Send(socket_send);
				printf("-- sent to party %d\n", p);
			}));
		}
	}
	joinNclean(res);
	printf("Step 4: sending a/b/f to shadows of the other side done.\n");

	printf("Step 5: shadows add the data.\n");
	vector<AddableVector<Ciphertext>> Ca_from_my_side_for_shadowing(num_players);
	vector<AddableVector<Ciphertext>> Cb_from_my_side_for_shadowing(num_players);
	vector<AddableVector<Ciphertext>> Cf_from_my_side_for_shadowing(num_players);

	if(my_num < 3){
		#pragma omp parallel for
		for(int p = 0; p < 3; p++){
			if(p == my_num) continue;

			Ca_from_my_side_for_shadowing[p].resize(16, params);
			Cb_from_my_side_for_shadowing[p].resize(16, params);
			Cf_from_my_side_for_shadowing[p].resize(16, params);

			for(int i = 0; i < 16; i++){
				Ca_from_my_side_for_shadowing[p][i].unpack(CaStream_from_my_side_for_shadowing[p]);
				Cb_from_my_side_for_shadowing[p][i].unpack(CbStream_from_my_side_for_shadowing[p]);
				Cf_from_my_side_for_shadowing[p][i].unpack(CfStream_from_my_side_for_shadowing[p]);
			}
		}

		Ca_from_my_side_for_shadowing[my_num].resize(16, params);
		Cb_from_my_side_for_shadowing[my_num].resize(16, params);
		Cf_from_my_side_for_shadowing[my_num].resize(16, params);

		for(int i = 0; i < 16; i++){
			Ca_from_my_side_for_shadowing[my_num][i] = Ca[i + my_shadow_job_start];
			Cb_from_my_side_for_shadowing[my_num][i] = Cb[i + my_shadow_job_start];
			Cf_from_my_side_for_shadowing[my_num][i] = Cf[i + my_shadow_job_start];
		}

		#pragma omp parallel for
		for(int i = 0; i < 16; i++){
			for(int p = 0 + 1; p < 3; p++){
				add(Ca_from_my_side_for_shadowing[0][i], Ca_from_my_side_for_shadowing[0][i], Ca_from_my_side_for_shadowing[p][i]);
				add(Cb_from_my_side_for_shadowing[0][i], Cb_from_my_side_for_shadowing[0][i], Cb_from_my_side_for_shadowing[p][i]);
				add(Cf_from_my_side_for_shadowing[0][i], Cf_from_my_side_for_shadowing[0][i], Cf_from_my_side_for_shadowing[p][i]);
			}
		}
	}else{
		#pragma omp parallel for
		for(int p = 3; p < 4; p++){
			if(p == my_num) continue;

			Ca_from_my_side_for_shadowing[p].resize(48, params);
			Cb_from_my_side_for_shadowing[p].resize(48, params);
			Cf_from_my_side_for_shadowing[p].resize(48, params);

			for(int i = 0; i < 48; i++){
				Ca_from_my_side_for_shadowing[p][i].unpack(CaStream_from_my_side_for_shadowing[p]);
				Cb_from_my_side_for_shadowing[p][i].unpack(CbStream_from_my_side_for_shadowing[p]);
				Cf_from_my_side_for_shadowing[p][i].unpack(CfStream_from_my_side_for_shadowing[p]);
			}
		}

		Ca_from_my_side_for_shadowing[my_num].resize(48, params);
		Cb_from_my_side_for_shadowing[my_num].resize(48, params);
		Cf_from_my_side_for_shadowing[my_num].resize(48, params);

		for(int i = 0; i < 48; i++){
			Ca_from_my_side_for_shadowing[my_num][i] = Ca[i + my_shadow_job_start];
			Cb_from_my_side_for_shadowing[my_num][i] = Cb[i + my_shadow_job_start];
			Cf_from_my_side_for_shadowing[my_num][i] = Cf[i + my_shadow_job_start];
		}

		#pragma omp parallel for
		for(int i = 0; i < 48; i++){
			for(int p = 3 + 1; p < 4; p++){
				add(Ca_from_my_side_for_shadowing[3][i], Ca_from_my_side_for_shadowing[3][i], Ca_from_my_side_for_shadowing[p][i]);
				add(Cb_from_my_side_for_shadowing[3][i], Cb_from_my_side_for_shadowing[3][i], Cb_from_my_side_for_shadowing[p][i]);
				add(Cf_from_my_side_for_shadowing[3][i], Cf_from_my_side_for_shadowing[3][i], Cf_from_my_side_for_shadowing[p][i]);
			}
		}
	}
	printf("Step 5: shadows add the data done.\n");


	vector<octetStream> CaStream_from_my_shadow_about_shadowing(num_players);
	vector<octetStream> CbStream_from_my_shadow_about_shadowing(num_players);
	vector<octetStream> CfStream_from_my_shadow_about_shadowing(num_players);

	printf("Step 6: shadows send aggregated data to the boss(es).\n");
	if(my_num < 3){
		int my_shadow_num_1 = 3 + (my_num / 3);

		res.push_back(pool->enqueue([&P, my_shadow_num_1, &CaStream_from_my_shadow_about_shadowing, &CbStream_from_my_shadow_about_shadowing, &CfStream_from_my_shadow_about_shadowing, CaSize, CbSize, CfSize]() {
			CaStream_from_my_shadow_about_shadowing[my_shadow_num_1].resize_precise(16 * CaSize);
			CaStream_from_my_shadow_about_shadowing[my_shadow_num_1].reset_write_head();

			CbStream_from_my_shadow_about_shadowing[my_shadow_num_1].resize_precise(16 * CbSize);
			CbStream_from_my_shadow_about_shadowing[my_shadow_num_1].reset_write_head();

			CfStream_from_my_shadow_about_shadowing[my_shadow_num_1].resize_precise(16 * CfSize);
			CfStream_from_my_shadow_about_shadowing[my_shadow_num_1].reset_write_head();

			printf("-- receiving data from shadow %d\n", my_shadow_num_1);
			auto socket_recv = P.socket(my_shadow_num_1);
			CaStream_from_my_shadow_about_shadowing[my_shadow_num_1].Receive(socket_recv);
			CbStream_from_my_shadow_about_shadowing[my_shadow_num_1].Receive(socket_recv);
			CfStream_from_my_shadow_about_shadowing[my_shadow_num_1].Receive(socket_recv);
			printf("-- received data from shadow %d\n", my_shadow_num_1);
		}));
	}else{
		int my_shadow_num_1 = (my_num - 3) * 3 + 0;
		int my_shadow_num_2 = (my_num - 3) * 3 + 1;
		int my_shadow_num_3 = (my_num - 3) * 3 + 2;

		res.push_back(pool->enqueue([&P, my_shadow_num_1, &CaStream_from_my_shadow_about_shadowing, &CbStream_from_my_shadow_about_shadowing, &CfStream_from_my_shadow_about_shadowing, CaSize, CbSize, CfSize]() {
			CaStream_from_my_shadow_about_shadowing[my_shadow_num_1].resize_precise(16 * CaSize);
			CaStream_from_my_shadow_about_shadowing[my_shadow_num_1].reset_write_head();

			CbStream_from_my_shadow_about_shadowing[my_shadow_num_1].resize_precise(16 * CbSize);
			CbStream_from_my_shadow_about_shadowing[my_shadow_num_1].reset_write_head();

			CfStream_from_my_shadow_about_shadowing[my_shadow_num_1].resize_precise(16 * CfSize);
			CfStream_from_my_shadow_about_shadowing[my_shadow_num_1].reset_write_head();

			printf("-- receiving data from shadow %d\n", my_shadow_num_1);
			auto socket_recv = P.socket(my_shadow_num_1);
			CaStream_from_my_shadow_about_shadowing[my_shadow_num_1].Receive(socket_recv);
			CbStream_from_my_shadow_about_shadowing[my_shadow_num_1].Receive(socket_recv);
			CfStream_from_my_shadow_about_shadowing[my_shadow_num_1].Receive(socket_recv);
			printf("-- received data from shadow %d\n", my_shadow_num_1);
		}));

		res.push_back(pool->enqueue([&P, my_shadow_num_2, &CaStream_from_my_shadow_about_shadowing, &CbStream_from_my_shadow_about_shadowing, &CfStream_from_my_shadow_about_shadowing, CaSize, CbSize, CfSize]() {
			CaStream_from_my_shadow_about_shadowing[my_shadow_num_2].resize_precise(16 * CaSize);
			CaStream_from_my_shadow_about_shadowing[my_shadow_num_2].reset_write_head();

			CbStream_from_my_shadow_about_shadowing[my_shadow_num_2].resize_precise(16 * CbSize);
			CbStream_from_my_shadow_about_shadowing[my_shadow_num_2].reset_write_head();

			CfStream_from_my_shadow_about_shadowing[my_shadow_num_2].resize_precise(16 * CfSize);
			CfStream_from_my_shadow_about_shadowing[my_shadow_num_2].reset_write_head();

			printf("-- receiving data from shadow %d\n", my_shadow_num_2);
			auto socket_recv = P.socket(my_shadow_num_2);
			CaStream_from_my_shadow_about_shadowing[my_shadow_num_2].Receive(socket_recv);
			CbStream_from_my_shadow_about_shadowing[my_shadow_num_2].Receive(socket_recv);
			CfStream_from_my_shadow_about_shadowing[my_shadow_num_2].Receive(socket_recv);
			printf("-- received data from shadow %d\n", my_shadow_num_2);
		}));

		res.push_back(pool->enqueue([&P, my_shadow_num_3, &CaStream_from_my_shadow_about_shadowing, &CbStream_from_my_shadow_about_shadowing, &CfStream_from_my_shadow_about_shadowing, CaSize, CbSize, CfSize]() {
			CaStream_from_my_shadow_about_shadowing[my_shadow_num_3].resize_precise(16 * CaSize);
			CaStream_from_my_shadow_about_shadowing[my_shadow_num_3].reset_write_head();

			CbStream_from_my_shadow_about_shadowing[my_shadow_num_3].resize_precise(16 * CbSize);
			CbStream_from_my_shadow_about_shadowing[my_shadow_num_3].reset_write_head();

			CfStream_from_my_shadow_about_shadowing[my_shadow_num_3].resize_precise(16 * CfSize);
			CfStream_from_my_shadow_about_shadowing[my_shadow_num_3].reset_write_head();

			printf("-- receiving data from shadow %d\n", my_shadow_num_3);
			auto socket_recv = P.socket(my_shadow_num_3);
			CaStream_from_my_shadow_about_shadowing[my_shadow_num_3].Receive(socket_recv);
			CbStream_from_my_shadow_about_shadowing[my_shadow_num_3].Receive(socket_recv);
			CfStream_from_my_shadow_about_shadowing[my_shadow_num_3].Receive(socket_recv);
			printf("-- received data from shadow %d\n", my_shadow_num_3);
		}));
	}

	vector<octetStream> CaStream_to_my_boss_about_shadowing(num_players);
	vector<octetStream> CbStream_to_my_boss_about_shadowing(num_players);
	vector<octetStream> CfStream_to_my_boss_about_shadowing(num_players);

	if(my_num < 3){
		int my_boss_num_1 = 3 + (my_num / 3);

		CaStream_to_my_boss_about_shadowing[my_boss_num_1].resize_precise(16 * CaSize);
		CaStream_to_my_boss_about_shadowing[my_boss_num_1].reset_write_head();

		CbStream_to_my_boss_about_shadowing[my_boss_num_1].resize_precise(16 * CbSize);
		CbStream_to_my_boss_about_shadowing[my_boss_num_1].reset_write_head();

		CfStream_to_my_boss_about_shadowing[my_boss_num_1].resize_precise(16 * CfSize);
		CfStream_to_my_boss_about_shadowing[my_boss_num_1].reset_write_head();

		for(int i = 0; i < 16; i++){
			Ca_from_my_side_for_shadowing[0][i].pack(CaStream_to_my_boss_about_shadowing[my_boss_num_1]);
			Cb_from_my_side_for_shadowing[0][i].pack(CbStream_to_my_boss_about_shadowing[my_boss_num_1]);
			Cf_from_my_side_for_shadowing[0][i].pack(CfStream_to_my_boss_about_shadowing[my_boss_num_1]);
		}

		printf("-- sending data to boss %d\n", my_boss_num_1);
		auto sock_send = P.socket(my_boss_num_1);
		CaStream_to_my_boss_about_shadowing[my_boss_num_1].Send(sock_send);
		CbStream_to_my_boss_about_shadowing[my_boss_num_1].Send(sock_send);
		CfStream_to_my_boss_about_shadowing[my_boss_num_1].Send(sock_send);
		printf("-- sent data to boss %d\n", my_boss_num_1);
	}else{
		int my_boss_num_1 = (my_num - 3) * 3 + 0;
		int my_boss_num_2 = (my_num - 3) * 3 + 1;
		int my_boss_num_3 = (my_num - 3) * 3 + 2;

		CaStream_to_my_boss_about_shadowing[my_boss_num_1].resize_precise(16 * CaSize);
		CaStream_to_my_boss_about_shadowing[my_boss_num_1].reset_write_head();
		CbStream_to_my_boss_about_shadowing[my_boss_num_1].resize_precise(16 * CbSize);
		CbStream_to_my_boss_about_shadowing[my_boss_num_1].reset_write_head();
		CfStream_to_my_boss_about_shadowing[my_boss_num_1].resize_precise(16 * CfSize);
		CfStream_to_my_boss_about_shadowing[my_boss_num_1].reset_write_head();

		CaStream_to_my_boss_about_shadowing[my_boss_num_2].resize_precise(16 * CaSize);
		CaStream_to_my_boss_about_shadowing[my_boss_num_2].reset_write_head();
		CbStream_to_my_boss_about_shadowing[my_boss_num_2].resize_precise(16 * CbSize);
		CbStream_to_my_boss_about_shadowing[my_boss_num_2].reset_write_head();
		CfStream_to_my_boss_about_shadowing[my_boss_num_2].resize_precise(16 * CfSize);
		CfStream_to_my_boss_about_shadowing[my_boss_num_2].reset_write_head();

		CaStream_to_my_boss_about_shadowing[my_boss_num_3].resize_precise(16 * CaSize);
		CaStream_to_my_boss_about_shadowing[my_boss_num_3].reset_write_head();
		CbStream_to_my_boss_about_shadowing[my_boss_num_3].resize_precise(16 * CbSize);
		CbStream_to_my_boss_about_shadowing[my_boss_num_3].reset_write_head();
		CfStream_to_my_boss_about_shadowing[my_boss_num_3].resize_precise(16 * CfSize);
		CfStream_to_my_boss_about_shadowing[my_boss_num_3].reset_write_head();

		#pragma omp parallel for
		for(int t = 0; t < 3; t++){
			if(t == 0){
				for(int i = 0; i < 16; i++){
					Ca_from_my_side_for_shadowing[3][i].pack(CaStream_to_my_boss_about_shadowing[my_boss_num_1]);
					Cb_from_my_side_for_shadowing[3][i].pack(CbStream_to_my_boss_about_shadowing[my_boss_num_1]);
					Cf_from_my_side_for_shadowing[3][i].pack(CfStream_to_my_boss_about_shadowing[my_boss_num_1]);
				}

				printf("-- sending data to boss %d\n", my_boss_num_1);
				auto sock_send_1 = P.socket(my_boss_num_1);
				CaStream_to_my_boss_about_shadowing[my_boss_num_1].Send(sock_send_1);
				CbStream_to_my_boss_about_shadowing[my_boss_num_1].Send(sock_send_1);
				CfStream_to_my_boss_about_shadowing[my_boss_num_1].Send(sock_send_1);
				printf("-- sent data to boss %d\n", my_boss_num_1);
			}

			if(t == 1){
				for(int i = 16; i < 32; i++){
					Ca_from_my_side_for_shadowing[3][i].pack(CaStream_to_my_boss_about_shadowing[my_boss_num_2]);
					Cb_from_my_side_for_shadowing[3][i].pack(CbStream_to_my_boss_about_shadowing[my_boss_num_2]);
					Cf_from_my_side_for_shadowing[3][i].pack(CfStream_to_my_boss_about_shadowing[my_boss_num_2]);
				}

				printf("-- sending data to boss %d\n", my_boss_num_2);
				auto sock_send_2 = P.socket(my_boss_num_2);
				CaStream_to_my_boss_about_shadowing[my_boss_num_2].Send(sock_send_2);
				CbStream_to_my_boss_about_shadowing[my_boss_num_2].Send(sock_send_2);
				CfStream_to_my_boss_about_shadowing[my_boss_num_2].Send(sock_send_2);
				printf("-- sent data to boss %d\n", my_boss_num_2);
			}

			if(t == 2){
				for(int i = 32; i < 48; i++){
					Ca_from_my_side_for_shadowing[3][i].pack(CaStream_to_my_boss_about_shadowing[my_boss_num_3]);
					Cb_from_my_side_for_shadowing[3][i].pack(CbStream_to_my_boss_about_shadowing[my_boss_num_3]);
					Cf_from_my_side_for_shadowing[3][i].pack(CfStream_to_my_boss_about_shadowing[my_boss_num_3]);
				}

				printf("-- sending data to boss %d\n", my_boss_num_3);
				auto sock_send_3 = P.socket(my_boss_num_3);
				CaStream_to_my_boss_about_shadowing[my_boss_num_3].Send(sock_send_3);
				CbStream_to_my_boss_about_shadowing[my_boss_num_3].Send(sock_send_3);
				CfStream_to_my_boss_about_shadowing[my_boss_num_3].Send(sock_send_3);
				printf("-- sent data to boss %d\n", my_boss_num_3);
			}
		}
	}
	joinNclean(res);
	printf("Step 6: shadows send aggregated data to the boss(es) done.\n");

	vector<AddableVector<Ciphertext>> Ca_from_my_side_directly(num_players);
	vector<AddableVector<Ciphertext>> Cb_from_my_side_directly(num_players);
	vector<AddableVector<Ciphertext>> Cf_from_my_side_directly(num_players);

	vector<AddableVector<Ciphertext>> Ca_from_my_shadow(num_players);
	vector<AddableVector<Ciphertext>> Cb_from_my_shadow(num_players);
	vector<AddableVector<Ciphertext>> Cf_from_my_shadow(num_players);

	printf("Step 7: add my data + my side data + data from shadow.\n");
	if(my_num < 3){
		int my_shadow_num_1 = 3 + (my_num / 3);

		#pragma omp parallel for
		for(int p = 0; p < 3; p++){
			if(p == my_num) continue;

			Ca_from_my_side_directly[p].resize(16, params);
			Cb_from_my_side_directly[p].resize(16, params);
			Cf_from_my_side_directly[p].resize(16, params);

			for(int i = 0; i < 16; i++){
				Ca_from_my_side_directly[p][i].unpack(CaStream_from_my_side[p]);
				Cb_from_my_side_directly[p][i].unpack(CbStream_from_my_side[p]);
				Cf_from_my_side_directly[p][i].unpack(CfStream_from_my_side[p]);
			}
		}

		Ca_from_my_side_directly[my_num].resize(16, params);
		Cb_from_my_side_directly[my_num].resize(16, params);
		Cf_from_my_side_directly[my_num].resize(16, params);

		int my_job_start = 16 * my_num;
		for(int i = 0; i < 16; i++){
			Ca_from_my_side_directly[my_num][i] = Ca[i + my_job_start];
			Cb_from_my_side_directly[my_num][i] = Cb[i + my_job_start];
			Cf_from_my_side_directly[my_num][i] = Cf[i + my_job_start];
		}

		#pragma omp parallel for
		for(int i = 0; i < 16; i++){
			for(int p = 0 + 1; p < 3; p++){
				add(Ca_from_my_side_directly[0][i], Ca_from_my_side_directly[0][i], Ca_from_my_side_directly[p][i]);
				add(Cb_from_my_side_directly[0][i], Cb_from_my_side_directly[0][i], Cb_from_my_side_directly[p][i]);
				add(Cf_from_my_side_directly[0][i], Cf_from_my_side_directly[0][i], Cf_from_my_side_directly[p][i]);
			}
		}

		Ca_from_my_shadow[my_shadow_num_1].resize(16, params);
		Cb_from_my_shadow[my_shadow_num_1].resize(16, params);
		Cf_from_my_shadow[my_shadow_num_1].resize(16, params);

		for(int i = 0; i < 16; i++){
			Ca_from_my_shadow[my_shadow_num_1][i].unpack(CaStream_from_my_shadow_about_shadowing[my_shadow_num_1]);
			Cb_from_my_shadow[my_shadow_num_1][i].unpack(CbStream_from_my_shadow_about_shadowing[my_shadow_num_1]);
			Cf_from_my_shadow[my_shadow_num_1][i].unpack(CfStream_from_my_shadow_about_shadowing[my_shadow_num_1]);
		}

		for(int i = 0; i < 16; i++){
			add(Ca_from_my_side_directly[0][i], Ca_from_my_side_directly[0][i], Ca_from_my_shadow[my_shadow_num_1][i]);
			add(Cb_from_my_side_directly[0][i], Cb_from_my_side_directly[0][i], Cb_from_my_shadow[my_shadow_num_1][i]);
			add(Cf_from_my_side_directly[0][i], Cf_from_my_side_directly[0][i], Cf_from_my_shadow[my_shadow_num_1][i]);
		}
	}else{
		int my_shadow_num_1 = (my_num - 3) * 3 + 0;
		int my_shadow_num_2 = (my_num - 3) * 3 + 1;
		int my_shadow_num_3 = (my_num - 3) * 3 + 2;

		#pragma omp parallel for
		for(int p = 3; p < 4; p++){
			if(p == my_num) continue;

			Ca_from_my_side_directly[p].resize(48, params);
			Cb_from_my_side_directly[p].resize(48, params);
			Cf_from_my_side_directly[p].resize(48, params);

			for(int i = 0; i < 48; i++){
				Ca_from_my_side_directly[p][i].unpack(CaStream_from_my_side[p]);
				Cb_from_my_side_directly[p][i].unpack(CbStream_from_my_side[p]);
				Cf_from_my_side_directly[p][i].unpack(CfStream_from_my_side[p]);
			}
		}

		Ca_from_my_side_directly[my_num].resize(48, params);
		Cb_from_my_side_directly[my_num].resize(48, params);
		Cf_from_my_side_directly[my_num].resize(48, params);

		int my_job_start = 3 * 16 + 48 * (my_num - 3);
		for(int i = 0; i < 48; i++){
			Ca_from_my_side_directly[my_num][i] = Ca[i + my_job_start];
			Cb_from_my_side_directly[my_num][i] = Cb[i + my_job_start];
			Cf_from_my_side_directly[my_num][i] = Cf[i + my_job_start];
		}

		#pragma omp parallel for
		for(int i = 0; i < 48; i++){
			for(int p = 3 + 1; p < 4; p++){
				add(Ca_from_my_side_directly[3][i], Ca_from_my_side_directly[3][i], Ca_from_my_side_directly[p][i]);
				add(Cb_from_my_side_directly[3][i], Cb_from_my_side_directly[3][i], Cb_from_my_side_directly[p][i]);
				add(Cf_from_my_side_directly[3][i], Cf_from_my_side_directly[3][i], Cf_from_my_side_directly[p][i]);
			}
		}


		#pragma omp parallel for
		for(int t = 0; t < 3; t++){
			if(t == 0){
				Ca_from_my_shadow[my_shadow_num_1].resize(16, params);
				Cb_from_my_shadow[my_shadow_num_1].resize(16, params);
				Cf_from_my_shadow[my_shadow_num_1].resize(16, params);

				for(int i = 0; i < 16; i++){
					Ca_from_my_shadow[my_shadow_num_1][i].unpack(CaStream_from_my_shadow_about_shadowing[my_shadow_num_1]);
					Cb_from_my_shadow[my_shadow_num_1][i].unpack(CbStream_from_my_shadow_about_shadowing[my_shadow_num_1]);
					Cf_from_my_shadow[my_shadow_num_1][i].unpack(CfStream_from_my_shadow_about_shadowing[my_shadow_num_1]);
				}

				for(int i = 0; i < 16; i++){
					add(Ca_from_my_side_directly[3][i + 0], Ca_from_my_side_directly[3][i + 0], Ca_from_my_shadow[my_shadow_num_1][i]);
					add(Cb_from_my_side_directly[3][i + 0], Cb_from_my_side_directly[3][i + 0], Cb_from_my_shadow[my_shadow_num_1][i]);
					add(Cf_from_my_side_directly[3][i + 0], Cf_from_my_side_directly[3][i + 0], Cf_from_my_shadow[my_shadow_num_1][i]);
				}
			}

			if(t == 1){
				Ca_from_my_shadow[my_shadow_num_2].resize(16, params);
				Cb_from_my_shadow[my_shadow_num_2].resize(16, params);
				Cf_from_my_shadow[my_shadow_num_2].resize(16, params);

				for(int i = 0; i < 16; i++){
					Ca_from_my_shadow[my_shadow_num_2][i].unpack(CaStream_from_my_shadow_about_shadowing[my_shadow_num_2]);
					Cb_from_my_shadow[my_shadow_num_2][i].unpack(CbStream_from_my_shadow_about_shadowing[my_shadow_num_2]);
					Cf_from_my_shadow[my_shadow_num_2][i].unpack(CfStream_from_my_shadow_about_shadowing[my_shadow_num_2]);
				}

				for(int i = 0; i < 16; i++){
					add(Ca_from_my_side_directly[3][i + 16], Ca_from_my_side_directly[3][i + 16], Ca_from_my_shadow[my_shadow_num_2][i]);
					add(Cb_from_my_side_directly[3][i + 16], Cb_from_my_side_directly[3][i + 16], Cb_from_my_shadow[my_shadow_num_2][i]);
					add(Cf_from_my_side_directly[3][i + 16], Cf_from_my_side_directly[3][i + 16], Cf_from_my_shadow[my_shadow_num_2][i]);
				}
			}

			if(t == 2){
				Ca_from_my_shadow[my_shadow_num_3].resize(16, params);
				Cb_from_my_shadow[my_shadow_num_3].resize(16, params);
				Cf_from_my_shadow[my_shadow_num_3].resize(16, params);

				for(int i = 0; i < 16; i++){
					Ca_from_my_shadow[my_shadow_num_3][i].unpack(CaStream_from_my_shadow_about_shadowing[my_shadow_num_3]);
					Cb_from_my_shadow[my_shadow_num_3][i].unpack(CbStream_from_my_shadow_about_shadowing[my_shadow_num_3]);
					Cf_from_my_shadow[my_shadow_num_3][i].unpack(CfStream_from_my_shadow_about_shadowing[my_shadow_num_3]);
				}

				for(int i = 0; i < 16; i++){
					add(Ca_from_my_side_directly[3][i + 32], Ca_from_my_side_directly[3][i + 32], Ca_from_my_shadow[my_shadow_num_3][i]);
					add(Cb_from_my_side_directly[3][i + 32], Cb_from_my_side_directly[3][i + 32], Cb_from_my_shadow[my_shadow_num_3][i]);
					add(Cf_from_my_side_directly[3][i + 32], Cf_from_my_side_directly[3][i + 32], Cf_from_my_shadow[my_shadow_num_3][i]);
				}
			}
		}
	}
	printf("Step 7: add my data + my side data + data from shadow done.\n");

	printf("Step 8: mul the data and add the f.\n");
	if(my_num < 3){
		int my_start = 16 * my_num;

		#pragma omp parallel for
		for(int i = 0; i < 16; i++){
			mul(Cc[my_start + i], Ca_from_my_side_directly[0][i], Cb_from_my_side_directly[0][i], pk);
		}

		#pragma omp parallel for
		for(int i = 0; i < 16; i++){
			if(Cc[my_start + i].level()==0){
				Cf_from_my_side_directly[0][i].Scale(FieldD.get_prime());
			}
			add(Cc[my_start + i], Cc[my_start + i], Cf_from_my_side_directly[0][i]);
		}
	}else{
		int my_start = 3 * 16 + 48 * (my_num - 3);

		#pragma omp parallel for
		for(int i = 0; i < 48; i++){
			mul(Cc[my_start + i], Ca_from_my_side_directly[3][i], Cb_from_my_side_directly[3][i], pk);
		}

		#pragma omp parallel for
		for(int i = 0; i < 48; i++){
			if(Cc[my_start + i].level()==0){
				Cf_from_my_side_directly[3][i].Scale(FieldD.get_prime());
			}
			add(Cc[my_start + i], Cc[my_start + i], Cf_from_my_side_directly[3][i]);
		}
	}
	printf("Step 8: mul the data and add the f done.\n");

	printf("Step 9: prepare c+f to my side and my shadow.\n");
	octetStream CcfStream_to_my_side;
	vector<octetStream> CcfStream_to_my_shadow(num_players);

	if(my_num < 3){
		int my_start = 16 * my_num;

		CcfStream_to_my_side.resize_precise(16 * Cc[0].report_size(USED));
		CcfStream_to_my_side.reset_write_head();

		int my_shadow_num_1 = 3 + (my_num / 3);

		CcfStream_to_my_shadow[my_shadow_num_1].resize_precise(16 * Cc[0].report_size(USED));
		CcfStream_to_my_shadow[my_shadow_num_1].reset_write_head();

		for(int i = 0; i < 16; i++){
			Cc[i + my_start].pack(CcfStream_to_my_side);
			Cc[i + my_start].pack(CcfStream_to_my_shadow[my_shadow_num_1]);
		}
	}else{
		int my_start = 3 * 16 + 48 * (my_num - 3);

		CcfStream_to_my_side.resize_precise(48 * Cc[0].report_size(USED));
		CcfStream_to_my_side.reset_write_head();

		int my_shadow_num_1 = (my_num - 3) * 3 + 0;
		int my_shadow_num_2 = (my_num - 3) * 3 + 1;
		int my_shadow_num_3 = (my_num - 3) * 3 + 2;

		CcfStream_to_my_shadow[my_shadow_num_1].resize_precise(16 * Cc[0].report_size(USED));
		CcfStream_to_my_shadow[my_shadow_num_1].reset_write_head();

		CcfStream_to_my_shadow[my_shadow_num_2].resize_precise(16 * Cc[0].report_size(USED));
		CcfStream_to_my_shadow[my_shadow_num_2].reset_write_head();

		CcfStream_to_my_shadow[my_shadow_num_3].resize_precise(16 * Cc[0].report_size(USED));
		CcfStream_to_my_shadow[my_shadow_num_3].reset_write_head();

		for(int i = 0; i < 16; i++){
			Cc[i + my_start].pack(CcfStream_to_my_side);
			Cc[i + my_start].pack(CcfStream_to_my_shadow[my_shadow_num_1]);
		}

		for(int i = 16; i < 32; i++){
			Cc[i + my_start].pack(CcfStream_to_my_side);
			Cc[i + my_start].pack(CcfStream_to_my_shadow[my_shadow_num_2]);
		}

		for(int i = 32; i < 48; i++){
			Cc[i + my_start].pack(CcfStream_to_my_side);
			Cc[i + my_start].pack(CcfStream_to_my_shadow[my_shadow_num_3]);
		}
	}
	printf("Step 9: prepare c+f to my side and my shadow done.\n");

	int CcSize = Cc[0].report_size(USED);

	printf("Step 10: exchanging c+f from my side, and obtaining c+f from my boss.\n");
	vector<octetStream> CcfStream_from_my_side(num_players);
	vector<octetStream> CcfStream_from_my_boss(num_players);

	if(my_num < 3){
		for(int p = 0; p < 3; p++){
			if(p == my_num) continue;

			res.push_back(pool->enqueue([&P, p, &CcfStream_from_my_side, CcSize]() {
				CcfStream_from_my_side[p].resize_precise(16 * CcSize);
				CcfStream_from_my_side[p].reset_write_head();

				printf("-- receiving from party %d\n", p);
				auto socket_recv = P.socket(p);
				CcfStream_from_my_side[p].Receive(socket_recv);
				printf("-- received from party %d\n", p);
			}));
		}

		int my_boss_num_1 = 3 + (my_num / 3);

		res.push_back(pool->enqueue([&P, my_boss_num_1, &CcfStream_from_my_boss, CcSize]() {
			CcfStream_from_my_boss[my_boss_num_1].resize_precise(16 * CcSize);
			CcfStream_from_my_boss[my_boss_num_1].reset_write_head();

			printf("-- receiving from boss_1 %d\n", my_boss_num_1);
			auto socket_recv = P.socket(my_boss_num_1);
			CcfStream_from_my_boss[my_boss_num_1].Receive(socket_recv);
			printf("-- received from boss_1 %d\n", my_boss_num_1);
		}));
	}else{
		for(int p = 3; p < 4; p++){
			if(p == my_num) continue;

			res.push_back(pool->enqueue([&P, p, &CcfStream_from_my_side, CcSize]() {
				CcfStream_from_my_side[p].resize_precise(48 * CcSize);
				CcfStream_from_my_side[p].reset_write_head();

				printf("-- receiving from party %d\n", p);
				auto socket_recv = P.socket(p);
				CcfStream_from_my_side[p].Receive(socket_recv);
				printf("-- received from party %d\n", p);
			}));
		}

		int my_boss_num_1 = (my_num - 3) * 3 + 0;
		int my_boss_num_2 = (my_num - 3) * 3 + 1;
		int my_boss_num_3 = (my_num - 3) * 3 + 2;

		res.push_back(pool->enqueue([&P, my_boss_num_1, &CcfStream_from_my_boss, CcSize]() {
			CcfStream_from_my_boss[my_boss_num_1].resize_precise(16 * CcSize);
			CcfStream_from_my_boss[my_boss_num_1].reset_write_head();

			printf("-- receiving from boss_1 %d\n", my_boss_num_1);
			auto socket_recv = P.socket(my_boss_num_1);
			CcfStream_from_my_boss[my_boss_num_1].Receive(socket_recv);
			printf("-- received from boss_1 %d\n", my_boss_num_1);
		}));

		res.push_back(pool->enqueue([&P, my_boss_num_2, &CcfStream_from_my_boss, CcSize]() {
			CcfStream_from_my_boss[my_boss_num_2].resize_precise(16 * CcSize);
			CcfStream_from_my_boss[my_boss_num_2].reset_write_head();

			printf("-- receiving from boss_2 %d\n", my_boss_num_2);
			auto socket_recv = P.socket(my_boss_num_2);
			CcfStream_from_my_boss[my_boss_num_2].Receive(socket_recv);
			printf("-- received from boss_2 %d\n", my_boss_num_2);
		}));

		res.push_back(pool->enqueue([&P, my_boss_num_3, &CcfStream_from_my_boss, CcSize]() {
			CcfStream_from_my_boss[my_boss_num_3].resize_precise(16 * CcSize);
			CcfStream_from_my_boss[my_boss_num_3].reset_write_head();

			printf("-- receiving from boss_3 %d\n", my_boss_num_3);
			auto socket_recv = P.socket(my_boss_num_3);
			CcfStream_from_my_boss[my_boss_num_3].Receive(socket_recv);
			printf("-- received from boss_3 %d\n", my_boss_num_3);
		}));
	}

	if(my_num < 3){
		for(int p = 0; p < 3; p++){
			if(p == my_num) continue;

			res.push_back(pool->enqueue([&P, p, &CcfStream_to_my_side]() {
				printf("-- sending to party %d\n", p);
				auto socket_send = P.socket(p);
				CcfStream_to_my_side.Send(socket_send);
				printf("-- sent to party %d\n", p);
			}));
		}

		int my_shadow_num_1 = 3 + (my_num / 3);
		res.push_back(pool->enqueue([&P, my_shadow_num_1, &CcfStream_to_my_shadow]() {
			printf("-- sending to shadow_1 %d\n", my_shadow_num_1);
			auto socket_send = P.socket(my_shadow_num_1);
			CcfStream_to_my_shadow[my_shadow_num_1].Send(socket_send);
			printf("-- sent to shadow_1 %d\n", my_shadow_num_1);
		}));
	}else{
		for(int p = 3; p < 4; p++){
			if(p == my_num) continue;

			res.push_back(pool->enqueue([&P, p, &CcfStream_to_my_side]() {
				printf("-- sending to party %d\n", p);
				auto socket_send = P.socket(p);
				CcfStream_to_my_side.Send(socket_send);
				printf("-- sent to party %d\n", p);
			}));
		}

		int my_shadow_num_1 = (my_num - 3) * 3 + 0;
		int my_shadow_num_2 = (my_num - 3) * 3 + 1;
		int my_shadow_num_3 = (my_num - 3) * 3 + 2;

		res.push_back(pool->enqueue([&P, my_shadow_num_1, &CcfStream_to_my_shadow]() {
			printf("-- sending to shadow_1 %d\n", my_shadow_num_1);
			auto socket_send = P.socket(my_shadow_num_1);
			CcfStream_to_my_shadow[my_shadow_num_1].Send(socket_send);
			printf("-- sent to shadow_1 %d\n", my_shadow_num_1);
		}));

		res.push_back(pool->enqueue([&P, my_shadow_num_2, &CcfStream_to_my_shadow]() {
			printf("-- sending to shadow_2 %d\n", my_shadow_num_2);
			auto socket_send = P.socket(my_shadow_num_2);
			CcfStream_to_my_shadow[my_shadow_num_2].Send(socket_send);
			printf("-- sent to shadow_2 %d\n", my_shadow_num_2);
		}));

		res.push_back(pool->enqueue([&P, my_shadow_num_3, &CcfStream_to_my_shadow]() {
			printf("-- sending to shadow_3 %d\n", my_shadow_num_3);
			auto socket_send = P.socket(my_shadow_num_3);
			CcfStream_to_my_shadow[my_shadow_num_3].Send(socket_send);
			printf("-- sent to shadow_3 %d\n", my_shadow_num_3);
		}));
	}
	joinNclean(res);
	printf("Step 10: exchanging c+f from my side, and obtaining c+f from my boss done.\n");

	printf("Step 11: exchanging the c+f from the other side via the shadows.\n");
	vector<octetStream> CcfStream_from_my_side_about_shadowing_9(num_players);
	vector<octetStream> CcfStream_from_my_side_about_shadowing_3_1(num_players);
	vector<octetStream> CcfStream_from_my_side_about_shadowing_3_2(num_players);
	vector<octetStream> CcfStream_from_my_side_about_shadowing_3_3(num_players);

	if(my_num < 3){
		for(int p = 0; p < 3; p++){
			if(p == my_num) continue;

			res.push_back(pool->enqueue([&P, p, &CcfStream_from_my_side_about_shadowing_9, CcSize]() {
				CcfStream_from_my_side_about_shadowing_9[p].resize_precise(16 * CcSize);
				CcfStream_from_my_side_about_shadowing_9[p].reset_write_head();

				printf("-- receiving from party %d\n", p);
				auto socket_recv = P.socket(p);
				CcfStream_from_my_side_about_shadowing_9[p].Receive(socket_recv);
				printf("-- received from party %d\n", p);
			}));
		}
	}else{
		for(int p = 3; p < 4; p++){
			if(p == my_num) continue;

			res.push_back(pool->enqueue([&P, p,
				&CcfStream_from_my_side_about_shadowing_3_1,
				&CcfStream_from_my_side_about_shadowing_3_2,
				&CcfStream_from_my_side_about_shadowing_3_3, CcSize]() {

				CcfStream_from_my_side_about_shadowing_3_1[p].resize_precise(16 * CcSize);
				CcfStream_from_my_side_about_shadowing_3_1[p].reset_write_head();

				CcfStream_from_my_side_about_shadowing_3_2[p].resize_precise(16 * CcSize);
				CcfStream_from_my_side_about_shadowing_3_2[p].reset_write_head();

				CcfStream_from_my_side_about_shadowing_3_3[p].resize_precise(16 * CcSize);
				CcfStream_from_my_side_about_shadowing_3_3[p].reset_write_head();

				printf("-- receiving from party %d\n", p);
				auto socket_recv = P.socket(p);
				CcfStream_from_my_side_about_shadowing_3_1[p].Receive(socket_recv);
				CcfStream_from_my_side_about_shadowing_3_2[p].Receive(socket_recv);
				CcfStream_from_my_side_about_shadowing_3_3[p].Receive(socket_recv);
				printf("-- received from party %d\n", p);
			}));
		}
	}

	if(my_num < 3){
		int my_boss_num_1 = 3 + (my_num / 3);

		for(int p = 0; p < 3; p++){
			if(p == my_num) continue;

			res.push_back(pool->enqueue([&P, p, my_boss_num_1, &CcfStream_from_my_boss]() {
				printf("-- forwarding to party %d\n", p);
				auto socket_send = P.socket(p);
				CcfStream_from_my_boss[my_boss_num_1].Send(socket_send);
				printf("-- forwarded to party %d\n", p);
			}));
		}
	}else{
		int my_boss_num_1 = (my_num - 3) * 3 + 0;
		int my_boss_num_2 = (my_num - 3) * 3 + 1;
		int my_boss_num_3 = (my_num - 3) * 3 + 2;

		for(int p = 3; p < 4; p++){
			if(p == my_num) continue;

			res.push_back(pool->enqueue([&P, p, my_boss_num_1, my_boss_num_2, my_boss_num_3, &CcfStream_from_my_boss, CcSize]() {
				printf("-- forwarding to party %d\n", p);
				auto socket_send = P.socket(p);
				CcfStream_from_my_boss[my_boss_num_1].Send(socket_send);
				CcfStream_from_my_boss[my_boss_num_2].Send(socket_send);
				CcfStream_from_my_boss[my_boss_num_3].Send(socket_send);
				printf("-- forwarded to party %d\n", p);
			}));
		}
	}
	joinNclean(res);
	printf("Step 11: exchanging the c+f from the other side via the shadows done.\n");

	printf("Step 12: unpacking c+f.\n");
	if(my_num < 3){
		for(int p = 0; p < 3; p++){
			if(p == my_num) continue;

			int p_start = p * 16;
			int p_end = p_start + 16;
			int p_batch_size = 16;

			res.push_back(pool->enqueue([&Cc, p, p_start, p_end, p_batch_size, &CcfStream_from_my_side]() {
				for(int i = p_start; i < p_end; i++){
					Cc[i].unpack(CcfStream_from_my_side[p]);
				}
			}));
		}

		int my_boss_num_1 = 3 + (my_num / 3);
		{
			int p_start = 3 * 16 + 16 * my_num;
			int p_end =  p_start + 16;
			int p_batch_size = 16;

			res.push_back(pool->enqueue([&Cc, p_start, p_end, p_batch_size, my_boss_num_1, &CcfStream_from_my_boss]() {
				for(int i = p_start; i < p_end; i++){
					Cc[i].unpack(CcfStream_from_my_boss[my_boss_num_1]);
				}
			}));
		}

		for(int p = 0; p < 3; p++){
			if(p == my_num) continue;

			int p_start = 3 * 16 +  16 * p;
			int p_end = p_start + 16;
			int p_batch_size = 16;

			res.push_back(pool->enqueue([&Cc, p, p_start, p_end, p_batch_size, &CcfStream_from_my_side_about_shadowing_9]() {
				for(int i = p_start; i < p_end; i++){
					Cc[i].unpack(CcfStream_from_my_side_about_shadowing_9[p]);
				}
			}));
		}
	}else{
		for(int p = 3; p < 4; p++){
			if(p == my_num) continue;

			int p_start = 3 * 16 + (p - 3) * 48;
			int p_end = p_start + 48;
			int p_batch_size = 48;

			res.push_back(pool->enqueue([&Cc, p, p_start, p_end, p_batch_size, &CcfStream_from_my_side]() {
				for(int i = p_start; i < p_end; i++){
					Cc[i].unpack(CcfStream_from_my_side[p]);
				}
			}));
		}

		int my_boss_num_1 = (my_num - 3) * 3 + 0;
		int my_boss_num_2 = (my_num - 3) * 3 + 1;
		int my_boss_num_3 = (my_num - 3) * 3 + 2;

		{
			int p_start = 16 * my_boss_num_1;
			int p_end =  p_start + 16;
			int p_batch_size = 16;

			res.push_back(pool->enqueue([&Cc, my_boss_num_1, p_start, p_end, p_batch_size, &CcfStream_from_my_boss]() {
				for(int i = p_start; i < p_end; i++){
					Cc[i].unpack(CcfStream_from_my_boss[my_boss_num_1]);
				}
			}));
		}

		{
			int p_start = 16 * my_boss_num_2;
			int p_end =  p_start + 16;
			int p_batch_size = 16;

			res.push_back(pool->enqueue([&Cc, my_boss_num_2, p_start, p_end, p_batch_size, &CcfStream_from_my_boss]() {
				for(int i = p_start; i < p_end; i++){
					Cc[i].unpack(CcfStream_from_my_boss[my_boss_num_2]);
				}
			}));
		}

		{
			int p_start = 16 * my_boss_num_3;
			int p_end =  p_start + 16;
			int p_batch_size = 16;

			res.push_back(pool->enqueue([&Cc, my_boss_num_3, p_start, p_end, p_batch_size, &CcfStream_from_my_boss]() {
				for(int i = p_start; i < p_end; i++){
					Cc[i].unpack(CcfStream_from_my_boss[my_boss_num_3]);
				}
			}));
		}

		for(int p = 3; p < 4; p++){
			if(p == my_num) continue;

			int p_start_1 = (p - 3) * 48;
			int p_end_1 = p_start_1 + 16;
			int p_batch_size_1 = 16;

			int p_start_2 = p_end_1;
			int p_end_2 = p_start_2 + 16;
			int p_batch_size_2 = 16;

			int p_start_3 = p_end_2;
			int p_end_3 = p_start_3 + 16;
			int p_batch_size_3 = 16;

			res.push_back(pool->enqueue([p, &Cc, p_start_1, p_end_1, p_batch_size_1, p_start_2, p_end_2, p_batch_size_2, p_start_3, p_end_3, p_batch_size_3,
				&CcfStream_from_my_side_about_shadowing_3_1,
				&CcfStream_from_my_side_about_shadowing_3_2,
				&CcfStream_from_my_side_about_shadowing_3_3]() {

				for(int i = p_start_1; i < p_end_1; i++){
					Cc[i].unpack(CcfStream_from_my_side_about_shadowing_3_1[p]);
				}

				for(int i = p_start_2; i < p_end_2; i++){
					Cc[i].unpack(CcfStream_from_my_side_about_shadowing_3_2[p]);
				}

				for(int i = p_start_3; i < p_end_3; i++){
					Cc[i].unpack(CcfStream_from_my_side_about_shadowing_3_3[p]);
				}
			}));
		}
	}
	joinNclean(res);
	printf("Step 12: unpacking c+f done.\n");

	printf("Step 13: distributed decryption computing vv.\n");
	AddableMatrix<bigint> vv;
	vv.resize(batch_size, pk.get_params().phi_m());
	bigint limit = pk.get_params().Q() << 64;
	vv.allocate_slots(limit);

	#pragma omp parallel for
	for(int i = 0; i < batch_size; i++){
		(*psk).dist_decrypt_1(vv[i], Cc[i], my_num, num_players);
	}

	printf("Step 13: distributed decryption computing vv done.\n");

	int vvSize = vv[0].report_size(USED);

	printf("Step 14: prepare vv to my side.\n");
	vector<octetStream> vvStream_to_my_side(num_players);
	if(my_num < 3){
		for(int p = 0; p < 3; p++){
			if(p == my_num) continue;

			int p_start = p * 16;
			int p_end = p_start + 16;
			int p_batch_size = 16;

			res.push_back(pool->enqueue([p, &vv, p_start, p_end, p_batch_size, &vvStream_to_my_side, vvSize](){
				vvStream_to_my_side[p].resize_precise(p_batch_size * vvSize);
				vvStream_to_my_side[p].reset_write_head();

				for(int i = p_start; i < p_end; i++){
					vv[i].pack(vvStream_to_my_side[p]);
				}
			}));
		}
	}else{
		for(int p = 3; p < 4; p++){
			if(p == my_num) continue;

			int p_start = 3 * 16 + (p - 3) * 48;
			int p_end = p_start + 48;
			int p_batch_size = 48;

			res.push_back(pool->enqueue([p, &vv, p_start, p_end, p_batch_size, &vvStream_to_my_side, vvSize](){
				vvStream_to_my_side[p].resize_precise(p_batch_size * vvSize);
				vvStream_to_my_side[p].reset_write_head();

				for(int i = p_start; i < p_end; i++){
					vv[i].pack(vvStream_to_my_side[p]);
				}
			}));
		}
	}
	joinNclean(res);
	printf("Step 14: prepare vv to my side done.\n");

	printf("Step 15: exchanging vv with my side.\n");
	vector<octetStream> vvStream_from_my_side(num_players);
	if(my_num < 3){
		for(int p = 0; p < 3; p++){
			if(p == my_num) continue;

			res.push_back(pool->enqueue([p, &P, &vvStream_from_my_side, vvSize](){
				vvStream_from_my_side[p].resize_precise(16 * vvSize);
				vvStream_from_my_side[p].reset_write_head();

				printf("-- receiving from party %d\n", p);
				auto sock_recv = P.socket(p);
				vvStream_from_my_side[p].Receive(sock_recv);
				printf("-- received from party %d\n", p);
			}));
		}
	}else{
		for(int p = 3; p < 4; p++){
			if(p == my_num) continue;

			res.push_back(pool->enqueue([p, &P, &vvStream_from_my_side, vvSize](){
				vvStream_from_my_side[p].resize_precise(48 * vvSize);
				vvStream_from_my_side[p].reset_write_head();

				printf("-- receiving from party %d\n", p);
				auto sock_recv = P.socket(p);
				vvStream_from_my_side[p].Receive(sock_recv);
				printf("-- received from party %d\n", p);
			}));
		}
	}

	if(my_num < 3){
		for(int p = 0; p < 3; p++){
			if(p == my_num) continue;

			res.push_back(pool->enqueue([p, &P, &vvStream_to_my_side, vvSize](){
				printf("-- sending to party %d\n", p);
				auto sock_send = P.socket(p);
				vvStream_to_my_side[p].Send(sock_send);
				printf("-- sent to party %d\n", p);
			}));
		}
	}else{
		for(int p = 3; p < 4; p++){
			if(p == my_num) continue;

			res.push_back(pool->enqueue([p, &P, &vvStream_to_my_side, vvSize](){
				printf("-- sending to party %d\n", p);
				auto sock_send = P.socket(p);
				vvStream_to_my_side[p].Send(sock_send);
				printf("-- sent to party %d\n", p);
			}));
		}
	}
	joinNclean(res);
	printf("Step 15: exchanging vv with my side done.\n");

	printf("Step 16: prepare vv for the shadows.\n");
	vector<octetStream> vvStream_to_my_side_about_shadowing(num_players);
	if(my_num < 3){
		#pragma omp parallel for
		for(int p = 0; p < 3; p++){
			if(p == my_num) continue;

			int p_start = 3 * 16 + p * 16;
			int p_end = p_start + 16;

			vvStream_to_my_side_about_shadowing[p].resize_precise(16 * vvSize);
			vvStream_to_my_side_about_shadowing[p].reset_write_head();

			for(int i = p_start; i < p_end; i++){
				vv[i].pack(vvStream_to_my_side_about_shadowing[p]);
			}
		}
	}else{
		#pragma omp parallel for
		for(int p = 3; p < 4; p++){
			if(p == my_num) continue;

			int p_start_1 = (p - 3) * 48;
			int p_end_1 = p_start_1 + 16;

			int p_start_2 = p_end_1;
			int p_end_2 = p_start_2 + 16;

			int p_start_3 = p_end_2;
			int p_end_3 = p_start_3 + 16;

			vvStream_to_my_side_about_shadowing[p].resize_precise(48 * vvSize);
			vvStream_to_my_side_about_shadowing[p].reset_write_head();

			for(int i = p_start_1; i < p_end_1; i++){
				vv[i].pack(vvStream_to_my_side_about_shadowing[p]);
			}

			for(int i = p_start_2; i < p_end_2; i++){
				vv[i].pack(vvStream_to_my_side_about_shadowing[p]);
			}

			for(int i = p_start_3; i < p_end_3; i++){
				vv[i].pack(vvStream_to_my_side_about_shadowing[p]);
			}
		}
	}
	printf("Step 16: prepare vv for the shadows done.\n");

	printf("Step 17: exchange vv for the shadows.\n");
	vector<octetStream> vvStream_from_my_side_about_shadowing(num_players);
	if(my_num < 3){
		for(int p = 0; p < 3; p++){
			if(p == my_num) continue;

			res.push_back(pool->enqueue([p, &P, &vvStream_from_my_side_about_shadowing, vvSize](){
				vvStream_from_my_side_about_shadowing[p].resize_precise(16 * vvSize);
				vvStream_from_my_side_about_shadowing[p].reset_write_head();

				printf("-- receiving from %d\n", p);
				auto sock_recv = P.socket(p);
				vvStream_from_my_side_about_shadowing[p].Receive(sock_recv);
				printf("-- received from %d\n", p);
			}));
		}
	}else{
		for(int p = 3; p < 4; p++){
			if(p == my_num) continue;

			res.push_back(pool->enqueue([p, &P, &vvStream_from_my_side_about_shadowing, vvSize](){
				vvStream_from_my_side_about_shadowing[p].resize_precise(16 * vvSize);
				vvStream_from_my_side_about_shadowing[p].reset_write_head();

				printf("-- receiving from %d\n", p);
				auto sock_recv = P.socket(p);
				vvStream_from_my_side_about_shadowing[p].Receive(sock_recv);
				printf("-- received from %d\n", p);
			}));
		}
	}

	if(my_num < 3){
		for(int p = 0; p < 3; p++){
			if(p == my_num) continue;

			res.push_back(pool->enqueue([p, &P, &vvStream_to_my_side_about_shadowing](){
				printf("-- sending to %d\n", p);
				auto sock_send = P.socket(p);
				vvStream_to_my_side_about_shadowing[p].Send(sock_send);
				printf("-- sent to %d\n", p);
			}));
		}
	}else{
		for(int p = 3; p < 4; p++){
			if(p == my_num) continue;

			res.push_back(pool->enqueue([p, &P, &vvStream_to_my_side_about_shadowing](){
				printf("-- sending to %d\n", p);
				auto sock_send = P.socket(p);
				vvStream_to_my_side_about_shadowing[p].Send(sock_send);
				printf("-- sent to %d\n", p);
			}));
		}
	}
	joinNclean(res);
	printf("Step 17: exchange vv for the shadows done.\n");

	printf("Step 18: shadows merge vv for shadowing.\n");
	vector<AddableMatrix<bigint>> vv_from_my_side_about_shadowing(num_players);
	if(my_num < 3){
		for(int p = 0; p < 3; p++){
			if(p == my_num) continue;

			res.push_back(pool->enqueue([p, &vvStream_from_my_side_about_shadowing, &vv_from_my_side_about_shadowing, &pk](){
				vv_from_my_side_about_shadowing[p].resize(16, pk.get_params().phi_m());
				bigint limit = pk.get_params().Q() << 64;
				vv_from_my_side_about_shadowing[p].allocate_slots(limit);

				for(int i = 0; i < 16; i++){
					vv_from_my_side_about_shadowing[p][i].unpack(vvStream_from_my_side_about_shadowing[p]);
				}
			}));
		}
		joinNclean(res);

		vv_from_my_side_about_shadowing[my_num].resize(16, pk.get_params().phi_m());
		bigint limit = pk.get_params().Q() << 64;
		vv_from_my_side_about_shadowing[my_num].allocate_slots(limit);

		int my_shadow_job_start = 3 * 16 + 16 * my_num;

		for(int i = 0; i < 16; i++){
			vv_from_my_side_about_shadowing[my_num][i] = vv[i + my_shadow_job_start];
		}

		bigint mod = pk.get_params().p0();
		for(int p = 1; p < 3; p++){
			for(int i = 0; i < 16; i++){
				for(int ind = 0; ind < pk.get_params().phi_m(); ind++){
					vv_from_my_side_about_shadowing[0][i][ind] += vv_from_my_side_about_shadowing[p][i][ind];
					vv_from_my_side_about_shadowing[0][i][ind] %= mod;
				}
			}
		}
	}else{
		for(int p = 3; p < 4; p++){
			if(p == my_num) continue;

			res.push_back(pool->enqueue([p, &vvStream_from_my_side_about_shadowing, &vv_from_my_side_about_shadowing, &pk](){
				vv_from_my_side_about_shadowing[p].resize(48, pk.get_params().phi_m());
				bigint limit = pk.get_params().Q() << 64;
				vv_from_my_side_about_shadowing[p].allocate_slots(limit);

				for(int i = 0; i < 48; i++){
					vv_from_my_side_about_shadowing[p][i].unpack(vvStream_from_my_side_about_shadowing[p]);
				}
			}));
		}
		joinNclean(res);

		vv_from_my_side_about_shadowing[my_num].resize(48, pk.get_params().phi_m());
		bigint limit = pk.get_params().Q() << 64;
		vv_from_my_side_about_shadowing[my_num].allocate_slots(limit);

		int my_shadow_job_start = 48 * (my_num - 3);

		for(int i = 0; i < 48; i++){
			vv_from_my_side_about_shadowing[my_num][i] = vv[i + my_shadow_job_start];
		}

		bigint mod = pk.get_params().p0();
		for(int p = 3 + 1; p < 4; p++){
			for(int i = 0; i < 48; i++){
				for(int ind = 0; ind < pk.get_params().phi_m(); ind++){
					if(my_num == 3 && i == 0 && ind == 0){
						cout << "vv_from_my_side_about_shadowing[3][0][0] = " << vv_from_my_side_about_shadowing[3][0][0] << endl;
						cout << "now adding " << vv_from_my_side_about_shadowing[p][0][0] << endl;
					}

					vv_from_my_side_about_shadowing[3][i][ind] += vv_from_my_side_about_shadowing[p][i][ind];
					vv_from_my_side_about_shadowing[3][i][ind] %= mod;

					if(my_num == 3 && i == 0 && ind == 0){
						cout << "vv_from_my_side_about_shadowing[3][0][0] + party " << p << " = " << vv_from_my_side_about_shadowing[3][0][0] << endl;
					}
				}
			}
		}
	}
	printf("Step 18: shadows merge vv for shadowing done.\n");

	printf("Step 19: shadows prepare vv to the boss.\n");
	vector<octetStream> vvStream_to_my_boss(num_players);
	if(my_num < 3){
		int my_boss_num = 3 + (my_num / 3);

		res.push_back(pool->enqueue([my_boss_num, &vvStream_to_my_boss, &vv_from_my_side_about_shadowing, &vvSize](){
			vvStream_to_my_boss[my_boss_num].resize_precise(16 * vvSize);
			vvStream_to_my_boss[my_boss_num].reset_write_head();

			for(int i = 0; i < 16; i++){
					vv_from_my_side_about_shadowing[0][i].pack(vvStream_to_my_boss[my_boss_num]);
			}
		}));
	}else{
		int my_boss_num_1 = (my_num - 3) * 3 + 0;
		int my_boss_num_2 = (my_num - 3) * 3 + 1;
		int my_boss_num_3 = (my_num - 3) * 3 + 2;

		res.push_back(pool->enqueue([my_boss_num_1, &vvStream_to_my_boss, &vv_from_my_side_about_shadowing, &vvSize](){
			vvStream_to_my_boss[my_boss_num_1].resize_precise(16 * vvSize);
			vvStream_to_my_boss[my_boss_num_1].reset_write_head();

			for(int i = 0; i < 16; i++){
					vv_from_my_side_about_shadowing[3][i].pack(vvStream_to_my_boss[my_boss_num_1]);
			}
		}));

		res.push_back(pool->enqueue([my_boss_num_2, &vvStream_to_my_boss, &vv_from_my_side_about_shadowing, &vvSize](){
			vvStream_to_my_boss[my_boss_num_2].resize_precise(16 * vvSize);
			vvStream_to_my_boss[my_boss_num_2].reset_write_head();

			for(int i = 16; i < 32; i++){
					vv_from_my_side_about_shadowing[3][i].pack(vvStream_to_my_boss[my_boss_num_2]);
			}
		}));

		res.push_back(pool->enqueue([my_boss_num_3, &vvStream_to_my_boss, &vv_from_my_side_about_shadowing, &vvSize](){
			vvStream_to_my_boss[my_boss_num_3].resize_precise(16 * vvSize);
			vvStream_to_my_boss[my_boss_num_3].reset_write_head();

			for(int i = 32; i < 48; i++){
					vv_from_my_side_about_shadowing[3][i].pack(vvStream_to_my_boss[my_boss_num_3]);
			}
		}));
	}
	joinNclean(res);
	printf("Step 19: shadows prepare vv to the boss done.\n");

	printf("Step 20: shadows send vv to the boss.\n");
	vector<octetStream> vvStream_from_my_shadow(num_players);
	if(my_num < 3){
		int my_shadow_num_1 = 3 + (my_num / 3);

		res.push_back(pool->enqueue([&P, my_shadow_num_1, &vvStream_from_my_shadow, &vvSize](){
			vvStream_from_my_shadow[my_shadow_num_1].resize_precise(16 * vvSize);
			vvStream_from_my_shadow[my_shadow_num_1].reset_write_head();

			printf("-- receiving from party %d\n", my_shadow_num_1);
			auto sock_recv = P.socket(my_shadow_num_1);
			vvStream_from_my_shadow[my_shadow_num_1].Receive(sock_recv);
			printf("-- received from party %d\n", my_shadow_num_1);
		}));
	}else{
		int my_shadow_num_1 = (my_num - 3) * 3 + 0;
		int my_shadow_num_2 = (my_num - 3) * 3 + 1;
		int my_shadow_num_3 = (my_num - 3) * 3 + 2;

		res.push_back(pool->enqueue([&P, my_shadow_num_1, &vvStream_from_my_shadow, &vvSize](){
			vvStream_from_my_shadow[my_shadow_num_1].resize_precise(16 * vvSize);
			vvStream_from_my_shadow[my_shadow_num_1].reset_write_head();

			printf("-- receiving from party %d\n", my_shadow_num_1);
			auto sock_recv = P.socket(my_shadow_num_1);
			vvStream_from_my_shadow[my_shadow_num_1].Receive(sock_recv);
			printf("-- received from party %d\n", my_shadow_num_1);
		}));

		res.push_back(pool->enqueue([&P, my_shadow_num_2, &vvStream_from_my_shadow, &vvSize](){
			vvStream_from_my_shadow[my_shadow_num_2].resize_precise(16 * vvSize);
			vvStream_from_my_shadow[my_shadow_num_2].reset_write_head();

			printf("-- receiving from party %d\n", my_shadow_num_2);
			auto sock_recv = P.socket(my_shadow_num_2);
			vvStream_from_my_shadow[my_shadow_num_2].Receive(sock_recv);
			printf("-- received from party %d\n", my_shadow_num_2);
		}));

		res.push_back(pool->enqueue([&P, my_shadow_num_3, &vvStream_from_my_shadow, &vvSize](){
			vvStream_from_my_shadow[my_shadow_num_3].resize_precise(16 * vvSize);
			vvStream_from_my_shadow[my_shadow_num_3].reset_write_head();

			printf("-- receiving from party %d\n", my_shadow_num_3);
			auto sock_recv = P.socket(my_shadow_num_3);
			vvStream_from_my_shadow[my_shadow_num_3].Receive(sock_recv);
			printf("-- received from party %d\n", my_shadow_num_3);
		}));
	}

	if(my_num < 3){
		int my_boss_num_1 = 3 + (my_num / 3);

		res.push_back(pool->enqueue([&P, my_boss_num_1, &vvStream_to_my_boss, &vvSize](){
			printf("-- sending to boss %d\n", my_boss_num_1);
			auto sock_send = P.socket(my_boss_num_1);
			vvStream_to_my_boss[my_boss_num_1].Send(sock_send);
			printf("-- sent to boss %d\n", my_boss_num_1);
		}));
	}else{
		int my_boss_num_1 = (my_num - 3) * 3 + 0;
		int my_boss_num_2 = (my_num - 3) * 3 + 1;
		int my_boss_num_3 = (my_num - 3) * 3 + 2;

		res.push_back(pool->enqueue([&P, my_boss_num_1, &vvStream_to_my_boss, &vvSize](){
			printf("-- sending to boss_1 %d\n", my_boss_num_1);
			auto sock_send = P.socket(my_boss_num_1);
			vvStream_to_my_boss[my_boss_num_1].Send(sock_send);
			printf("-- sent to boss_1 %d\n", my_boss_num_1);
		}));

		res.push_back(pool->enqueue([&P, my_boss_num_2, &vvStream_to_my_boss, &vvSize](){
			printf("-- sending to boss_2 %d\n", my_boss_num_2);
			auto sock_send = P.socket(my_boss_num_2);
			vvStream_to_my_boss[my_boss_num_2].Send(sock_send);
			printf("-- sent to boss_2 %d\n", my_boss_num_2);
		}));

		res.push_back(pool->enqueue([&P, my_boss_num_3, &vvStream_to_my_boss, &vvSize](){
			printf("-- sending to boss_3 %d\n", my_boss_num_3);
			auto sock_send = P.socket(my_boss_num_3);
			vvStream_to_my_boss[my_boss_num_3].Send(sock_send);
			printf("-- sent to boss_3 %d\n", my_boss_num_3);
		}));
	}
	joinNclean(res);
	printf("Step 20: shadows send vv to the boss done.\n");

	printf("Step 21: offloading vv.\n");
	vector<AddableMatrix<bigint>> vv_from_my_shadow(num_players);
	if(my_num < 3){
		int my_shadow_num_1 = 3 + (my_num / 3);

		vv_from_my_shadow[my_shadow_num_1].resize(16, pk.get_params().phi_m());
		vv_from_my_shadow[my_shadow_num_1].allocate_slots(limit);

		for(int i = 0; i < 16; i++){
			vv_from_my_shadow[my_shadow_num_1][i].unpack(vvStream_from_my_shadow[my_shadow_num_1]);
		}
	}else{
		int my_shadow_num_1 = (my_num - 3) * 3 + 0;
		int my_shadow_num_2 = (my_num - 3) * 3 + 1;
		int my_shadow_num_3 = (my_num - 3) * 3 + 2;

		vv_from_my_shadow[my_shadow_num_1].resize(16, pk.get_params().phi_m());
		vv_from_my_shadow[my_shadow_num_1].allocate_slots(limit);

		vv_from_my_shadow[my_shadow_num_2].resize(16, pk.get_params().phi_m());
		vv_from_my_shadow[my_shadow_num_2].allocate_slots(limit);

		vv_from_my_shadow[my_shadow_num_3].resize(16, pk.get_params().phi_m());
		vv_from_my_shadow[my_shadow_num_3].allocate_slots(limit);

		for(int i = 0; i < 16; i++){
			vv_from_my_shadow[my_shadow_num_1][i].unpack(vvStream_from_my_shadow[my_shadow_num_1]);
			vv_from_my_shadow[my_shadow_num_2][i].unpack(vvStream_from_my_shadow[my_shadow_num_2]);
			vv_from_my_shadow[my_shadow_num_3][i].unpack(vvStream_from_my_shadow[my_shadow_num_3]);
		}
	}
	printf("Step 21: offloading vv done.\n");

	printf("Step 22: adding vv from my side, myself, and from shadows.\n");
	vector<AddableMatrix<bigint>> vv_from_my_side_directly(num_players);
	if(my_num < 3){
		for(int p = 0; p < 3; p++){
			if(p == my_num) continue;

			res.push_back(pool->enqueue([p, &vvStream_from_my_side, &vv_from_my_side_directly, &pk](){
				vv_from_my_side_directly[p].resize(16, pk.get_params().phi_m());
				bigint limit = pk.get_params().Q() << 64;
				vv_from_my_side_directly[p].allocate_slots(limit);

				for(int i = 0; i < 16; i++){
					vv_from_my_side_directly[p][i].unpack(vvStream_from_my_side[p]);
				}
			}));
		}
		joinNclean(res);

		vv_from_my_side_directly[my_num].resize(16, pk.get_params().phi_m());
		bigint limit = pk.get_params().Q() << 64;
		vv_from_my_side_directly[my_num].allocate_slots(limit);

		int my_start = 16 * my_num;

		#pragma omp parallel for
		for(int i = 0; i < 16; i++){
			vv_from_my_side_directly[my_num][i] = vv[i + my_start];
		}

		bigint mod = pk.get_params().p0();
		for(int p = 0 + 1; p < 3; p++){
			#pragma omp parallel for
			for(int i = 0; i < 16; i++){
				for(int ind = 0; ind < pk.get_params().phi_m(); ind++){
					vv_from_my_side_directly[0][i][ind] += vv_from_my_side_directly[p][i][ind];
					vv_from_my_side_directly[0][i][ind] %= mod;
				}
			}
		}

		int my_shadow_num_1 = 3 + (my_num / 3);

		#pragma omp parallel for
		for(int i = 0; i < 16; i++){
			for(int ind = 0; ind < pk.get_params().phi_m(); ind++){
				vv_from_my_side_directly[0][i][ind] += vv_from_my_shadow[my_shadow_num_1][i][ind];
				vv_from_my_side_directly[0][i][ind] %= mod;
			}
		}
	}else{
		for(int p = 3; p < 4; p++){
			if(p == my_num) continue;

			res.push_back(pool->enqueue([p, &vvStream_from_my_side, &vv_from_my_side_directly, &pk](){
				vv_from_my_side_directly[p].resize(48, pk.get_params().phi_m());
				bigint limit = pk.get_params().Q() << 64;
				vv_from_my_side_directly[p].allocate_slots(limit);

				for(int i = 0; i < 48; i++){
					vv_from_my_side_directly[p][i].unpack(vvStream_from_my_side[p]);
				}
			}));
		}
		joinNclean(res);

		vv_from_my_side_directly[my_num].resize(48, pk.get_params().phi_m());
		bigint limit = pk.get_params().Q() << 64;
		vv_from_my_side_directly[my_num].allocate_slots(limit);

		int my_start = 3 * 16 + 48 * (my_num - 3);

		#pragma omp parallel for
		for(int i = 0; i < 48; i++){
			vv_from_my_side_directly[my_num][i] = vv[i + my_start];
		}

		bigint mod = pk.get_params().p0();
		for(int p = 3 + 1; p < 4; p++){
			#pragma omp parallel for
			for(int i = 0; i < 48; i++){
				for(int ind = 0; ind < pk.get_params().phi_m(); ind++){
					vv_from_my_side_directly[3][i][ind] += vv_from_my_side_directly[p][i][ind];
					vv_from_my_side_directly[3][i][ind] %= mod;
				}
			}
		}

		int my_shadow_num_1 = (my_num - 3) * 3 + 0;
		int my_shadow_num_2 = (my_num - 3) * 3 + 1;
		int my_shadow_num_3 = (my_num - 3) * 3 + 2;

		#pragma omp parallel for
		for(int i = 0; i < 16; i++){
			for(int ind = 0; ind < pk.get_params().phi_m(); ind++){
				vv_from_my_side_directly[3][i + 0][ind] += vv_from_my_shadow[my_shadow_num_1][i][ind];
				vv_from_my_side_directly[3][i + 0][ind] %= mod;
			}
		}

		#pragma omp parallel for
		for(int i = 0; i < 16; i++){
			for(int ind = 0; ind < pk.get_params().phi_m(); ind++){
				vv_from_my_side_directly[3][i + 16][ind] += vv_from_my_shadow[my_shadow_num_2][i][ind];
				vv_from_my_side_directly[3][i + 16][ind] %= mod;
			}
		}

		#pragma omp parallel for
		for(int i = 0; i < 16; i++){
			for(int ind = 0; ind < pk.get_params().phi_m(); ind++){
				vv_from_my_side_directly[3][i + 32][ind] += vv_from_my_shadow[my_shadow_num_3][i][ind];
				vv_from_my_side_directly[3][i + 32][ind] %= mod;
			}
		}
	}
	printf("Step 22: adding vv from my side, myself, and from shadows done.\n");

	printf("Step 23: put vv into c.\n");
	#pragma omp parallel for
	for(int i = 0; i < batch_size; i++){
		int my_start, my_end;
		if(my_num < 3){
			my_start = 16 * my_num;
			my_end = my_start + 16;
		}else{
			my_start = 3 * 16 + 48 * (my_num - 3);
			my_end = my_start + 48;
		}

		if(i >= my_start && i < my_end){
			bigint mod = params.p0();
			if(my_num < 3){
				c[i].set_poly_mod(vv_from_my_side_directly[0][i - my_start], mod);
			}else{
				c[i].set_poly_mod(vv_from_my_side_directly[3][i - my_start], mod);
			}
			sub(c[i], c[i], f[i]);
		}else{
			c[i] = f[i];
			c[i].negate();
		}
	}
	printf("Step 23: put vv into c done.\n");

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


int main(int argc, const char** argv)
{
	// Simulation parameters
	ez::ezOptionParser opt;
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
	opt.parse(argc, argv);

	// Input the simulation parameters
	int nplayers = 4;
	int plainlength = 64;
	int my_num = 0;
	int batch_size = 288;

	string hostname;

	opt.get("-l")->getInt(plainlength);
	opt.get("-h")->getString(hostname);
	opt.get("-p")->getInt(my_num);

	Names N;
	network_setup(N, nplayers, my_num, hostname);

	PlainPlayer P(N, 0xffff << 16);

	FHE_PK *ppk;
	FHE_SK *psk;
	FHE_Params params;
	FD FieldD;

	FHE_keygen(ppk, psk, P, plainlength, params, FieldD);

	ThreadPool pool(64);
	PlaintextVector<FD> a;
	PlaintextVector<FD> b;
	PlaintextVector<FD> c;
	int num_rounds = 1;

	std::chrono::time_point<std::chrono::system_clock> start_time, end_time;
	start_time = std::chrono::system_clock::now();
	for (int i = 0; i < num_rounds; i++)
		generate_one_batch(a, b, c, batch_size, params, FieldD, P, *ppk, psk, &pool);
	end_time = std::chrono::system_clock::now();

	check_first_result(a, b, c, params, FieldD, P, &pool);

	cerr << "Time " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()) / 1000.0 << " seconds " << endl;
	cerr << "Generated " << (a[0].num_slots()) * batch_size << " triplets." << endl;

	cerr << endl;
	cerr << "Rate: " << (a[0].num_slots()) * batch_size * num_rounds / ((std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()) / 1000.0) << " triplets/second" << endl;
}
