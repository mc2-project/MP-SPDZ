/*
* READ FIRST
*
* How to compile?
*   make bench_offline
*
* How to benchmark for two parties?
*   Assume Party 0's IP is x.x.x.x
*   Party 0: ./bench_spdz_triple.x -N 2 -l 64 -h x.x.x.x -p 0
*   Party 1: ./bench_spdz_triple.x -N 2 -l 64 -h x.x.x.x -p 1
*   (to support floating points with sufficient space, we need to set -l to a higher value)
*
* How to benchmark for three parties?
*   Party 0: ./bench_spdz_triple.x -N 3 -l 64 -h x.x.x.x -p 0
*   Party 1: ./bench_spdz_triple.x -N 3 -l 64 -h x.x.x.x -p 1
*   Party 2: ./bench_spdz_triple.x -N 3 -l 64 -h x.x.x.x -p 2
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

	/*
	* Step 1: Generate a_i, b_i, f_i randomly
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

	{
		a.randomize(G);
		b.randomize(G);
		f.randomize(G);
	}
	end_timer_map["sampling randomized a/b/f"][0] = std::chrono::system_clock::now();
	printf("sampling randomized a/b/f done.\n");

	/*
	* Step 2: Encrypt a_i/b_i/f_i and prepare to send it out to the first party
	*/
	printf("encrypting a/b/f.\n");
	start_timer_map["encrypting a/b/f"][0] = std::chrono::system_clock::now();

	Random_Coins rc(params);

	PRNG G_array[omp_get_max_threads()];
	for(int i = 0; i < omp_get_max_threads(); i++){
		G_array[i].ReSeed();
	}

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

	octetStream CaStream;
	octetStream CbStream;
	octetStream CfStream;
	int CaSize = Ca[0].report_size(USED);
	int CbSize = Cb[0].report_size(USED);
	int CfSize = Cf[0].report_size(USED);
	if(my_num != 0){
		CaStream.resize_precise(batch_size * CaSize);
		CaStream.reset_write_head();
		for(int i = 0; i < batch_size; i++){
			Ca[i].pack(CaStream);
		}

		CbStream.resize_precise(batch_size * CbSize);
		CbStream.reset_write_head();
		for(int i = 0; i < batch_size; i++){
			Cb[i].pack(CbStream);
		}

		CfStream.resize_precise(batch_size * CfSize);
		CfStream.reset_write_head();
		for(int i = 0; i < batch_size; i++){
			Cf[i].pack(CfStream);
		}
	}

	/*
	* Step 3:
	* For the first party, receiving others' encrypted a_i/b_i/f_i, adding them together, multiplying a_i and b_i, adding f_i, and sending the resulting ciphertexts back (separate rounds)
	* For the rest of the parties, sending encrypted a_i/b_i/f_i and receiving the masked products back (separate rounds)
	*/
	if(my_num == 0){
		AddableVector<Ciphertext> Ca_others[num_players];
		AddableVector<Ciphertext> Cb_others[num_players];
		AddableVector<Ciphertext> Cf_others[num_players];

		for(int i = 0; i < num_players; i++) {
			Ca_others[i].resize(batch_size, params);
			Cb_others[i].resize(batch_size, params);
			Cf_others[i].resize(batch_size, params);
		}

		Ca_others[0] = Ca;
		Cb_others[0] = Cb;
		Cf_others[0] = Cf;

		Ca.clear();
		Cb.clear();
		Cf.clear();
		Ca.shrink_to_fit();
		Cb.shrink_to_fit();
		Cf.shrink_to_fit();

		octetStream CaStream_other[num_players];
		octetStream CbStream_other[num_players];
		octetStream CfStream_other[num_players];

		start_timer_map["receiving encrypted a/b/f"][0] = std::chrono::system_clock::now();
		vector<future<void>> res;
		for(int j = 1; j < num_players; j++){
			int party = j;
			res.push_back(pool->enqueue([party, &P, &CaStream_other, &CbStream_other, &CfStream_other, batch_size, CaSize, CbSize, CfSize]() {
				CaStream_other[party].resize_precise(batch_size * CaSize);
				CaStream_other[party].reset_write_head();

				CbStream_other[party].resize_precise(batch_size * CbSize);
				CbStream_other[party].reset_write_head();

				CfStream_other[party].resize_precise(batch_size * CfSize);
				CfStream_other[party].reset_write_head();

				printf("party0: receiving encrypted a/b/f from party %d.\n", party);
				auto socket_recv = P.socket(party);
				CaStream_other[party].Receive(socket_recv);
				CbStream_other[party].Receive(socket_recv);
				CfStream_other[party].Receive(socket_recv);
				printf("party0: receiving encrypted a/b/f from party %d done.\n", party);
			}));
		}
		joinNclean(res);
    end_timer_map["receiving encrypted a/b/f"][0] = std::chrono::system_clock::now();

		for(int j = 1; j < num_players; j++){
        int party = j;
        res.push_back(pool->enqueue([party, &P, &CaStream_other, &CbStream_other, &CfStream_other, &Ca_others, &Cb_others, &Cf_others, batch_size, CaSize, CbSize, CfSize]() {
				for(int i = 0; i < batch_size; i++){
					Ca_others[party][i].unpack(CaStream_other[party]);
				}
				for(int i = 0; i < batch_size; i++){
					Cb_others[party][i].unpack(CbStream_other[party]);
				}
				for(int i = 0; i < batch_size; i++){
					Cf_others[party][i].unpack(CfStream_other[party]);
				}
			}));
		}
		joinNclean(res);

		printf("party0: adding encrypted a/b/f.\n");
		start_timer_map["adding encrypted a/b/f"][0] = std::chrono::system_clock::now();
		#pragma omp parallel for
		for(int i = 0; i < batch_size; i++){
			for(int j = 1; j < num_players; j++){
				add(Ca_others[0][i], Ca_others[0][i], Ca_others[j][i]);
				add(Cb_others[0][i], Cb_others[0][i], Cb_others[j][i]);
				add(Cf_others[0][i], Cf_others[0][i], Cf_others[j][i]);
			}
		}
		end_timer_map["adding encrypted a/b/f"][0] = std::chrono::system_clock::now();
		printf("party0: adding encrypted a/b/f done.\n");

		printf("party0: multiplying encrypted a/b.\n");
		start_timer_map["multiplying encrypted a/b"][0] = std::chrono::system_clock::now();
		#pragma omp parallel for
		for(int i = 0; i < batch_size; i++){
			mul(Cc[i], Ca_others[0][i], Cb_others[0][i], pk);
		}
		printf("party0: multiplying encrypted a/b done.\n");
		end_timer_map["multiplying encrypted a/b"][0] = std::chrono::system_clock::now();

		printf("party0: masking c with f.\n");
		start_timer_map["masking c with f"][0] = std::chrono::system_clock::now();
		#pragma omp parallel for
		for(int i = 0; i < batch_size; i++){
			if(Cc[i].level()==0){
				Cf_others[0][i].Scale(FieldD.get_prime());
			}
			add(Cc[i], Cf_others[0][i], Cc[i]);
		}
		printf("party0: masking c with f done.\n");
		end_timer_map["masking c with f"][0] = std::chrono::system_clock::now();

		octetStream CcfStream;
		CcfStream.resize_precise(batch_size * Cc[0].report_size(USED));
		CcfStream.reset_write_head();

		for(int i = 0; i < batch_size; i++){
			Cc[i].pack(CcfStream);
		}

		printf("party0: sending out masked c+f.\n");
		start_timer_map["sending out masked c+f"][0] = std::chrono::system_clock::now();
		vector<future<void>> res2;
		for(int j = 1; j < num_players; j++){
			int party = j;
			res2.push_back(pool->enqueue([party, &P, &CcfStream]() {
				auto socket_send = P.socket(party);
				CcfStream.Send(socket_send);
			}));
		}
		joinNclean(res2);
		printf("party0: sending out masked c+f done.\n");
		end_timer_map["sending out masked c+f"][0] = std::chrono::system_clock::now();
	}else{
		octetStream CaStream_other;
		CaStream_other.resize_precise(batch_size * CaSize);
		CaStream_other.reset_write_head();

		octetStream CbStream_other;
		CbStream_other.resize_precise(batch_size * CbSize);
		CbStream_other.reset_write_head();

		octetStream CfStream_other;
		CfStream_other.resize_precise(batch_size * CfSize);
		CfStream_other.reset_write_head();

		for(int i = 0; i < batch_size; i++){
			Ca[i].pack(CaStream_other);
		}
		for(int i = 0; i < batch_size; i++){
			Cb[i].pack(CbStream_other);
		}
		for(int i = 0; i < batch_size; i++){
			Cf[i].pack(CfStream_other);
		}

		Ca.clear();
		Cb.clear();
		Cf.clear();
		Ca.shrink_to_fit();
		Cb.shrink_to_fit();
		Cf.shrink_to_fit();

		printf("sending encrypted a/b/f to party0.\n");
		start_timer_map["sending encrypted a/b/f"][0] = std::chrono::system_clock::now();
		auto socket_send = P.socket(0);
		CaStream_other.Send(socket_send);
		CbStream_other.Send(socket_send);
		CfStream_other.Send(socket_send);
		end_timer_map["sending encrypted a/b/f"][0] = std::chrono::system_clock::now();
		printf("sending encrypted a/b/f to party0 done.\n");

		octetStream CcfStream;
		CcfStream.resize_precise(batch_size * Cc[0].report_size(USED));
		CcfStream.reset_write_head();

		printf("receiving masked c+f.\n");
		start_timer_map["receiving masked c+f"][0] = std::chrono::system_clock::now();

		auto socket_recv = P.socket(0);
		CcfStream.Receive(socket_recv);

		for(int i = 0; i < batch_size; i++){
			Cc[i].unpack(CcfStream);
		}
		printf("receiving masked c+f done.\n");
		end_timer_map["receiving masked c+f"][0] = std::chrono::system_clock::now();
	}

	AddableMatrix<bigint> vv;
	vv.resize(batch_size, pk.get_params().phi_m());
	bigint limit = pk.get_params().Q() << 64;
	vv.allocate_slots(limit);

	/*
	* Step 4:
	* For the first party, receiving others' partial decryption vv, decrypting those ciphertexts as c, and saving c = c - f.
	* For the rest of the parties, sending the partial decryption vv and saving c = -f
	*/

	printf("making distributed decryption.\n");
	start_timer_map["making distributed decryption"][0] = std::chrono::system_clock::now();
	for(int i = 0; i < batch_size; i++){
		(*psk).dist_decrypt_1(vv[i], Cc[i], my_num, num_players);
	}
	printf("making distributed decryption done.\n");
	end_timer_map["making distributed decryption"][0] = std::chrono::system_clock::now();

	if(my_num == 0){
		AddableMatrix<bigint> vv1;
		vv1.resize(batch_size, pk.get_params().phi_m());
		vv1.allocate_slots(limit);

		vector<octetStream> vvStream_others(num_players);
		for(int j = 1; j < num_players; j++){
			vvStream_others[j].resize_precise(vv.report_size(USED));
			vvStream_others[j].reset_write_head();
		}

		start_timer_map["receiving vv for distributed decryption"][0] = std::chrono::system_clock::now();
		vector<future<void>> res;
		for(int j = 1; j < num_players; j++){
			int party = j;
			res.push_back(pool->enqueue([party, &P, &vvStream_others]() {
				printf("party 0: receiving vv for distributed decryption from party %d.\n", party);
				auto socket_recv = P.socket(party);
				vvStream_others[party].Receive(socket_recv);
				printf("party 0: receiving vv for distributed decryption from party %d, done.\n", party);
			}));
		}
		joinNclean(res);
		end_timer_map["receiving vv for distributed decryption"][0] = std::chrono::system_clock::now();

		start_timer_map["finishing distributed decryption"][0] = std::chrono::system_clock::now();
		printf("party 0: finishing distributed decryption.\n");
		for (int j = 1; j < num_players; j++) {
			vv1.unpack(vvStream_others[j]);

			for (int i = 0; i < batch_size; i++){
				(*psk).dist_decrypt_2(vv[i],vv1[i]);
			}
		}
		printf("party 0: finishing distributed decryption done.\n");
		end_timer_map["finishing distributed decryption"][0] = std::chrono::system_clock::now();

		start_timer_map["setting c = c - f"][0] = std::chrono::system_clock::now();
		printf("party 0: setting c = c - f.\n");
		bigint mod = params.p0();
		for(int i = 0; i < batch_size; i++){
			c[i].set_poly_mod(vv[i], mod);
			sub(c[i], c[i], f[i]);
		}
		end_timer_map["setting c = c - f"][0] = std::chrono::system_clock::now();
		printf("party 0: setting c = c - f done.\n");
	}else{
		octetStream vvStream;
		vvStream.resize_precise(vv.report_size(USED));
		vvStream.reset_write_head();

		vv.pack(vvStream);

		start_timer_map["sending vv for distributed decryption"][0] = std::chrono::system_clock::now();
		printf("sending vv for distributed decryption.\n");
		auto socket_send = P.socket(0);
		vvStream.Send(socket_send);
		printf("sending vv for distributed decryption done.\n");
		end_timer_map["sending vv for distributed decryption"][0] = std::chrono::system_clock::now();

		start_timer_map["setting c = - f"][0] = std::chrono::system_clock::now();
		printf("setting c = - f.\n");
		c = f;
		for(int i = 0; i < batch_size; i++){
			c[i].negate();
		}
		end_timer_map["setting c = - f"][0] = std::chrono::system_clock::now();
		printf("setting c = - f done.\n");
	}

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

	FHE_PK *ppk;
	FHE_SK *psk;
	FHE_Params params;
	FD FieldD;

	FHE_keygen(ppk, psk, P, plainlength, params, FieldD);

	/*
	* The current ThreadPool size 8 is smaller than the number of parties.
	*/
	ThreadPool pool(64);
	PlaintextVector<FD> a;
	PlaintextVector<FD> b;
	PlaintextVector<FD> c;
	int num_rounds = 10;
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
