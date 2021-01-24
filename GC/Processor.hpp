/*
 * Processor.cpp
 *
 */

#include <GC/Processor.h>

#include <iostream>
#include <iomanip>
using namespace std;

#include "GC/Program.h"
#include "Secret.h"
#include "Access.h"
#include "ReplicatedSecret.h"

namespace GC
{

template <class T>
Processor<T>::Processor(Machine<T>& machine) :
		machine(machine), PC(0), time(0),
		complexity(0)
{
}

template<class T>
Processor<T>::~Processor()
{
	cerr << "Finished after " << time << " instructions" << endl;
}

template <class T>
void Processor<T>::reset(const Program<T>& program, int arg)
{
    S.resize(program.num_reg(SBIT), "registers");
    C.resize(program.num_reg(CBIT), "registers");
    I.resize(program.num_reg(INT), "registers");
    set_arg(arg);
    PC = 0;
}

template <class T>
void Processor<T>::reset(const Program<T>& program)
{
    reset(program, 0);
    machine.reset(program);
}

template<class T>
inline long long GC::Processor<T>::get_input(int n_bits, bool interactive)
{
    long long res = ProcessorBase::get_input(interactive);
    check_input(res, n_bits);
    return res;
}

template<class T>
void GC::Processor<T>::check_input(long long in, int n_bits)
{
	auto test = in >> (n_bits - 1);
	if (n_bits == 1)
	{
		if (not (in == 0 or in == 1))
			throw runtime_error("input not a bit: " + to_string(in));
	}
	else if (not (test == 0 or test == -1))
	{
		throw runtime_error(
				"input too large for a " + std::to_string(n_bits)
						+ "-bit signed integer: " + to_string(in));
	}
}

template <class T>
void Processor<T>::bitdecc(const vector<int>& regs, const Clear& x)
{
    for (unsigned int i = 0; i < regs.size(); i++)
        C[regs[i]] = (x >> i) & 1;
}

template <class T>
void Processor<T>::bitdecint(const vector<int>& regs, const Integer& x)
{
    for (unsigned int i = 0; i < regs.size(); i++)
        I[regs[i]] = (x >> i) & 1;
}

template<class T>
void GC::Processor<T>::load_dynamic_direct(const vector<int>& args)
{
    vector< ReadAccess<T> > accesses;
    if (args.size() % 3 != 0)
        throw runtime_error("invalid number of arguments");
    for (size_t i = 0; i < args.size(); i += 3)
        accesses.push_back({S[args[i]], args[i+1], args[i+2], complexity});
    T::load(accesses, machine.MD);
}

template<class T>
void GC::Processor<T>::load_dynamic_indirect(const vector<int>& args)
{
    vector< ReadAccess<T> > accesses;
    if (args.size() % 3 != 0)
        throw runtime_error("invalid number of arguments");
    for (size_t i = 0; i < args.size(); i += 3)
        accesses.push_back({S[args[i]], C[args[i+1]], args[i+2], complexity});
    T::load(accesses, machine.MD);
}

template<class T>
void GC::Processor<T>::store_dynamic_direct(const vector<int>& args)
{
    vector< WriteAccess<T> > accesses;
    if (args.size() % 2 != 0)
        throw runtime_error("invalid number of arguments");
    for (size_t i = 0; i < args.size(); i += 2)
        accesses.push_back({args[i+1], S[args[i]]});
    T::store(machine.MD, accesses);
    complexity += accesses.size() / 2 * T::default_length;
}

template<class T>
void GC::Processor<T>::store_dynamic_indirect(const vector<int>& args)
{
    vector< WriteAccess<T> > accesses;
    if (args.size() % 2 != 0)
        throw runtime_error("invalid number of arguments");
    for (size_t i = 0; i < args.size(); i += 2)
        accesses.push_back({C[args[i+1]], S[args[i]]});
    T::store(machine.MD, accesses);
    complexity += accesses.size() / 2 * T::default_length;
}

template<class T>
void GC::Processor<T>::store_clear_in_dynamic(const vector<int>& args)
{
    vector<ClearWriteAccess> accesses;
	check_args(args, 2);
    for (size_t i = 0; i < args.size(); i += 2)
    	accesses.push_back({C[args[i+1]], C[args[i]]});
    T::store_clear_in_dynamic(machine.MD, accesses);
}

template <class T>
void Processor<T>::xors(const vector<int>& args)
{
    check_args(args, 4);
    size_t n_args = args.size();
    for (size_t i = 0; i < n_args; i += 4)
    {
        S[args[i+1]].xor_(args[i], S[args[i+2]], S[args[i+3]]);
#ifndef FREE_XOR
        complexity += args[i];
#endif
    }
}

template <class T>
void Processor<T>::and_(const vector<int>& args, bool repeat)
{
    check_args(args, 4);
    for (size_t i = 0; i < args.size(); i += 4)
    {
        S[args[i+1]].and_(args[i], S[args[i+2]], S[args[i+3]], repeat);
        complexity += args[i];
    }
}

template <class T>
void Processor<T>::input(const vector<int>& args)
{
    check_args(args, 3);
    for (size_t i = 0; i < args.size(); i += 3)
    {
        int n_bits = args[i + 1];
        S[args[i+2]] = T::input(args[i] + 1, *this, n_bits);
#ifdef DEBUG_INPUT
        cout << "input to " << args[i+2] << "/" << &S[args[i+2]] << endl;
#endif
    }
}

template <class T>
void Processor<T>::print_reg(int reg, int n)
{
#ifdef DEBUG_VALUES
    cout << "print_reg " << typeid(T).name() << " " << reg << " " << &C[reg] << endl;
#endif
    T::out << "Reg[" << reg << "] = " << hex << showbase << C[reg] << dec << " # ";
    print_str(n);
    T::out << endl << flush;
}

template <class T>
void Processor<T>::print_reg_plain(Clear& value)
{
    T::out << hex << showbase << value << dec << flush;
}

template <class T>
void Processor<T>::print_reg_signed(unsigned n_bits, Clear& value)
{
    unsigned n_shift = 0;
    if (n_bits > 1)
        n_shift = sizeof(value.get()) * 8 - n_bits;
    T::out << dec << (value.get() << n_shift >> n_shift) << flush;
}

template <class T>
void Processor<T>::print_chr(int n)
{
    T::out << (char)n << flush;
}

template <class T>
void Processor<T>::print_str(int n)
{
    T::out << string((char*)&n,sizeof(n)) << flush;
}

template <class T>
void Processor<T>::print_float(const vector<int>& args)
{
    T::out << bigint::get_float(C[args[0]], C[args[1]], C[args[2]], C[args[3]]) << flush;
}

template <class T>
void Processor<T>::print_float_prec(int n)
{
    T::out << setprecision(n);
}

} /* namespace GC */
