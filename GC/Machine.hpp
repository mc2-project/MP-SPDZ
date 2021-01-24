/*
 * Machine.cpp
 *
 */

#include <GC/Machine.h>

#include "GC/Program.h"
#include "Secret.h"
#include "ReplicatedSecret.h"
#include "ThreadMaster.h"

namespace GC
{

template <class T>
Machine<T>::Machine(Memory<typename T::DynamicType>& dynamic_memory) : MD(dynamic_memory)
{
    use_encryption = false;
    more_comm_less_comp = false;
    start_timer();
}

template<class T>
Machine<T>::~Machine()
{
    for (auto it = timer.begin(); it != timer.end(); it++)
        cerr << T::phase_name() << " timer " << it->first << " at end: "
                << it->second.elapsed() << " seconds" << endl;
}

template<class T>
void Machine<T>::load_program(string threadname, string filename)
{
    (void)threadname;
    progs.push_back({});
    progs.back().parse_file(filename);
    reset(progs.back());
}

template<class T>
void Machine<T>::load_schedule(string progname)
{
    BaseMachine::load_schedule(progname);
    for (auto i : {1, 0, 0})
    {
        int n;
        inpf >> n;
        if (n != i)
            throw runtime_error("old schedule format not supported");
    }
    print_compiler();
}

template <class T>
void Machine<T>::reset(const Program<T>& program)
{
    MS.resize_min(program.direct_mem(SBIT), "memory");
    MC.resize_min(program.direct_mem(CBIT), "memory");
    MI.resize_min(program.direct_mem(INT), "memory");
    MD.resize_min(program.direct_mem(DYN_SBIT), "dynamic memory");
}

template<class T>
void Machine<T>::run_tape(int thread_number, int tape_number, int arg)
{
    ThreadMaster<T>::s().run_tape(thread_number, tape_number, arg);
}

template<class T>
void Machine<T>::join_tape(int thread_number)
{
    ThreadMaster<T>::s().join_tape(thread_number);
}

} /* namespace GC */
