#include "Parallel_Thrust.h"

template <  class Data, class Mem , template <class D > class Num >
class testcls: public Data, public Mem, public Num <Data>
{
    public:
    typedef typename Data::Int_1d Int_1d;
    typedef typename Data::Int_2d Int_2d;
    typedef typename Data::Int Int;
    typedef typename Data::Float Float;
    typedef typename Data::Float_1d Float_1d;
    typedef typename Data::Float_2d Float_2d;

    testcls(){};
    void test();
};

typedef testcls < ParallelDataThrust, GPUMemoryThrust, ParallelComputeThrust > testclsinit;
