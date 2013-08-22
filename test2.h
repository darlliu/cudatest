#include "test.h"

template <  class Data, class Mem  >
class testcls: public Data, public Mem
{
    public:
    typedef typename Data::Int_1d Int_1d;
    typedef typename Data::Int Int;
    testcls(){};
    void test();
};


typedef testcls < ParallelDataThrust, GPUMemoryThrust > testclsinit;
