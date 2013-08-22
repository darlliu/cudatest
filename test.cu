#include "Parallel_Thrust.h"

struct zero
{
    zero () {};
    __host__ __device__
    float operator () (const float& in) const
    {
        return 0.f;
    };
};
int main ()
{
    ParallelDataThrust types;
    GPUMemoryThrust mems;
    ParallelComputeThrust<ParallelDataThrust> computes;
    thrust::device_vector <int> tmp (16);
    mems.free < device_vector<int> > (int);
    return 1;
};
