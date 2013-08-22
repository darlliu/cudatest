#include "test.h"
int main ()
{
    GPUMemoryThrust mems;
    thrust::device_vector <int> tmp (16);
    //errors below
    mems.free < device_vector<int> > (int);
    mems.template free < device_vector<int> > (int);
    return 1;
};
