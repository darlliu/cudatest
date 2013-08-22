#include "test2.h"
int main ()
{
    testcls < ParallelDataThrust, GPUMemoryThrust > t;
    t.test();
    return 1;
};
