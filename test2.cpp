#include "test2.h"
template <  class Data, class Mem  >
void testcls <Data, Mem>:: test()
{
    Int_1d tmp = this->template malloc < Int, Int_1d > (16) ;
    this -> template free < Int_1d > (tmp);
    return;
};
template void testclsinit::test();
