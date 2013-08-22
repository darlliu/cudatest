#include "test2.h"
//#include <iostream>
struct zero2
{
    zero2 () {};
    __host__ __device__
    float operator () (const float& in) const
    {
        return 11.4;
    };
};
template <  class Data, class Mem, template <class D> class Num  >
void testcls <Data, Mem, Num>:: test()
{
    Float_1d tmp = this->template malloc<Float, Float_1d>(16) ;
    this -> template map <zero2> (tmp, 16);
    Float_2d tmp2 = this -> template malloc <Float, Float_2d > (8,8);
    this -> template map <zero2> (tmp2, 8, 8);
    //std::cout << tmp2[4][1]<<std::endl;
    this -> template free < Float_2d > (tmp2);
    return;
};
template void testclsinit::test();
