//Policy classes using only the thrust library to provide for
//parallel acceleration. Uses device memory and containers.
//Note: The device vector provided by thrust can be directed accessed
//from host. This is NOT the expected behavior of any other parallel implementation.
//
//This is mainly useful for porting initial parallel code.
//
#ifndef Parallel_Thrust_h
#define Parallel_Thrust_h

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/functional.h>

// Here we overwrite the null value for consistency with thrust vectors
//

using thrust::device_vector;

template <typename T>
struct wrapped_row
{
    // Thrust arrays do not allow multiple layers
    // we have to wrap everything around 1d array
    wrapped_row():offset(0),x(0){};
    wrapped_row(int sz):offset(0),x(0),arr(sz){};
    int offset;
    // Note that this offset is handled at allocation
    // i.e. by the memory class
    int x;
    device_vector<T> arr;
    thrust::device_reference<T> operator[](int y) {return arr[y+x*offset];};
};

template <typename T>
struct arr_2d
{
    arr_2d(){};
    arr_2d(int s):data(s){};
    wrapped_row <T> data;
    wrapped_row <T>& operator[](int x)
    {
        data.x=x;
        return data;
    };
};

class ParallelDataThrust
{
    public:
        typedef int Int;
        typedef float Float;
        typedef device_vector<int> Int_1d;
        typedef device_vector<float> Float_1d;
        typedef arr_2d<int> Int_2d;
        typedef arr_2d<float> Float_2d;
};

class GPUMemoryThrust
{
    public:
        template <typename S, typename S_1d>
            S_1d malloc (const size_t sz)
        {
            return S_1d(sz);
        };
        template <typename S, typename S_2d>
            S_2d malloc (const int x, const int y)
        {
            S_2d tmp;
            tmp.data.arr.resize(x*y);
            tmp.data.offset = y;
            return tmp;
        };
        template <typename S_1d> void free (S_1d in){};
        template <typename S_2d> void free (S_2d in,const int len){};
        // memory is self managed by thrust
};

template <class Data>
class ParallelComputeThrust
{
    typedef typename Data::Float S;
    typedef typename Data::Float_1d S_1d;
    typedef typename Data::Float_2d S_2d;

    public:
        template <class f >
        void map (S_2d arr, const int len)
        {
            thrust::transform (arr.begin(), arr.end(),arr.begin(),f());
        };

        template <class f >
        void map (S_2d arr, const int x, const int y)
        {
            thrust::transform (arr.data.arr.begin(), arr.data.arr.end(),arr.data.arr.begin(), f());
        };

        template <class f >
        S reduceWith (S_1d arr, const int len)
        {
            if (len <1) return (S)0;
            device_vector<S> tmp (arr);
            thrust::transform (tmp.begin(), tmp.end(), tmp.begin(), f());
            return thrust::reduce(tmp.begin(), tmp.end(),(S)0);
        };
        template <class f >
        S reduceWith (S_1d arr1, S_1d arr2, const int len)
        {
            if (len<1) return (S)0;
            device_vector<S> tmp2 (arr2);
            thrust::transform (arr1.begin(), arr1.end(),\
                    tmp2.begin(), tmp2.begin(), f());
            return thrust::reduce(tmp2.begin(), tmp2.end(),(S)0);
        };
        template <class f >
        S reduceWith (wrapped_row<S> arr, const int len)
        {
            if (len <1) return (S)0;
            device_vector<S> tmp (arr.arr.begin()+arr.offset*arr.x, arr.arr.begin()+arr.offset*(arr.x+1));
            thrust::transform (tmp.begin(), tmp.end(),tmp.begin(), f());
            return thrust::reduce(tmp.begin(), tmp.end(),(S)0);
        };
        template <class f >
        S reduceWith (wrapped_row<S> arr1, wrapped_row<S> arr2, const int len)
        {
            if (len<1) return (S)0;
            device_vector<S> tmp1 (arr1.arr.begin()+arr1.offset*arr1.x,\
                    arr1.arr.begin()+arr1.offset*(arr1.x+1));

            device_vector<S> tmp2 (arr2.arr.begin()+arr2.offset*arr2.x,\
                    arr2.arr.begin()+arr2.offset*(arr2.x+1));

            thrust::transform (tmp1.begin(), tmp1.end(),\
                    tmp2.begin(), tmp2.begin(), f());
            return thrust::reduce(tmp2.begin(), tmp2.end(),(S)0);
        };

        template <class f >
        S reduceWith2D (S_2d arr, const int x, const int y)
        {
            return this->template reduceWith <f> (arr.data.arr, x*y);
        };
        template <class f >
        S reduceWith2D (S_2d arr1, S_2d arr2, const int x, const int y)
        {
            return this->template reduceWith <f> (arr1.data.arr, arr2.data.arr, x*y);
        };

};
#endif
