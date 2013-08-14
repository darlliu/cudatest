//Udacity HW 6
//Poisson Blending

/* Background
==========

The goal for this assignment is to take one image (the source) and
paste it into another image (the destination) attempting to match the
two images so that the pasting is non-obvious. This is
known as a "seamless clone".

The basic ideas are as follows:

1) Figure out the interior and border of the source image
2) Use the values of the border pixels in the destination image
as boundary conditions for solving a Poisson equation that tells
us how to blend the images.

No pixels from the destination except pixels on the border
are used to compute the match.

Solving the Poisson Equation
============================

There are multiple ways to solve this equation - we choose an iterative
method - specifically the Jacobi method. Iterative methods start with
a guess of the solution and then iterate to try and improve the guess
until it stops changing. If the problem was well-suited for the method
then it will stop and where it stops will be the solution.

The Jacobi method is the simplest iterative method and converges slowly -
that is we need a lot of iterations to get to the answer, but it is the
easiest method to write.

Jacobi Iterations
=================

Our initial guess is going to be the source image itself. This is a pretty
good guess for what the blended image will look like and it means that
we won't have to do as many iterations compared to if we had started far
from the final solution.

ImageGuess_prev (Floating point)
ImageGuess_next (Floating point)

DestinationImg
SourceImg

Follow these steps to implement one iteration:

1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
else if the neighbor in on the border then += DestinationImg[neighbor]

Sum2: += SourceImg[p] - SourceImg[neighbor] (for all four neighbors)

2) Calculate the new pixel value:
float newVal= (Sum1 + Sum2) / 4.f <------ Notice that the result is FLOATING POINT
ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


In this assignment we will do 800 iterations.
*/

#define xdim 16
#define ydim 16
#define bdim xdim*ydim
#include "utils.h"
#include <thrust/host_vector.h>
#include "reference_calc.cpp"
#include <stdio.h>
__device__ inline int getidx(const int r, const int c)

{
    int xx = threadIdx.x+ blockDim.x*blockIdx.x;
    if (xx > r || xx < 0) return -1;
    int yy = threadIdx.y + blockIdx.y*blockDim.y;
    if (yy > c || yy < 0) return -1;
    return xx + r*yy;
    // row major
};
__device__ inline int getidx(const int xoff, const int yoff, const int r, const int c)
{
    int x,y, bx, by;

    x = xoff<0?blockDim.x-xoff:xoff; // this is 0, 1 or 0
    bx = xoff<0?-1:1;
    x= xoff==0?threadIdx.x:x; // this eliminates the 0 condition
    bx = xoff==0?0:bx;

    y = yoff<0?blockDim.y-xoff:yoff;
    by = yoff<0?-1:1;
    y= yoff==0?threadIdx.y:y;
    by = yoff==0?0:by;

    int xx = x+ blockDim.x*(blockIdx.x+bx);
    if (xx > r || xx < 0) return -1;
    int yy = y+ blockDim.y*(blockIdx.y+by);
    if (yy > c || yy < 0) return -1;
    return xx + r*yy;
};
__device__ inline int gettx()
{
    return threadIdx.x+threadIdx.y*blockDim.x;
};
__device__ inline int gettx(const int x, const int y)
{
    return x+y*blockDim.x;
};
__device__ inline int getoffset()
{
    return blockDim.x*blockDim.y*gridDim.x*gridDim.y;
};
// For this project we use a 2d grid and a 2d block

__global__ void routine0(
        const uchar4* const d_src, float* const d_r, float* const d_g, float * const d_b,
        const int r, const int c)
{
    int idx = getidx(r,c);
    if (idx == -1) return;
    uchar4 val = d_src[idx];
    d_r[idx] = (float)val.x;
    d_g[idx] = (float)val.y;
    d_b[idx] = (float)val.z;
    return;
};

__global__
void swap (float* a, float* b, const int r, const int c)
{

    if (getidx(r,c)!=0) return;
    void* tmp;
    tmp = (void*)b;
    b = (float*)((void*)a);
    a = (float*)((void*)tmp);
    return;
};

__global__ void routine1(const uchar4* const d_src,
        int* const d_i, float* const d_r, float* const d_g, float * const d_b,
        const int r, const int c)
{
    int idx = getidx(r,c);
    if (idx == -1) return;
    uchar4 val = d_src[idx];
    int flag = 1;
    flag = val.x == 255 ? 0: flag;
    flag = val.y == 255 ? 0: flag;
    flag = val.z == 255 ? 0: flag;

    d_i[idx] = flag;
    if (flag==0) return;
    //consider changing this to iff
    d_r[idx] = (float)val.x;
    d_g[idx] = (float)val.y;
    d_b[idx] = (float)val.z;

    return;
};

__global__ void routine2(const int* const in, int* const out, const int r, const int c)
    //Computes neighboring condition
{
    volatile __shared__ int tmp[bdim];
    int idx = getidx(r,c);
    int tx = gettx();
    tmp[tx]=0;
    __syncthreads();
    if (idx == -1) return;
    int val = in[idx];
    tmp[tx]=val;
    __syncthreads();
    int idx2, val2, flag=2;
    if (val == 1)
        for (int xoff = -1; xoff < 2; ++xoff)
            for (int yoff = -1; yoff < 2; ++yoff)
            {
                if ((int)threadIdx.x+xoff>=0 && (int)threadIdx.x+xoff<xdim\
                        && (int)threadIdx.y+yoff>=0 && (int)threadIdx.y+yoff<ydim)
                {
                    idx2 = gettx(xoff+(int)threadIdx.x, yoff+(int)threadIdx.y);
                    val2 = tmp[idx2];
                }
                else
                {
                    idx2 = getidx(xoff,yoff,r,c);
                    val2 = in[idx2];
                }
                flag = val2 == 0? 1:flag;
            }
    else flag=0;
    out[idx]= flag;
    return;
};

__global__ void jacobi
        (const float* const in, float* const out,const float* const dst,
         const int* const flags, const int r, const int c)
//jacobi routine note that each successive cycle should be followed by a pointer swap
{
    volatile __shared__ int tmp_flags[bdim];
    volatile __shared__ float tmp[bdim];
    volatile __shared__ float tmp2[bdim];
    int idx = getidx(r,c);
    int tx = gettx();
    tmp[tx]=0;
    tmp_flags[tx]=0;
    __syncthreads();
    if (idx==-1) return;
    // initialize the shared memory then close most threads
    float val = in[idx], dst_val=dst[idx];
    int flag = flags[idx];
    tmp_flags[tx]=flag;
    tmp[tx] = val;
    tmp2[tx]=dst_val;
    __syncthreads();
    //initialize shared cache to actual interate vals

    float val2, val3;
    int idx2, flag2;
    float sum1=0, sum2=0;
    int xoff, yoff;
    float oo;
    // variable allocations
    
    if (flag == 2) // only for interior pts
    {

        for (xoff = -1; xoff != 2; ++xoff)
            for (yoff = -1; yoff != 2; ++yoff)
            {
                if ((int)threadIdx.x+xoff>=0 && (int)threadIdx.x+xoff<xdim\
                        && (int)threadIdx.y+yoff>=0 && (int)threadIdx.y+yoff<ydim)
                {
                    idx2 = gettx(xoff+(int)threadIdx.x, yoff+(int)threadIdx.y);
                    flag2 = tmp_flags[idx2];
                    val2 = tmp[idx2];
                    val3 = tmp2[idx2];
                }
                else
                {
                    idx2 = getidx(xoff,yoff,r,c);
                    flag2 = flags[idx2];
                    val2 = in[idx2];
                    val3 = dst[idx2];
                }
                if (flag2 == 2)
                {
                    sum1 += val2;
                }
                else if (flag2 ==1)
                {
                    sum1 += val3;
                }
                else continue;
                sum2 += val-val2;
            }
        oo = min(255.f, max(0.f, (sum1+sum2)/4.f));
    }
    else
    {
        oo = val;
    }
    out[idx]=oo;
    return;
};

void your_blend(const uchar4* const h_sourceImg, //IN
        const size_t numRowsSource, const size_t numColsSource,
        const uchar4* const h_destImg, //IN
        uchar4* const h_blendedImg) //OUT
{
    const unsigned int len = numRowsSource*numColsSource, r = (numRowsSource+xdim-1)/xdim,\
                             c = (numColsSource+ydim-1)/ydim;
    uchar4* d_src , *d_dst;
    //checkCudaErrors(cudaHostRegister((void*)h_sourceImg,sizeof(uchar4)*len,cudaHostRegisterPortable));
    checkCudaErrors(cudaMalloc((void **)&d_src, sizeof(uchar4)*len));
    checkCudaErrors(cudaMalloc((void **)&d_dst, sizeof(uchar4)*len));
    checkCudaErrors(cudaMemcpy(d_src, h_sourceImg, sizeof(uchar4)*len, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_dst, h_destImg, sizeof(uchar4)*len, cudaMemcpyHostToDevice));
    
    float *d_r1, *d_r2, *d_g1, *d_g2, *d_b1, *d_b2;
    checkCudaErrors(cudaMalloc((void **)&d_r1, sizeof(float)*len));
    checkCudaErrors(cudaMalloc((void **)&d_g1, sizeof(float)*len));
    checkCudaErrors(cudaMalloc((void **)&d_b1, sizeof(float)*len));

    int *d_i1, *d_i2;
    checkCudaErrors(cudaMalloc((void **)&d_i1, sizeof(int)*len));
    checkCudaErrors(cudaMalloc((void **)&d_i2, sizeof(int)*len));
    dim3 G(r,c,1), B(xdim,ydim,1)
    float *d_dr, *d_dg, *d_db;
    checkCudaErrors(cudaMalloc((void **)&d_dr, sizeof(float)*len));
    checkCudaErrors(cudaMalloc((void **)&d_dg, sizeof(float)*len));
    checkCudaErrors(cudaMalloc((void **)&d_db, sizeof(float)*len));
    routine0 <<<G,B >>> (
            d_dst,d_dr, d_dg, d_db, numRowsSource, numColsSource );
    checkCudaErrors(cudaDeviceSynchronize()); checkCudaErrors(cudaGetLastError());
    
    routine1 <<<G,B>>> (
            d_src,d_i1, d_r1, d_g1, d_b1, numRowsSource, numColsSource );
    checkCudaErrors(cudaDeviceSynchronize()); checkCudaErrors(cudaGetLastError());
    
    routine2 <<<G,B >>> (d_i1, d_i2, numRowsSource, numColsSource );
    checkCudaErrors(cudaDeviceSynchronize()); checkCudaErrors(cudaGetLastError());

    cudaFree(d_i1);

    cudaStream_t sa, sb, sc;
    cudaStreamCreate(&sa);
    cudaStreamCreate(&sb);
    cudaStreamCreate(&sc);

    cudaMallocAsync((void**) &d_r2,sizeof(float)*len, sa);
    cudaMemcpyAsync(d_r2, d_r1, sizeof(float)*len, cudaMemcpyDeviceToDevice,sa);

    cudaMallocAsync((void**) &d_g2,sizeof(float)*len, sb);
    cudaMemcpyAsync(d_g2, d_g1, sizeof(float)*len, cudaMemcpyDeviceToDevice,sb);

    cudaMallocAsync((void**) &d_b2,sizeof(float)*len, sc);
    cudaMemcpyAsync(d_b2, d_b1, sizeof(float)*len, cudaMemcpyDeviceToDevice,sc);

    for (int i =0; i<1; ++i)
    {
        jacobi <<<G,B,0,sa>>> (d_r1, d_r2, d_dr1, d_i2, numRowsSource,numColsSource );
        jacobi <<<G,B,0,sb>>> (d_g1, d_g2, d_dg1, d_i2, numRowsSource,numColsSource );
        jacobi <<<G,B,0,sc>>> (d_b1, d_b2, d_db1, d_i2, numRowsSource,numColsSource );
        swap <<<dim3(1,1,1), dim3(1,1,1),0,sa>>> (d_r1,d_r2, numRowsSource, numColsSource);
        swap <<<dim3(1,1,1), dim3(1,1,1),0,sb>>> (d_g1,d_g2, numRowsSource, numColsSource);
        swap <<<dim3(1,1,1), dim3(1,1,1),0,sc>>> (d_b1,d_b2, numRowsSource, numColsSource);
    }
    checkCudaErrors(cudaDeviceSynchronize());

    int* h_tt = (int*) malloc(sizeof(int)*len);
    checkCudaErrors(cudaMemcpy(h_tt, d_i1,sizeof(int)*len, cudaMemcpyDeviceToHost));
    int sum = 0,sum2=0, sum3=0, sum4=0;
    for (int i = 0; i<len; ++i)
    {
        if (h_sourceImg[i].x!=255&&h_sourceImg[i].y!=255&&h_sourceImg[i].z!=255)
            ++sum2;
        sum+=h_tt[i];
    }
    printf("total is %d, got %d serial %d parallel\n", len,sum2, sum);
    checkCudaErrors(cudaMemcpy(h_tt, d_i2,sizeof(int)*len, cudaMemcpyDeviceToHost));
    for (int i = 0; i<len; ++i)
    {
        if (h_tt[i]==1) sum3++;
        else if (h_tt[i]==2) sum4++;
    }
    printf("total is %d, got %d inner %d boundary\n", sum,sum3, sum4);
    //printf("total is %d, got %d \n", len,h_tt[len-1]+1);
   /* To Recap here are the steps you need to implement

1) Compute a mask of the pixels from the source image to be copied
The pixels that shouldn't be copied are completely white, they
have R=255, G=255, B=255. Any other pixels SHOULD be copied.

2) Compute the interior and border regions of the mask. An interior
pixel has all 4 neighbors also inside the mask. A border pixel is
in the mask itself, but has at least one neighbor that isn't.

3) Separate out the incoming image into three separate channels

4) Create two float(!) buffers for each color channel that will
act as our guesses. Initialize them to the respective color
channel of the source image since that will act as our intial guess.

5) For each color channel perform the Jacobi iteration described
above 800 times.

6) Create the output image by replacing all the interior pixels
in the destination image with the result of the Jacobi iterations.
Just cast the floating point values to unsigned chars since we have
already made sure to clamp them to the correct range.

Since this is final assignment we provide little boilerplate code to
help you. Notice that all the input/output pointers are HOST pointers.

You will have to allocate all of your own GPU memory and perform your own
memcopies to get data in and out of the GPU memory.

Remember to wrap all of your calls with checkCudaErrors() to catch any
thing that might go wrong. After each kernel call do:

cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

to catch any errors that happened while executing the kernel.
*/



    /* The reference calculation is provided below, feel free to use it
for debugging purposes.
*/

    /*
uchar4* h_reference = new uchar4[srcSize];
reference_calc(h_sourceImg, numRowsSource, numColsSource,
h_destImg, h_reference);

checkResultsEps((unsigned char *)h_reference, (unsigned char *)h_blendedImg, 4 * srcSize, 2, .01);
delete[] h_reference; */
}
