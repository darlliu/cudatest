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
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
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

   Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
   float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
   ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


   In this assignment we will do 800 iterations.
 */

#define xdim 16
#define ydim 16

#include "utils.h"
#include <thrust/host_vector.h>
#include "reference_calc.cpp"
#include <stdio.h>
__device__ inline int getidx()

{
    return threadIdx.x+\
        blockDim.x*(threadIdx.y+\
                blockIdx.x*blockDim.y+\
                blockIdx.y*gridDim.x*blockDim.y);
};
__device__ inline int getidx(const int xoff, const int yoff)
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

    return x+\
        blockDim.x*(y+\
                (blockIdx.x+bx)*blockDim.y+\
                (blockIdx.y+by)*gridDim.x*blockDim.y);
};
__device__ inline int gettx()
{
    return threadIdx.x+threadIdx.y*blockDim.x;
};
__device__ inline int gettx(const int x, const int y)
{
    return x+y*blockDim.x;
}
__device__ inline int getoffset()
{
    return blockDim.x*blockDim.y*gridDim.x*gridDim.y;
};
// For this project we use a 2d grid and a 2d block

template <typename S, typename T>
    inline void swap (const S* a, const T* b)
{
    void* tmp;
    tmp = (void*)b;
    b = (T*)((void*)a);
    a = (S*)((void*)b);
    return;
};

__global__ void routine1(const uchar4* const d_src,
        int* const d_i, float* const d_r, float* const d_g, float * const d_b,
        const int len)
{
    int idx = getidx();
    if (idx >= len) return;
    uchar4 val = d_src[idx];
    int flag = 1;
    flag = val.x == 255 ? 0: flag;
    flag = val.y == 255 ? 0: flag;
    flag = val.z == 255 ? 0: flag;
    //consider changing this to iff
    d_r[idx] = (float)val.x;
    d_g[idx] = (float)val.y;
    d_b[idx] = (float)val.b;
    d_i[idx] = flag;
    return;
};
void your_blend(const uchar4* const h_sourceImg,  //IN
        const size_t numRowsSource, const size_t numColsSource,
        const uchar4* const h_destImg, //IN
        uchar4* const h_blendedImg) //OUT
{
    const unsigned int len = numRowsSource*numColSource, r = (numRowsSource+bdim-1)/xdim,\
                             c = (numRowsSource+bdim-1)/ydim;
    uchar4* d_src ;
    checkCudaErrors(cudaMalloc(d_src, sizeof(uchar4)*len));
    checkCudaErrors(cudaMemcpy(d_src, h_sourceImg, cudaMemcpyHostToDevice));
    float *d_r1, *d_r2, *d_g1, *d_g2, *d_b1, *d_b2;
    checkCudaErrors(cudaMalloc(d_r1, sizeof(float)*len));
    checkCudaErrors(cudaMalloc(d_r1, sizeof(float)*len));
    checkCudaErrors(cudaMalloc(d_g1, sizeof(float)*len));
    checkCudaErrors(cudaMalloc(d_g2, sizeof(float)*len));
    checkCudaErrors(cudaMalloc(d_b1, sizeof(float)*len));
    checkCudaErrors(cudaMalloc(d_b2, sizeof(float)*len));

    int *d_i1, *d_i2;
    checkCudaErrors(cudaMalloc(d_i1, sizeof(int)*len));
    checkCudaErrors(cudaMalloc(d_i2, sizeof(int)*len));


    int* h_tt = (int*) malloc(sizeof(int)*len);
    routine1  <<<dim3(r,c,1), dim3(xdim,ydim,1) >>> (
            d_src,d_i, d_r1, d_g1, d_b1, len );

    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(h_tt, d_i, cudaMemcpyDeviceToHost));
 
    int sum = 0;
    for (int i = 0; i<len; ++i)
    {
        sum+=h_tt[i];
    }
    printf("total is %d, got %d\n in", len, sum);
    /* To Recap here are the steps you need to implement

       1) Compute a mask of the pixels from the source image to be copied
       The pixels that shouldn't be copied are completely white, they
       have R=255, G=255, B=255.  Any other pixels SHOULD be copied.

       2) Compute the interior and border regions of the mask.  An interior
       pixel has all 4 neighbors also inside the mask.  A border pixel is
       in the mask itself, but has at least one neighbor that isn't.

       3) Separate out the incoming image into three separate channels

       4) Create two float(!) buffers for each color channel that will
       act as our guesses.  Initialize them to the respective color
       channel of the source image since that will act as our intial guess.

       5) For each color channel perform the Jacobi iteration described 
       above 800 times.

       6) Create the output image by replacing all the interior pixels
       in the destination image with the result of the Jacobi iterations.
       Just cast the floating point values to unsigned chars since we have
       already made sure to clamp them to the correct range.

       Since this is final assignment we provide little boilerplate code to
       help you.  Notice that all the input/output pointers are HOST pointers.

       You will have to allocate all of your own GPU memory and perform your own
       memcopies to get data in and out of the GPU memory.

       Remember to wrap all of your calls with checkCudaErrors() to catch any
       thing that might go wrong.  After each kernel call do:

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
