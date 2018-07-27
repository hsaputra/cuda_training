// y=convolve(x,k)

#include <stdio.h>
#include <stdlib.h>  // rand(), RAND_MAX

#include "../common.h"


__global__
static void kernel(float *d_y, int nx, const float *d_x, int nk, const float *d_k)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    if (x >= nx) return;

    float sum = 0;

    // TODO convolution (dot product)
    //   y[0] = x[0]*k[0] + x[1]*k[1] + x[2]*k[2]
    //   y[1] = x[1]*k[0] + x[2]*k[1] + x[3]*k[2]
    //   y[2] = x[2]*k[0] + x[3]*k[1] + x[4]*k[2]
    //   y[3] = x[3]*k[0] + x[4]*k[1] + x[5]*k[2]
    //   ...

    // several extensions to play:
    // TODO textured cached read of d_x (tex1Dfetch)
    // TODO shared mem cached read of d_x
    // TODO __constant__ read of d_k   (cudaMemcpyToSymbol)

    // TODO template<nk>

    d_y[x] = sum;
}

int main()
{
    int nx = 5;
    int nk = 3;

    float *h_x = (float *)malloc(nx*sizeof(*h_x));
    for (int i = 0; i < nx; ++i)
        h_x[i] = 1; //10 * float(rand())/RAND_MAX; // [0-10]
    float *h_k = (float *)malloc(nk*sizeof(*h_k));
    for (int i = 0; i < nk; ++i)
        h_k[i] = 1; // constant kernel: 1

    // print inputs
    printf("x:    ");
    for (int i = 0; i < min(13,nx); ++i)
        printf("%6.2f ", h_x[i]);
    printf("\nk:    ");
    for (int i = 0; i < min(13,nk); ++i)
        printf("%6.2f ", h_k[i]);
    putchar('\n');

    // allocate and populate input (all ones)
    float *d_x, *d_k, *d_y;
    size_t xbytes = nx*sizeof(*d_x);
    size_t kbytes = nk*sizeof(*d_k);
    CUDA(cudaMalloc(&d_x, xbytes));
    CUDA(cudaMalloc(&d_k, kbytes));
    CUDA(cudaMalloc(&d_y, xbytes));
    CUDA(cudaMemcpy(d_x,  h_x, xbytes, cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(d_k,  h_k, kbytes, cudaMemcpyHostToDevice));
    CUDA(cudaMemset(d_y,  0,   xbytes)); // zero-out output

    // create events
    cudaEvent_t start, stop;
    CUDA(cudaEventCreate(&start));
    CUDA(cudaEventCreate(&stop));

    // filter
    CUDA(cudaEventRecord(start, 0));
    int threads = 256;
    kernel<<<divup(nx,threads), threads>>>(d_y, nx, d_x, nk, d_k);
    CUDA(cudaGetLastError());
    CUDA(cudaEventRecord(stop, 0));

    // time kernel
    CUDA(cudaEventSynchronize(stop));
    float time_ms = 0;
    CUDA(cudaEventElapsedTime(&time_ms, start, stop));

    // print upper-left corner to verify
    size_t bytes = nx*sizeof(*d_y); // max needed since only care about corner
    float *h_y = (float *)malloc(bytes);
    CUDA(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));
    printf("y:    ");
    for (int i = 0; i < nx; ++i) {
        if (i < 13)  printf("%6.2f ", h_y[i]);
        if ((i < nx - nk && h_y[i] != nk) || (nx - nk <= i && h_y[i] != nx - i)) {
            printf("\nFAIL at i=%d!", i);
            break;
        }
    }
    putchar('\n');

    double flops = nx * 2*nk; // (nk) multiply and add ops for each input (nx)
    printf("performance: %f GFLOP/s\n", (flops/1e9) / (time_ms/1e3));

    return 0;
}
