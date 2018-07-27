#include <stdio.h>

#include "../common.h"

// TODO textures (easy)
// TODO shared memory (advanced)
__global__
static void boxfilter(int nx, int ny, int *dst, const int *src)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    // sum 3x3 neighborhood
    int sum = 0;
    for (int xx = max(0,x-1); xx <= min(nx-1,x+1); ++xx) {
        for (int yy = max(0,y-1); yy <= min(ny-1,y+1); ++yy)
            sum += src[nx * yy + xx];
    }

    dst[nx * y + x] = sum;
}

// write ones (1) to output array  (ignore performance here)
__global__
static void ones(unsigned n, int *d_out)
{
    const unsigned i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        d_out[i] = 1;
}


int main()
{
    int nx = 16, ny = 16; // TODO larger sizes

    // allocate and populate input (all ones)
    int *d_src, *d_dst;
    size_t bytes = nx*ny*sizeof(*d_src);
    CUDA(cudaMalloc(&d_src, bytes));
    CUDA(cudaMalloc(&d_dst, bytes));
    CUDA(cudaMemset(d_src, 0, bytes)); // zero-out input
    CUDA(cudaMemset(d_dst, 0, bytes)); // zero-out output
    ones<<<divup(nx*ny, 4), 4>>>(nx*ny, d_src);
    CUDA(cudaGetLastError());

    // create events
    cudaEvent_t start, stop;
    CUDA(cudaEventCreate(&start));
    CUDA(cudaEventCreate(&stop));

    // filter
    dim3 thr(8,8);
    dim3 blk(divup(nx, thr.x), divup(ny, thr.y));
    CUDA(cudaEventRecord(start, 0));
    boxfilter<<<blk,thr>>>(nx, ny, d_dst, d_src);
    CUDA(cudaGetLastError());
    CUDA(cudaEventRecord(stop, 0));

    // time kernel
    CUDA(cudaEventSynchronize(stop));
    float time_ms = 0;
    CUDA(cudaEventElapsedTime(&time_ms, start, stop));
    bytes = (9+1)*nx*ny*sizeof(*d_src); // 9 read, 1 write
    printf("bandwidth: %f GB/s\n",  bytes/double(1<<30) / (time_ms/1e3));

    // print upper-left corner to verify
    bytes = nx*8*sizeof(*d_dst); // max needed since only care about corner
    int *h_dst = (int *)malloc(bytes);
    CUDA(cudaMemcpy(h_dst, d_dst, bytes, cudaMemcpyDeviceToHost));
    for (int y = 0; y < min(8,ny); ++y) {
        for (int x = 0; x < min(13,nx); ++x)
            printf("%5d ", h_dst[y*nx + x]);
        putchar('\n');
    }

    return 0;
}
