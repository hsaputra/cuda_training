#include <stdio.h>

#include "../common.h"


__global__
static void median2(int nx, int ny, int *d_dst, const int *d_src)
{
    // TODO get image window (global memory or textures)
    // TODO (advanced) shared memory

    // TODO sort and get the median value (bubble sort)
    // TODO (advanced) 3x3 exchange sort: http://graphics.cs.williams.edu/papers/MedianShaderX6/median.pix

    // TODO store the value in the output image
}

int main()
{
    int nx = 1024, ny = 1024; // TODO larger sizes

    int *h_src = (int *)malloc(nx*ny*sizeof(*h_src));
    for (int x = 0; x < nx; ++x) {
        for (int y = 0; y < ny; ++y)
            h_src[y*nx + x] = 10 * float(rand())/RAND_MAX; // [0-10]
    }

    // print upper-left corner to verify
    printf("source:\n");
    for (int y = 0; y < min(8,ny); ++y) {
        for (int x = 0; x < min(13,nx); ++x)
            printf("%5d ", h_src[y*nx + x]);
        putchar('\n');
    }

    // allocate and populate input (all ones)
    int *d_src, *d_dst;
    size_t bytes = nx*ny*sizeof(*d_src);
    CUDA(cudaMalloc(&d_src, bytes));
    CUDA(cudaMalloc(&d_dst, bytes));
    CUDA(cudaMemcpy(d_src, h_src, bytes, cudaMemcpyHostToDevice));
    CUDA(cudaMemset(d_dst, 0, bytes)); // zero-out output

    // create events
    cudaEvent_t start, stop;
    CUDA(cudaEventCreate(&start));
    CUDA(cudaEventCreate(&stop));

    // filter
    dim3 thr(8,8);
    dim3 blk(divup(nx, thr.x), divup(ny, thr.y));
    CUDA(cudaEventRecord(start, 0));
    median2<<<blk,thr>>>(nx, ny, d_dst, d_src);
    CUDA(cudaGetLastError());
    CUDA(cudaEventRecord(stop, 0));

    // time kernel
    CUDA(cudaEventSynchronize(stop));
    float time_ms = 0;
    CUDA(cudaEventElapsedTime(&time_ms, start, stop));

    // print upper-left corner to verify
    bytes = nx*8*sizeof(*d_dst); // max needed since only care about corner
    int *h_dst = (int *)malloc(bytes);
    CUDA(cudaMemcpy(h_dst, d_dst, bytes, cudaMemcpyDeviceToHost));
    printf("destination:\n");
    for (int y = 0; y < min(8,ny); ++y) {
        for (int x = 0; x < min(13,nx); ++x)
            printf("%5d ", h_dst[y*nx + x]);
        putchar('\n');
    }

    bytes = (9+1)*nx*ny*sizeof(*d_src); // 9 read, 1 write
    printf("bandwidth: %f GB/s\n",  bytes/double(1<<30) / (time_ms/1e3));

    return 0;
}
