#include <stdio.h>

#include "../common.h"


__inline__
__device__ // afterward: (a <= b)
static void cmpswap(int &a, int &b)
{
    if (a > b) {
        int tmp = a;
        a = b;
        b = tmp;
    }
}

texture<int> tex;

__global__
static void median2(int nx, int ny, int *d_dst, const int *d_src, size_t offset)
{
    // TODO get image window (global memory or textures)
    // TODO (advanced) shared memory
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    int v[9] = {0,0,0, 0,0,0, 0,0,0};
    int index = 0;
    for (int yy = max(0,y-1); yy <= min(ny-1,y+1); ++yy) {
        for (int xx = max(0,x-1); xx <= min(nx-1,x+1); ++xx)
            v[index++] = tex1Dfetch(tex, yy*nx + xx + offset);
    }

    // TODO sort and get the median value (bubble sort)
    // TODO (advanced) 3x3 exchange sort: http://graphics.cs.williams.edu/papers/MedianShaderX6/median.pix
	for (int i = 0; i < 9; ++i) {
		for (int j = i + 1; j < 9; j++)
            cmpswap(v[i], v[j]);
	}

    // TODO store the value in the output image
    d_dst[y*nx + x] = v[4];
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

    size_t offset;
    CUDA(cudaBindTexture(&offset, tex, d_src, bytes));

    // filter
    dim3 thr(8,8);
    dim3 blk(divup(nx, thr.x), divup(ny, thr.y));
    CUDA(cudaEventRecord(start, 0));
    median2<<<blk,thr>>>(nx, ny, d_dst, d_src, offset);
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
