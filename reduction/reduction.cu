#include <cuda_runtime.h>
#include "../common.h"

static int reduceCPU(int *data, int n)
{
    int sum = 0;
    for (int i = 0; i < n; i++)
        sum += data[i];
    return sum;
}

const int n = (1 << 22);  // number of elements to reduce
const int BLOCK_SIZE = 1024;

__global__
void reduceGPUShuffle(int* d_odata, int* d_idata, int n) {

  // Get Current index for the thread.
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  // Access from global memory
  int threadData = d_idata[index];

  // Sync down within wrap
  for (int offset = warpSize/2; offset > 0; offset /= 2)
  {
    threadData += __shfl_down_sync(0xffffffff, threadData, offset);
  } 

  /**
  threadData += __shfl_down_sync(0xffffffff, threadData, 16);
  threadData += __shfl_down_sync(0xffffffff, threadData, 8);
  threadData += __shfl_down_sync(0xffffffff, threadData, 4);
  threadData += __shfl_down_sync(0xffffffff, threadData, 2);
  threadData += __shfl_down_sync(0xffffffff, threadData, 1);
  */

  if (threadIdx.x % warpSize == 0) {
    atomicAdd(&d_odata[blockIdx.x], threadData);   
  }   
}

__global__ 
void reduceGPU(int* d_odata, int* d_idata, int n) {

  __shared__ int blockReduce[BLOCK_SIZE];

  // Get Current index for the thread.
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  // Coalesce read from global memory and write it to shared memory.
  blockReduce[threadIdx.x] = d_idata[index];

  // Sync threads
  __syncthreads();

  // Block level reduction
  for(int offset = 1; offset < BLOCK_SIZE; offset <<= 1)
  {
    
    // Even thread
    if((threadIdx.x % (offset << 1)) == 0)
    {
        blockReduce[threadIdx.x] += blockReduce[threadIdx.x + offset];
    } 
    __syncthreads();

  }

  // Compute the reduce from shared memory.
  if (threadIdx.x == 0) {
    // And store back to global memory.
    d_odata[blockIdx.x] = blockReduce[0];
  }
}

int main()
{
    // int n = (1 << 22);  // number of elements to reduce

    unsigned bytes = n * sizeof(int);
    int *h_idata = (int *) malloc(bytes);
    for (int i=0; i<n; i++)
        h_idata[i] = (int)(rand() & 0xFF);

    // TODO determine numBlocks and numThreads
    int numBlocks = divup(n, BLOCK_SIZE);
    int numThreads = BLOCK_SIZE;

    // allocate device memory and data
    int *d_idata = NULL, *d_odata = NULL;
    CUDA(cudaMalloc((void **) &d_idata, bytes));
    CUDA(cudaMalloc((void **) &d_odata, numBlocks*sizeof(int)));  // FIX

    // copy data to device memory
    CUDA(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));

    int gpu_result = 0;
    cudaEvent_t start, stop;
    CUDA(cudaEventCreate(&start));
    CUDA(cudaEventCreate(&stop));

    // Start record
    CUDA(cudaEventRecord(start, 0));

    // TODO call your reduce kernel(s) with the right parameters
    // INPUT:       d_idata
    // OUTPUT:      d_odata
    // ELEMENTS:    n

    // (1) reduce across all elements
    //reduceGPU<<<numBlocks, numThreads>>>(d_odata, d_idata, n);
    reduceGPUShuffle<<<numBlocks, numThreads>>>(d_odata, d_idata, n);

    // (2) reduce across all blocks
    size_t block_bytes = numBlocks * sizeof(int);
    int *h_blocks = (int *)malloc(block_bytes);
    CUDA(cudaMemcpy(h_blocks, d_odata, block_bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < numBlocks; ++i)
        gpu_result += h_blocks[i];

    CUDA(cudaEventRecord(stop, 0));
    CUDA(cudaEventSynchronize(stop));
    float time_ms;
    CUDA(cudaEventElapsedTime(&time_ms, start, stop)); // that's the time your kernel took to run in ms!
    printf("bandwidth %.2f GB/s   elements %u   blocks %u   threads %u time_in_kernel %.4f ms\n",
           1e-9 * bytes/(time_ms/1e3), n, numBlocks, numThreads, time_ms);

    // check result against CPU
    int cpu_result = reduceCPU(h_idata, n);
    printf("gpu %u   cpu %u   ", gpu_result, cpu_result);
    printf((gpu_result==cpu_result) ? "pass\n" : "FAIL\n");

    printf("Time to run kernel %.4f ms\n", time_ms);

    // cleanup
    CUDA(cudaEventDestroy(start));
    CUDA(cudaEventDestroy(stop));
    free(h_idata);
    CUDA(cudaFree(d_idata));
    CUDA(cudaFree(d_odata));

    return 0;
}
