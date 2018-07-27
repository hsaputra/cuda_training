/*
  Accelereyes

  Monte Carlo Pi Estimation

  Estimate pi by calculating the ratio of points that fell inside of a
  unit circle with the points that did not.
*/

#include <iostream>
#include <vector>
#include "../../common.h"

using namespace std;

// Create a kernel to estimate pi
__global__
void pi(float* randx, float* randy, int *block_sums, int nsamples) {
    extern __shared__ int sums[];
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    int count = 0;
    int stride = gridDim.x * blockDim.x;
    for(int index = id; index < nsamples; index += stride) {
        float x = randx[index];
        float y = randy[index];
        count += (x*x + y*y) < 1.0f;
    }
    sums[threadIdx.x] = count;
    __syncthreads();

    if(threadIdx.x == 0) {
        int sum = 0;
        for(int i = 0; i < blockDim.x; i++) {
            sum += count;
        }
        block_sums[blockIdx.x] = sum;
    }
}

// Create a kernel to estimate pi
__global__
void pi_reduce(float* randx, float* randy, int *block_sums, int nsamples) {
    extern __shared__ int sums[];
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    int count = 0;
    int stride = gridDim.x * blockDim.x;
    for(int index = id; index < nsamples; index += stride) {
        float x = randx[index];
        float y = randy[index];
        count += (x*x + y*y) < 1.0f;
    }
    sums[threadIdx.x] = count;

    int offset = blockDim.x >> 1;
    while (offset >= 32) {
        __syncthreads();
        if (threadIdx.x < offset) {
          sums[threadIdx.x] += sums[threadIdx.x + offset];
        }
        offset >>= 1;
    }

    if(threadIdx.x < 32)
      atomicAdd(block_sums, sums[threadIdx.x]);

}

// Create a kernel to estimate pi
__global__
void pi_reduce_atomic(float* randx, float* randy, int *block_sums, int nsamples) {
    __shared__ int sums[32];
    if(threadIdx.x < 32) sums[threadIdx.x] = 0;

    int id = blockIdx.x * blockDim.x + threadIdx.x;

    int count = 0;
    int stride = gridDim.x * blockDim.x;
    for(int index = id; index < nsamples; index += stride) {
        if(index < nsamples) {
            float x = randx[index];
            float y = randy[index];
            count += (x*x + y*y) < 1.0f;
        }
    }

    int widx = threadIdx.x % 32;
    atomicAdd(sums + widx, count);
    __syncthreads();

    if(threadIdx.x < 32) {
      atomicAdd(block_sums, sums[threadIdx.x]);
    }
}

int nsamples = 1e8;


int main(void)
{
    // allocate space to hold random values
    vector<float> h_randNumsX(nsamples);
    vector<float> h_randNumsY(nsamples);

    srand(time(NULL)); // seed with system clock

    //Initialize vector with random values
    for (int i = 0; i < h_randNumsX.size(); ++i) {
        h_randNumsX[i] = float(rand()) / RAND_MAX;
        h_randNumsY[i] = float(rand()) / RAND_MAX;
    }

    // Send random values to the GPU
    size_t size = nsamples * sizeof(float);
    float* d_randNumsX;
    float* d_randNumsY;
    cudaMalloc(&d_randNumsX, size);  // TODO check return cuda* return codes
    cudaMalloc(&d_randNumsY, size);

    cudaMemcpy(d_randNumsX, h_randNumsX.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_randNumsY, h_randNumsY.data(), size, cudaMemcpyHostToDevice);

    int samples_per_thread = 1000;
    int threads = 256;
    int blocks = nsamples /(threads * samples_per_thread);

    int* block_sums;
    cudaMalloc(&block_sums, blocks * sizeof(int));

    pi <<< blocks, threads, threads * sizeof(int)>>> (d_randNumsX, d_randNumsY, block_sums, nsamples);
    cudaMemset(block_sums, 0, sizeof(int));
    pi_reduce <<< blocks, threads, threads * sizeof(int)>>> (d_randNumsX, d_randNumsY, block_sums, nsamples);
    cudaMemset(block_sums, 0, sizeof(int));
    pi_reduce_atomic <<< blocks, threads>>> (d_randNumsX, d_randNumsY, block_sums, nsamples);

    vector<int> h_block_sums(blocks, 0);
    cudaMemcpy(h_block_sums.data(), block_sums, sizeof(int), cudaMemcpyDeviceToHost);

    int nsamples_in_circle = 0;
    for(int sum : h_block_sums) {
      nsamples_in_circle += sum;
    }

    // fraction that fell within (quarter) of unit circle
    float estimatedValue = 4.0 * float(nsamples_in_circle) / nsamples;

    cout << "Estimated Value: " << estimatedValue << endl;
}