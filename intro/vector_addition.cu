/*************************************************
** Accelereyes Training Day 1					**
** Vector Addition								**
**						 						**
** This program will add two vectors and store  **
** the result in a third vector using the GPU	**
*************************************************/

#include <iostream>
#include <vector>
#include "cuda.h"
#include "../common.h"

__global__ void add(int* a, int* b, int* c) {
    // calculate global id
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // perform calculation
    c[index] = a[index] + b[index];
}

int main(void) {
    using namespace std;
    long N = 1000 * 10;
    size_t size = N * sizeof(int);

    // initialize device pointers and allocate memory on the GPU
    int *a_d, *b_d, *c_d;

    cudaMalloc(&a_d, size);
    cudaMalloc(&b_d, size);
    cudaMalloc(&c_d, size);

    // initalize data on hosta
    int *a_h = new int[N];
    int *b_h = new int[N];

    for (int i = 0; i < N; i++) {
      a_h[i] = 1;
      b_h[i] = 2;
    }

    // move host data to the GPU
    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);

    // launch kernel
    add<<<10, 1000>>>(a_d, b_d, c_d);
    CUDA(cudaPeekAtLastError());
    CUDA(cudaDeviceSynchronize());

    // get the results from the GPU
    int *c_h = new int[N];
    cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);

    // print results
    for(int i = 0; i < N; ++i) {
      cout << c_h[i] << ", ";
    }

    cout << "\n";

    free(a_h);
    free(b_h);
    free(c_h);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    
    return 0;
}
