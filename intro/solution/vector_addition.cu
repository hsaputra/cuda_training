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

__global__ void add(int* a, int* b, int* c) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    c[id] = a[id] + b[id];
}

int main(void) {
    using namespace std;
    long N = 1000;
    size_t size = N * sizeof(int);

    // initialize device pointers and allocate memory on the GPU
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

	// initalize data on host
	vector<int> h_a(N, 1);
	vector<int> h_b(N, 2);
	vector<int> h_c(N);

	// move host data to the GPU
	cudaMemcpy(d_a, &h_a.front(), size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &h_b.front(), size, cudaMemcpyHostToDevice);

	// launch kernel
	int blocks = 10;
	add <<< blocks, N/blocks >>> (d_a, d_b, d_c);

	// get the results from the GPU
	cudaMemcpy(&h_c.front(), d_c, size, cudaMemcpyDeviceToHost);

	for(int i = 0; i < N; ++i) {
		cout << h_c[i] << ", ";
	}

	return 0;
}
