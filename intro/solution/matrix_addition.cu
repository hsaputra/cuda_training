
/*************************************************
** Accelereyes Training Day 1					**
** Matrix Addition								**
** 												**
** This program will add two matrices and store **
** the result in a third matrix using the GPU	**
*************************************************/

#include <iostream>
#include <vector>

#define THREADS 10
using namespace std;

__global__ void add(int *a, int *b, int *c,int columns,int rows)
{
	// get the global id for the thread
	int x = (blockIdx.x * blockDim.x + threadIdx.x);
	int y = (blockIdx.y * blockDim.y + threadIdx.y);

	// calculate the index of the input data
	int index = y * columns + x;

	c[index] = a[index] + b[index];
}

int main(void)
{
	int rows = 100;
	int columns = 100;
	int elements = rows * columns;

	size_t size = rows * columns * sizeof(int);

	// create device pointers
	int* d_a;
	int* d_b;
	int* d_c;

	// allocate memory on the device
	cudaMalloc(&d_a, size);
	cudaMalloc(&d_b, size);
	cudaMalloc(&d_c, size);

	// initalize host variables
	vector<int> h_a(elements, 5);
	vector<int> h_b(elements, 5);
	vector<int> h_c(elements);

	// transfer the host data to the GPU
	cudaMemcpy(d_a, &h_a.front(), size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &h_b.front(), size, cudaMemcpyHostToDevice);

	// calculate the number of threads and blocks
	dim3 threads(THREADS, THREADS);
	dim3 blocks(columns/threads.x, rows/threads.y);

	// Launch the add kernel
	add<<<blocks, threads>>>(d_a, d_b, d_c, columns, rows);

	// get the results from the GPU
	cudaMemcpy(&h_c.front(), d_c, size, cudaMemcpyDeviceToHost);

    // print top left corner
	for(int i = 0; i < 5; i++) {
		for(int j = 0; j < 10; j++)
			cout << h_c[i * rows + j] << " ";
		cout << endl;
	}
}

