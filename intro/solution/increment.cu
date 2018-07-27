// simple increment kernel

#include <cuda.h>
#include <stdio.h>

__global__
void inc(unsigned n, float *d_data)
{
    d_data[0]+=2;
}

int main(void)
{
    // create host array and initialize
    int n = 1; // one element
    size_t bytes = n * sizeof(float);
    float *h_data = (float *)malloc(bytes);
    h_data[0] = 40;

    // print original value
    printf("original: %g\n", h_data[0]);

    // allocate device memory
    float *d_data;
    cudaMalloc((void **)&d_data, bytes);

    // memcpy to device
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    // launch the increment kernel
    inc<<<1,1>>>(n, d_data);

    // memcpy results back to host
    cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);

    // print new value
    printf("new:      %g\n", h_data[0]);

    return 0;
}
