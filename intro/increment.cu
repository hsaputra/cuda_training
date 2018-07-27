// simple increment kernel

#include <cuda.h>
#include <stdio.h>

//TODO: increment kernel
__global__
void increment(float *val) {
  *val += 2.0f; 
}

int main(void)
{
    // create host array and initialize
    float *device_pointer;
 
    // print original value
    float input = 40.0f;    
    printf("Input: %f\n", input); 

    // allocate device memory
    cudaMalloc(&device_pointer, sizeof(float));

    // memcpy to device
    cudaMemcpy(device_pointer, &input, sizeof(float), cudaMemcpyHostToDevice);

    // launch the increment kernel
    increment<<<1, 1>>>(device_pointer);

    // memcpy results back to host
    cudaMemcpy(&input, device_pointer, sizeof(float), cudaMemcpyDeviceToHost);

    // print new value
    printf("New Input: %f\n", input);

    return 0;
}
