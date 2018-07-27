/*
 * CUDA C++ code to multiply two square matrices
 *
 * Written by: Abhijit Joshi <abhijit@accelereyes.com>
 * Last modified: Wed Apr 10 2013 @2:44 am
 *
 * To compile and link this example, use
 *
 *     nvcc matMul.cu -o matMul.x
 *
 * To run this code, use
 *
 *     ./matMul.x
 */

#include <iostream>
#include <assert.h>


/*
  Helper functions for timing CPU code.  Does not handle async GPU
  computation, use cudaDeviceSynchronize().

*/

#include <sys/time.h>
typedef struct { timeval val; } timer;

// get current time
static inline timer timenow(void)
{
    timer time;
    gettimeofday(&time.val, NULL);
    return time;
}

// difference in time between two timers (in seconds)
static double timediff(timer start, timer end)
{
    struct timeval elapsed;
    timersub(&start.val, &end.val, &elapsed);
    long sec = elapsed.tv_sec;
    long usec = elapsed.tv_usec;
    return fabs(sec + usec * 1e-6);
}




// parameter describing the size of the matrices
const int rows = 1024;
const int cols = 1024;

// block size for tiled multiplication using shared memory
const int BLOCK_SIZE = 16;

// total number of blocks along X and Y
const int NUM_BLOCKS = rows/BLOCK_SIZE;

// print the matrix
void displayMatrix(float *a)
{
    std::cout << std::endl;
    for(int i=0; i<rows; i++) {
        for(int j=0; j<cols; j++) {
            std::cout << a[i*cols+j] << " ";
        }
        std::cout << std::endl;
    }
}

void matrixMultiplyCPU(float *h_a,   // pointer to matrix A on the cpu
                       float *h_b,   // pointer to matrix B on the cpu
                       float *h_c)   // pointer to matrix C = AB on the cpu
{
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            float sum = 0;
            for (int k = 0; k < cols; ++k)
                sum += h_a[row*rows+k] * h_b[k*rows+col];
            h_c[row*cols+col] = sum;
        }
    }
}


// using global memory
__global__ void matrixMultiplyNaive(float *_a,   // pointer to matrix A on the device
                                    float *_b,   // pointer to matrix B on the device
                                    float *_c)   // pointer to matrix C = AB on the device
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // make sure we stay within bounds while calculating C(i,j)
    if ((row < rows) && (col < cols)) {
        // compute the inner product using data from shared memory
        float sum = 0;
        for(int k=0; k<cols; k++) {
            sum += _a[row*rows+k]*_b[k*rows+col];
        }
        _c[row*cols+col] = sum;
    }
}

__global__ void matrixMultiplyTiled(float *_a,   // pointer to matrix A on the device
                                    float *_b,   // pointer to matrix B on the device
                                    float *_c)   // pointer to matrix C = AB on the device
{
    // compute the "row" and "col" inside C = AB handled by this thread

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // define two 2D arrays in shared memory, one a subset of A and another a subset of B
    __shared__ float Asub[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bsub[BLOCK_SIZE][BLOCK_SIZE];

    // the complete inner product will be stored here but assembled piece-by-piece
    float global_sum = 0.0;

    // loop over all the tiles in A and B
    for(int block=0;block<NUM_BLOCKS;block++) {
        // copy data from global memory to shared memory
        // shared memory is shared between all threads in a block
        Asub[threadIdx.y][threadIdx.x] = _a[row*cols + BLOCK_SIZE*block+threadIdx.x];   // copy one element from A into shared memory
        Bsub[threadIdx.y][threadIdx.x] = _b[(BLOCK_SIZE*block+threadIdx.y)*cols+col];   // copy one element from B into shared memory

        // synchronize threads
        __syncthreads();

        // compute part of the inner product using data from shared memory
        for(int k=0; k<BLOCK_SIZE; k++) {
            global_sum += Asub[threadIdx.y][k]*Bsub[k][threadIdx.x];
        }
    }

    _c[row*cols+col] = global_sum;
}

// the main program starts life on the CPU and calls device kernels as required
int main(int argc, char *argv[])
{
    // allocate space in the host for storing input arrays (a and b) and the output array (c)
    float *a = new float[rows*cols];
    float *b = new float[rows*cols];
    float *c = new float[rows*cols];

    // define device pointers for the same arrays when they'll be copied to the device
    float *_a, *_b, *_c;

    // allocate memory on the device (GPU) and check for errors (if any) during this call
    cudaError_t err;

    // allocate space for matrix A
    err = cudaMalloc((void **) &_a, rows*cols*sizeof(float));
    if (err!= cudaSuccess) {
        std::cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << std::endl;
        exit(EXIT_FAILURE);
    }

    // allocate space for matrix B
    err = cudaMalloc((void **) &_b, rows*cols*sizeof(float));
    if (err!= cudaSuccess) {
        std::cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << std::endl;
        exit(EXIT_FAILURE);
    }

    // allocate space for matrix C = AB
    err = cudaMalloc((void **) &_c, rows*cols*sizeof(float));
    if (err!= cudaSuccess) {
        std::cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << std::endl;
        exit(EXIT_FAILURE);
    }

    // Fill matrix A
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            a[row + col*rows] = 2.0;
        }
    }
    if((rows<33) && (cols<33)) displayMatrix(a);

    // Fill matrix B
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            b[row + col*rows] = 4.0;
        }
    }
    if((rows<33) && (cols<33)) displayMatrix(b);

    // perform multiply on cpu (host)
    {
        float *c = new float[rows*cols];
        timer start = timenow();
        matrixMultiplyCPU(a,b,c);
        timer stop  = timenow();
        for (int i = 0; i < rows*cols; ++i)
            assert(c[i] == 1024*2*4);  // ensure results match (assume a=2 b=4 1024x1024)
        delete [] c;

        double time_s = timediff(start,stop); // seconds
        double GFLOPs = (double)(rows*cols) * 2*rows / 1e9 / time_s;
        std::cout << "cpu elapsed time  = " << time_s*1e3 << " ms,  GFLOPs = " << GFLOPs << std::endl;
    }


    // Copy array contents of A and B from the host (CPU) to the device (GPU)
    // Note that this is copied to the "global" memory on the device and is accessible to all threads in all blocks
    // WARNING: Global memory is slow (latency of a few 100 cycles)
    //
    cudaMemcpy(_a, a, rows*cols*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(_b, b, rows*cols*sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

    // calculate number of blocks along X and Y in a 2D CUDA "grid"
    dim3 dimGrid( ceil(float(cols)/float(dimBlock.x)), ceil(float(rows)/float(dimBlock.y)), 1 );

    // create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start the timer
    cudaEventRecord( start, 0);

    // launch the GPU kernel for parallel matrix multiplication of A and B
    // matrixMultiplyNaive<<<dimGrid,dimBlock>>>(_a, _b, _c);

    matrixMultiplyTiled<<<dimGrid,dimBlock>>>(_a, _b, _c);

    // stop the timer
    cudaEventRecord( stop, 0);
    cudaEventSynchronize( stop );
    float time_ms = 0;
    cudaEventElapsedTime( &time_ms, start, stop); // milliseconds

    // print out the number of GFLOPs
    double GFLOPs = (double)(rows*cols) * 2*rows / 1e9 / (time_ms / 1e3);
    std::cout << "gpu elapsed time  = " << time_ms << " ms,  GFLOPs = " << GFLOPs << std::endl;

    // copy the answer back to the host (CPU) from the device (GPU)
    cudaMemcpy(c, _c, rows*cols*sizeof(float), cudaMemcpyDeviceToHost);

    if((rows<33) && (cols<33)) displayMatrix(c);

    // free device memory
    cudaFree(_a);
    cudaFree(_b);
    cudaFree(_c);

    // free host memory
    delete a;
    delete b;
    delete c;

    // successful program termination
    return 0;
}
