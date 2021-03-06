#include <iostream>
#include <vector>

using std::vector;

void printMat(float *mat, int rows, int cols, unsigned noutput=0, unsigned offset=0);

// parameter describing the size of matrix A
const int rows = 8;
const int cols = 8;

const int BLOCK_SIZE = 4;

// Transpose kernel Coalese with shared memory.
__global__
void transposeCoales(float* dest, float* source) {

  __shared__ float blockTemp[BLOCK_SIZE][BLOCK_SIZE];

  int yIndex = blockIdx.x * blockDim.x + threadIdx.x;
  int xIndex = blockIdx.y * blockDim.y + threadIdx.y;

  // Position in the source.
  int index_in = xIndex * cols + yIndex;

  // switch i to j and j to i in the shared block.
  blockTemp[threadIdx.y][threadIdx.x] = source[index_in];

  __syncthreads();

  // Lets do coales write.

  // Need to target new block along x so need to move y
  int i = blockIdx.y * blockDim.x + threadIdx.x;
  int j = blockIdx.x * blockDim.y + threadIdx.y;
  int index_out = j * rows + i;
  dest[index_out] = blockTemp[threadIdx.x][threadIdx.y];
}


// Transpose kernel
__global__
void transpose(float* dest, float* source) {
  int yIndex = blockIdx.x * blockDim.x + threadIdx.x;
  int xIndex = blockIdx.y * blockDim.y + threadIdx.y;

  int index_in = xIndex * cols + yIndex; 
  int index_out = yIndex * rows + xIndex; 

  dest[index_out] = source[index_in];    
}


// the main program starts life on the CPU and calls device kernels as required
int main(int argc, char *argv[])
{
    // allocate space in the host for storing input arrays (a and b) and the output array (c)
    vector<float> a(rows*cols);
    vector<float> b(rows*cols);

    // define device pointers for the same arrays when they'll be copied to the device
    float *_a, *_b;

    // allocate memory on the device (GPU) and check for errors (if any) during this call
    cudaError_t err;

    // allocate space for matrix A 
    err = cudaMalloc((void **) &_a, rows*cols*sizeof(float));
    if (err!= cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    // allocate space for matrix B
    err = cudaMalloc((void **) &_b, rows*cols*sizeof(float));
    if (err!= cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    // Fill matrix A
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            a[row * cols + col] = row * cols + col;
        }
    }

    // Copy array contents of A from the host (CPU) to the device (GPU)
    // Note that this is copied to the "global" memory on the device and is accessible to all threads in all blocks
    cudaMemcpy(_a, a.data(), rows*cols*sizeof(float), cudaMemcpyHostToDevice);

    // assign a 2D distribution of 16 x 16 x 1 CUDA "threads" within each CUDA "block"
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);

    // calculate number of blocks along X and Y in a 2D CUDA "grid"
    dim3 dimGrid( ceil(float(rows)/float(dimBlock.x)), ceil(float(cols)/float(dimBlock.y)), 1 );

    float time;

    // create CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start the timer
    cudaEventRecord( start, 0);


    // Launch the GPU kernel
    //transpose<<<dimGrid, dimBlock>>>(_b, _a);
    transposeCoales<<<dimGrid, dimBlock>>>(_b, _a);

    // stop the timer
    cudaEventRecord( stop, 0);
    cudaEventSynchronize( stop );
    cudaEventElapsedTime( &time, start, stop);

    // print out the time required for the kernel to finish the transpose operation
    double Bandwidth = 2.0*1000*(double)(rows*cols*sizeof(float)) / (1000*1000*1000*time);
    std::cout << "Elapsed Time  = " << time << " Bandwidth used (GB/s) = " << Bandwidth << std::endl;

    // copy the answer back to the host (CPU) from the device (GPU)
    cudaMemcpy(b.data(), _b, cols*rows*sizeof(float), cudaMemcpyDeviceToHost);

    // PRINT
    printMat(a.data(), rows, cols);
    printMat(b.data(), rows, cols);


    // TODO LETS NOW TRY TO DO MEMCOPY DEVICE to DEVICE.
    cudaMemset(_b, 0, cols*rows*sizeof(float));    


    // free device memory
    cudaFree(_a);
    cudaFree(_b);

    // successful program termination
    return 0;
}

//assumes square matrix
void printMat(float *mat, int rows, int cols, unsigned noutput, unsigned offset) {
    if(noutput < 1) {
        noutput = rows - offset;
    }
    int end = min(offset + noutput, rows);

    if(end == rows)
        printf("outputting %dx%d matrix\n", rows, cols);
    else if(offset == 0)
        printf("outputting %dx%d subsection of %dx%d matrix\n", noutput, noutput, rows, cols);
    else
        printf("outputting %dx%d subsection of %dx%d matrix with offset %d\n", noutput, noutput, rows, cols, offset);

    for(int r = offset; r < end; ++r) {
        for(int c = offset; c < end; ++c) {
            printf("%f ", mat[r*cols + c]);
        }
        printf("\n");
    }
}

