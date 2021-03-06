/*

Solution of the Laplace equation for heat conduction in a square plate

*/

#include <iostream>

// global variables

const int NX = 1024;      // mesh size (number of node points along X)
const int NY = 1024;      // mesh size (number of node points along Y)

const int MAX_ITER=1000;  // number of Jacobi iterations

// device function to update the array T_new based on the values in array T_old
// note that all locations are updated simultaneously on the GPU
__global__ void Laplace(double *T_old, double *T_new)
{
    // compute the "i" and "j" location of the node point
    // handled by this thread

    int tx = threadIdx.x, ty = threadIdx.y;
    int i = blockIdx.x * blockDim.x + tx ;
    int j = blockIdx.y * blockDim.y + ty ;

    // get the natural index values of node (i,j) and its neighboring nodes
                                //                         N
    int P = i + j*NX;           // node (i,j)              |

#define smem(x,y)  s_old[(x)+1][(y)+1]

    __shared__ double s_old[16+2][16+2];
    smem(tx,ty) = T_old[P];

    bool is_x_top = tx==0, is_x_bot = tx==15;
    bool is_y_top = ty==0, is_y_bot = ty==15;

    if (is_x_top)       smem(tx-1,ty) = T_old[(i-1)*NX+j];
    else if (is_x_bot)  smem(tx+1,ty) = T_old[(i+1)*NX+j];
    if (is_y_top)       smem(tx,ty-1) = T_old[i*NX+j-1];
    else if (is_y_bot)  smem(tx,ty+1) = T_old[i*NX+j+1];

    __syncthreads();

    // update "interior" node points
    if(i>0 && i<NX-1 && j>0 && j<NY-1) {
        T_new[P] = 0.25*( smem(tx,ty+1) + smem(tx,ty-1) +
                          smem(tx-1,ty) + smem(tx+1,ty) );
    }

#undef smem
}

// initialization

void Initialize(double *TEMPERATURE)
{
    for(int i=0;i<NX;i++) {
        for(int j=0;j<NY;j++) {
            int index = i + j*NX;
            TEMPERATURE[index]=0.0;
        }
    }

    // set left wall to 1

    for(int j=0;j<NY;j++) {
        int index = j*NX;
        TEMPERATURE[index]=1.0;
    }
}

int main(int argc,char **argv)
{
    double *_T1, *_T2;  // pointers to device (GPU) memory

    // allocate a "pre-computation" T array on the host
    double *T = new double [NX*NY];

    // initialize array on the host
    Initialize(T);

    // allocate storage space on the GPU
    cudaMalloc((void **)&_T1,NX*NY*sizeof(double));
    cudaMalloc((void **)&_T2,NX*NY*sizeof(double));

    // copy (initialized) host arrays to the GPU memory from CPU memory
    cudaMemcpy(_T1,T,NX*NY*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(_T2,T,NX*NY*sizeof(double),cudaMemcpyHostToDevice);

    // assign a 2D distribution of CUDA "threads" within each CUDA "block"
    int ThreadsPerBlock=16;
    dim3 dimBlock( ThreadsPerBlock, ThreadsPerBlock );

    // calculate number of blocks along X and Y in a 2D CUDA "grid"
    dim3 dimGrid( ceil(double(NX)/double(dimBlock.x)), ceil(double(NY)/double(dimBlock.y)), 1 );

    // begin Jacobi iteration
    int k = 0;
    while(k<MAX_ITER) {
        Laplace<<<dimGrid, dimBlock>>>(_T1,_T2);   // update T1 using data stored in T2
        Laplace<<<dimGrid, dimBlock>>>(_T2,_T1);   // update T2 using data stored in T1
        k+=2;
    }

    // copy final array to the CPU from the GPU
    cudaMemcpy(T,_T2,NX*NY*sizeof(double),cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
/*
    // print the results to screen
    for (int j=NY-1;j>=0;j--) {
        for (int i=0;i<NX;i++) {
            int index = i + j*NX;
            std::cout << T[index] << " ";
        }
        std::cout << std::endl;
    }
*/
    // release memory on the host
    delete T;

    // release memory on the device
    cudaFree(_T1);
    cudaFree(_T2);

    return 0;
}
