// get device (GPU) information and specifications

#include <iostream>

int main(void)
{
    cudaDeviceProp prop;

    int count;

    cudaGetDeviceCount( &count );

    for(int i=0; i<count; i++)
    {
        cudaGetDeviceProperties( &prop, i);

        // print some useful info about the device here

        std::cout << "Name = " << prop.name << std::endl;

        std::cout << "Compute capability : " << prop.major << "  " << prop.minor << std::endl;

        std::cout << "Clock rate            = " << prop.clockRate << std::endl;
        std::cout << "Total global memory   = " << prop.totalGlobalMem << std::endl;
        std::cout << "Total shared memory per block  = " << prop.sharedMemPerBlock << std::endl;
        std::cout << "Total constant memory = " << prop.totalConstMem << std::endl;
        std::cout << "Max memory pitch      = " << prop.memPitch << std::endl;
        std::cout << "Max threads per block = " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "Warp size             = " << prop.warpSize << std::endl;
        std::cout << "Max threads along X   = " << prop.maxThreadsDim[0] << std::endl;
        std::cout << "                  Y   = " << prop.maxThreadsDim[1] << std::endl;
        std::cout << "                  Z   = " << prop.maxThreadsDim[2] << std::endl;
        std::cout << "Max grid size aong X   = " << prop.maxGridSize[0] << std::endl;
        std::cout << "                   Y   = " << prop.maxGridSize[1] << std::endl;
        std::cout << "                   Z   = " << prop.maxGridSize[2] << std::endl;
        std::cout << "Multi Processor count = " << prop.multiProcessorCount << std::endl;

        std::cout << std::endl;
    }

    return 0;
}
