#include <cuda.h>
#include "stdio.h"
#include <arrayfire.h>
using namespace std;

#define BLOCK_ROWS 16
#define BLOCK_COLUMNS 16

#define SMEM(x, y) window[(x)+1][(y)+1]
#define IN(x,y)    image_in[(y)*columns + (x)]

__global__ void filter_shared(float* image_in, float* image_out, int columns, int rows)
{
    int tx = threadIdx.x, ty = threadIdx.y;

    // guards: is at boundary?
    bool is_x_top = (tx == 0), is_x_bot = (tx == blockDim.x-1);
    bool is_y_top = (ty == 0), is_y_bot = (ty == blockDim.y-1);

    __shared__ float window[BLOCK_COLUMNS+2][BLOCK_ROWS+2];
    // clear out shared memory (zero padding)
    if (is_x_top)           SMEM(tx-1, ty  ) = 0;
    else if (is_x_bot)      SMEM(tx+1, ty  ) = 0;
    if (is_y_top) {         SMEM(tx  , ty-1) = 0;
        if (is_x_top)       SMEM(tx-1, ty-1) = 0;
        else if (is_x_bot)  SMEM(tx+1, ty-1) = 0;
    } else if (is_y_bot) {  SMEM(tx  , ty+1) = 0;
        if (is_x_top)       SMEM(tx-1, ty+1) = 0;
        else if (is_x_bot)  SMEM(tx+1, ty+1) = 0;
    }

    // guards: is at boundary and still more image?
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    is_x_top &= (x > 0); is_x_bot &= (x < columns - 1);
    is_y_top &= (y > 0); is_y_bot &= (y < rows - 1);

    // each thread pulls from image
                            SMEM(tx  , ty  ) = IN(x  , y  ); // self
    if (is_x_top)           SMEM(tx-1, ty  ) = IN(x-1, y  );
    else if (is_x_bot)      SMEM(tx+1, ty  ) = IN(x+1, y  );
    if (is_y_top) {         SMEM(tx  , ty-1) = IN(x  , y-1);
        if (is_x_top)       SMEM(tx-1, ty-1) = IN(x-1, y-1);
        else if (is_x_bot)  SMEM(tx+1, ty-1) = IN(x+1, y-1);
    } else if (is_y_bot) {  SMEM(tx  , ty+1) = IN(x  , y+1);
        if (is_x_top)       SMEM(tx-1, ty+1) = IN(x-1, y+1);
        else if (is_x_bot)  SMEM(tx+1, ty+1) = IN(x+1, y+1);
    }
    __syncthreads();

    // pull my window from shared memory
    float v[9] = { SMEM(tx-1, ty-1), SMEM(tx  , ty-1), SMEM(tx+1, ty-1),
                   SMEM(tx-1, ty  ), SMEM(tx  , ty  ), SMEM(tx+1, ty  ),
                   SMEM(tx-1, ty+1), SMEM(tx  , ty+1), SMEM(tx+1, ty+1) };

    // bubble-sort
    for (int i = 0; i < 9; i++) {
        for (int j = i + 1; j < 9; j++) {
            if (v[i] > v[j]) { // swap?
                float tmp = v[i];
                v[i] = v[j];
                v[j] = tmp;
            }
        }
    }
    // pick the middle one
    image_out[y*columns + x] = v[4];
}

int main(int argc, char** argv)
{
	using namespace af;
	info();
	array earth = loadimage("earth.jpg", false);
	array boat = loadimage("boat.tiff", false);

	float* d_earth_orig = earth.device<float>();
	float* h_boat = boat.host<float>();
	float* d_boat_orig;
	float* d_earth_filtered;
	float* d_boat_filtered;

	size_t earth_size = earth.dims(0) * earth.dims(1) * sizeof(float);
	size_t boat_size = boat.dims(0) * boat.dims(1) * sizeof(float);

	//device input pointers
	cudaMalloc(&d_boat_orig, boat_size);

	//device output pointers
	cudaMalloc(&d_earth_filtered, earth_size);
	cudaMalloc(&d_boat_filtered, boat_size);


	cudaStream_t earthStream, boatStream;
	cudaStreamCreate(&earthStream);
	cudaStreamCreate(&boatStream);

	int columnThreads = BLOCK_COLUMNS;
	int rowThreads = BLOCK_ROWS;

	dim3 earth_blocks(earth.dims(0)/columnThreads, earth.dims(1)/rowThreads);
	dim3 boat_blocks(boat.dims(0)/columnThreads, boat.dims(1)/rowThreads);
	dim3 threads(columnThreads,rowThreads);

	int iterations = atoi(argv[1]);
	float timeSum = 0;

	filter_shared<<<earth_blocks, threads, 0, earthStream>>>(d_earth_orig, d_earth_filtered, earth.dims(1), earth.dims(0));

	cudaError_t error = cudaGetLastError();
	cout << "Error: " << cudaGetErrorString(error);

	cudaMemcpyAsync(d_boat_orig, h_boat, boat_size, cudaMemcpyHostToDevice, boatStream);
	cudaMemcpyAsync(d_boat_orig, h_boat, boat_size, cudaMemcpyHostToDevice, boatStream);
	cudaMemcpyAsync(d_boat_orig, h_boat, boat_size, cudaMemcpyHostToDevice, boatStream);
	cudaMemcpyAsync(d_boat_orig, h_boat, boat_size, cudaMemcpyHostToDevice, boatStream);
	filter_shared<<<boat_blocks, threads, 0, boatStream>>>(d_boat_orig, d_boat_filtered, boat.dims(1), boat.dims(0));
	filter_shared<<<boat_blocks, threads, 0, earthStream>>>(d_boat_orig, d_boat_filtered, boat.dims(1), boat.dims(0));
	filter_shared<<<boat_blocks, threads, 0, boatStream>>>(d_boat_orig, d_boat_filtered, boat.dims(1), boat.dims(0));
	filter_shared<<<boat_blocks, threads, 0, boatStream>>>(d_boat_orig, d_boat_filtered, boat.dims(1), boat.dims(0));


	if(atoi(argv[2]) == 1)
	{
		fig("title", argv[0]);
		fig("sub", 2, 2, 1); image(boat);
		fig("sub", 2, 2, 2); image(d_boat_filtered, boat.dims(1), boat.dims(0));
		fig("sub", 2, 2, 3); image(earth);
		fig("sub", 2, 2, 4); image(d_earth_filtered, earth.dims(1), earth.dims(0));

		getchar();
	}
//	for(int i = 0; i < iterations; ++i)
//	{
//		float time;
//		cudaEvent_t start, stop;
//		cudaEventCreate(&start);
//		cudaEventCreate(&stop);
//
//		cudaEventRecord( start, 0);
//		filter_shared<<<blocks, threads>>>(d_orig, d_filtered, imageLogo.dims(1), imageLogo.dims(0));
//		cudaEventRecord( stop, 0);
//
//		cudaEventSynchronize( stop );
//		cudaEventElapsedTime( &time, start, stop);
//
//		timeSum += time;
//	}
//	timeSum = timeSum / iterations;
//	cout << "Elapsed Time for " << BLOCK_COLUMNS << " by " << BLOCK_ROWS << " blocks: " << timeSum << endl;
//
//	if(atoi(argv[2]) == 1)
//	{
//		fig("title", argv[0]);
//		fig("sub", 2, 1, 1); image(imageLogo);
//		fig("sub", 2, 1, 2); image(d_filtered, imageLogo.dims(1), imageLogo.dims(0));
//
//		getchar();
//	}

}
