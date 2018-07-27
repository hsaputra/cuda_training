#include <cuda.h>
#include "stdio.h"
#include <arrayfire.h>
using namespace std;

#define BLOCK_ROWS 16
#define BLOCK_COLUMNS 16

#define SMEM(x, y) window[(x)+1][(y)+1]

#define IN(x,y)    image_in[(y)*columns + (x)]

__global__ void filter_simple(float* image_in, float* image_out, int columns, int rows)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	int index = 0;
	float v[9] = {0,0,0,0,0,0,0,0,0};
	for (int i = x -1; i <= x + 1; ++i)
	{
		for(int j = y -1; j <= y + 1; ++j)
		{
			if(0 <= i && i < columns && 0 <= j && j < rows)
				v[index++] = image_in[j * columns + i];
		}
	}

	for(int i = 0; i < 9; ++i)
	{
		for(int j = i + 1; j < 9; j++)
		{
			if(v[i] > v[j]){
				float tmp = v[i];
				v[i] = v[j];
				v[j] = tmp;
			}
		}
	}

	image_out[y * columns + x] = v[4];
}

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

	float* d_orig = earth.device<float>();
	float* d_filtered;

	size_t size = earth.dims(0) * earth.dims(1) * sizeof(float);

	cudaMalloc(&d_filtered, size);

	int columnThreads = BLOCK_COLUMNS;
	int rowThreads = BLOCK_ROWS;

	dim3 blocks(earth.dims(0)/columnThreads, earth.dims(1)/rowThreads);
	dim3 threads(columnThreads,rowThreads);

	int iterations = atoi(argv[1]);
	float timeSum = 0;
	for(int i = 0; i < iterations; ++i)
	{
		float time;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord( start, 0);
		filter_simple<<<blocks, threads>>>(d_orig, d_filtered, earth.dims(1), earth.dims(0));
		cudaEventRecord( stop, 0);

		cudaEventSynchronize( stop );
		cudaEventElapsedTime( &time, start, stop);

		timeSum += time;
	}
	timeSum = timeSum / iterations;
	cout << "Elapsed Time for the simple kernel " << BLOCK_COLUMNS << " by " << BLOCK_ROWS << " blocks: " << timeSum << endl;

	timeSum = 0;
	for(int i = 0; i < iterations; ++i)
	{
		float time;
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord( start, 0);
		filter_shared<<<blocks, threads>>>(d_orig, d_filtered, earth.dims(1), earth.dims(0));
		cudaEventRecord( stop, 0);

		cudaEventSynchronize( stop );
		cudaEventElapsedTime( &time, start, stop);

		timeSum += time;
	}
	timeSum = timeSum / iterations;
	cout << "Elapsed Time using shared memory: " << BLOCK_COLUMNS << " by " << BLOCK_ROWS << " blocks: " << timeSum << endl;

	if(atoi(argv[2]) == 1)
	{
		fig("title", argv[0]);
		fig("sub", 2, 1, 1); image(earth);
		fig("sub", 2, 1, 2); image(d_filtered, earth.dims(1), earth.dims(0));

		getchar();
	}

}
