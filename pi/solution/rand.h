__global__
static void rand(int n, int *dx, int nbins)
{
    const int id = blockDim.x * blockIdx.x + threadIdx.x;
    curandState_t state;
    curand_init(1234, id, 0, &state);

    while (id < n) {
        dx[id] = nbins * curand_uniform();
        id += gridDim.x * blockDim.x;
    }
}
