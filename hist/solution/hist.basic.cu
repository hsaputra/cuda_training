#include <stdio.h>

#include "../common.h"


// TODO shared memory bining (advanced)
__global__
static void hist(int nbins, int *dbins, int nvals, const int *dvals)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if (x >= nvals) return;

    int v = dvals[x];
    int bin = v % nbins;
    atomicAdd(dbins + bin, 1);
}


int main()
{
    // number of (random) input values
    int nvals = 4 << 20;
    int nbins = 256;

    size_t vbytes = nvals * sizeof(int);
    size_t bbytes = nbins * sizeof(int);

    // initialize random data on cpu and place in appropriate bin
    // TODO (optional) move this random data creation into a kernel (see example day1/pi)
    int *h_vals = (int *)malloc(vbytes);
    int *h_bins_true = (int *)malloc(bbytes); // correct bins
    memset(h_bins_true, 0, bbytes);
    for (int i = 0; i < nvals; ++i) {
        int val = 200 * nbins * float(rand()) / RAND_MAX;
        int bin = val % nbins;
        h_vals[i] = val;
        h_bins_true[bin]++; // check against these later
        // printf("val %d   bin %d\n", h_vals[i], bin);
    }
    printf("elements  %u\n", nvals);

    // allocate gpu memory and transmit input values
    int *d_vals, *d_bins;
    CUDA(cudaMalloc(&d_vals, vbytes));
    CUDA(cudaMalloc(&d_bins, bbytes));
    CUDA(cudaMemcpy(d_vals, h_vals, vbytes, cudaMemcpyHostToDevice));

    // create events
    cudaEvent_t start, stop;
    CUDA(cudaEventCreate(&start));
    CUDA(cudaEventCreate(&stop));

    // compute histogram
    int threads = 256;
    int blocks = (nvals + threads - 1)/threads;
    CUDA(cudaEventRecord(start, 0));
    CUDA(cudaMemset(d_bins, 0, bbytes)); // zero bins .. consider part of timing?
    hist<<<blocks,threads>>>(nbins, d_bins, nvals, d_vals);
    CUDA(cudaGetLastError());
    CUDA(cudaEventRecord(stop, 0));

    // time kernel
    CUDA(cudaEventSynchronize(stop));
    float time_ms = 0;
    CUDA(cudaEventElapsedTime(&time_ms, start, stop));
    printf("time  %.3f ms\n", time_ms);

    // verify CPU match
    int *h_bins = (int *)malloc(bbytes);
    CUDA(cudaMemcpy(h_bins, d_bins, bbytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < nbins; ++i) {
        if (h_bins[i] != h_bins_true[i]) {
            printf("error: invalid bin:  cpu[%d]=%d  gpu[%d]=%d\n",
                   i, h_bins_true[i], i, h_bins[i]);
            break;
        }
    }

    return 0;
}
