/*

Calculating which points in the specified window of the complex plane
lie inside the Mandelbrot set

Can be easily extended using OpenGL to include graphics
and actually plot the results on screen

*/

#include <iostream>
#include <complex>
#include <cmath>
#include <cstdlib>

// -----------------------------
// declare some global variables

const int MAX_ITER = 200;    // maximum number of iterations

const int NX = 1000;  // number of points along X inside the window
const int NY = 1000;  // number of points along Y inside the window

// -------------------------------------------------------
// define a class for complex numbers and their operations

class dcmplx
{
public:
    double re;   // real component
    double im;   // imaginary component

// function to calculate the magnitude or absolute value of the complex number
// this function is called from and executes on the device (GPU) 

__device__
double magnitude()
{
    return pow((re*re + im*im),0.5);
}

};

// ----------------------------------------------------------------------------------
// function to check all points inside the specified window for membership in the set
// this function is called from the host (CPU) but executes on the device (GPU)

__global__
void Mandelbrot(double xmin, double xmax, double ymin, double ymax, int *dev_color)
{
    double dx = (xmax - xmin)/NX; // grid spacing along X
    double dy = (ymax - ymin)/NY; // grid spacing along Y

    int i = 10*blockIdx.x + threadIdx.x;
    int j = 10*blockIdx.y + threadIdx.y;

    double x = xmin + (double) i*dx;   // actual x coordinate (real component)
    double y = ymin + (double) j*dy;   // actual y coordinate (imaginary component)

    dcmplx c;
    c.re = x;
    c.im = y;

    dcmplx z;
    z.re = 0.0;
    z.im = 0.0;

    // ---------------
    // z <---- z*z + c

    int iter = 0;

    while(iter<MAX_ITER) {
        iter++;
        dcmplx temp = z;
        z.re = temp.re*temp.re - temp.im*temp.im  +  c.re;
        z.im = 2.0*temp.re*temp.im                +  c.im;
        
        if (z.magnitude() > 2.0) break;
    }

    // the 2D array "dev_color" stores how many iterations were required for divergence
    // for points outside the Mandelbrot set, this is typically a small number
    // points inside the set do not diverge and thus iter is a large number for such points

    dev_color[i+j*NX] = iter;

    // GRAPHICS: the value of iter can be used to pick a suitable "color" when drawing the set
}

// ---------------------------------------
// main program executes on the host (CPU)

int main(int argc, char* argv[])
{   
    // allocate memory for a 2D array on the host (CPU)
    int *color = new int[NX*NY];

    // fill the 2D array with zeroes
    for(int i=0;i<NX;i++) {
        for(int j=0;j<NY;j++) {
            color[i+j*NX] = 0;
        }
    }

    // allocate memory on the device (GPU)
    int *dev_color;
    cudaMalloc((void **) &dev_color, NX*NY*sizeof(int));

    // initialize parameters on the host
    double xmin = -2, xmax = 1, ymin = -1.5, ymax = 1.5;

    // check each pixel for membership and fill the 2D array accordingly
    // this kernel is executed in parallel on the GPU using a grid of 100 x 100 blocks
    // where each block contains 10 x 10 threads

    dim3 dimGrid (1000, 1000, 1);   // 1000 x 1000 blocks in a grid
    dim3 dimBlock(1   , 1   , 1);   // 1 x 1 x 1 threads in each block

    for(int k=0;k<10;k++) {
        Mandelbrot<<<dimGrid,dimBlock>>>(xmin,xmax,ymin,ymax,dev_color);
    }

    // copy the 2D array from device (GPU) to host (CPU)
    cudaMemcpy(color, dev_color, NX*NY*sizeof(int), cudaMemcpyDeviceToHost);
    
    return 0;
}
