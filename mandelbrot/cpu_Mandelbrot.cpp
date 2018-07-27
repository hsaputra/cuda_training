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

double magnitude()
{
    return pow((re*re + im*im),0.5);
}

};

// ----------------------------------------------------------------------------------
// function to check all points inside the specified window for membership in the set

void Mandelbrot(double xmin, double xmax, double ymin, double ymax, int *color)
{
    double dx = (xmax - xmin)/NX; // grid spacing along X
    double dy = (ymax - ymin)/NY; // grid spacing along Y

    for(int i=0;i<NX;i++) {
        for(int j=0;j<NY;j++) {
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

            color[i+j*NX] = iter;

            // GRAPHICS: the value of iter can be used to pick a suitable "color" when drawing the set
        }
    }
}

// ------------
// main program

int main(int argc, char* argv[])
{   
    // allocate memory for a 2D array
    int *color = new int[NX*NY];

    // fill the 2D array with zeroes
    for(int i=0;i<NX;i++) {
        for(int j=0;j<NY;j++) {
            color[i+j*NX] = 0;
        }
    }

    // initialize parameters on the host
    double xmin = -2, xmax = 1, ymin = -1.5, ymax = 1.5;

    // check each pixel for membership and fill the 2D array accordingly
    // this kernel is executed in parallel using NX*NY blocks on the GPU

    for(int k=0;k<10;k++) {
        Mandelbrot(xmin,xmax,ymin,ymax,color);
    }
    return 0;
}
