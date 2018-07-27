/*
  
Simulation of flow inside a 2D square cavity
using the lattice Boltzmann method (LBM)
  
Written by:       Abhijit Joshi (abhijit@accelereyes.com)
  
Last modified on: Saturday, March 23 @ 6:41 pm

Build instructions: g++ cpu_lbm.cpp -o lbmCPU.x

Run instructions: ./lbmCPU.x
  
*/

#include<iostream>

// problem parameters
const int N = 512;            // number of node points along X and Y (cavity length in lattice units)
const int TIME_STEPS = 1000;  // number of time steps for which the simulation is run

const int NDIR = 9;           // number of discrete velocity directions used in the D2Q9 model

const double DENSITY = 2.7;          // fluid density in lattice units
const double LID_VELOCITY = 0.05;    // lid velocity in lattice units
const double REYNOLDS_NUMBER = 100;  // Re =

int main(int argc, char *argv[])
{
    // the base vectors and associated weight coefficients
    double *ex = new double[NDIR];
    double *ey = new double[NDIR];
    double *alpha = new double[NDIR];

    ex[0] =  0.0;   ey[0] =  0.0;   alpha[0] = 4.0 /  9.0;
    ex[1] =  1.0;   ey[1] =  0.0;   alpha[1] = 1.0 /  9.0;
    ex[2] =  0.0;   ey[2] =  1.0;   alpha[2] = 1.0 /  9.0;
    ex[3] = -1.0;   ey[3] =  0.0;   alpha[3] = 1.0 /  9.0;
    ex[4] =  0.0;   ey[4] = -1.0;   alpha[4] = 1.0 /  9.0;
    ex[5] =  1.0;   ey[5] =  1.0;   alpha[5] = 1.0 / 36.0;
    ex[6] = -1.0;   ey[6] =  1.0;   alpha[6] = 1.0 / 36.0;
    ex[7] = -1.0;   ey[7] = -1.0;   alpha[7] = 1.0 / 36.0;
    ex[8] =  1.0;   ey[8] = -1.0;   alpha[8] = 1.0 / 36.0;

    // anti-vector direction for each base vector direction
    // (useful for implementing the bounce-back scheme)
    int *ant = new int[NDIR];

    ant[0] = 0;      //      6        2        5
    ant[1] = 3;      //               ^       
    ant[2] = 4;      //               |
    ant[3] = 1;      //               |  
    ant[4] = 2;      //      3 <----- 0 -----> 1
    ant[5] = 7;      //               |
    ant[6] = 8;      //               |
    ant[7] = 5;      //               v
    ant[8] = 6;      //      7        4        8

    // distribution functions
    double *f = new double[N*N*NDIR];
    double *feq = new double[N*N*NDIR];
    double *f_new = new double[N*N*NDIR];

    // density
    double *rh = new double[N*N];

    // velocity components 
    double *ux = new double[N*N]; 
    double *uy = new double[N*N]; 

    // initialize density and velocity fields inside the cavity
    for(int i=0;i<N;i++) {
        for(int j=0;j<N;j++) {
            rh[i+N*j] = DENSITY;
            ux[i+N*j] = 0;
            uy[i+N*j] = 0;
        }
    }
    for(int i=0;i<N;i++) ux[i+N*(N-1)] = LID_VELOCITY;

    // assign initial values for distribution functions
    for(int i=0;i<N;i++) {
        for(int j=0;j<N;j++) {
            int ixy = i+N*j;
            for(int dir=0;dir<NDIR;dir++) {
                int index = i+j*N+dir*N*N;
                double edotu = ex[dir]*ux[ixy] + ey[dir]*uy[ixy];
                double udotu = ux[ixy]*ux[ixy] + uy[ixy]*uy[ixy];
                feq[index] = rh[ixy] * alpha[dir] * (1 + 3*edotu + 4.5*edotu*edotu - 1.5*udotu); 
                f[index] = feq[index]; 
                f_new[index] = feq[index]; 
            }
        }
    }

    // calculate fluid viscosity based on the Reynolds number
    double kinematicViscosity = LID_VELOCITY * (double)N / REYNOLDS_NUMBER;

    // calculate relaxation time tau
    double tau =  0.5 + 3.0 * kinematicViscosity;

    // time integration
    int time=0;
    while(time<TIME_STEPS) {

        time++;

        // collision
        for(int i=1;i<N-1;i++) {
            for(int j=1;j<N-1;j++) {
                int ixy = i+N*j;
                for(int dir=0;dir<NDIR;dir++) {
                    int index = i+j*N+dir*N*N;
                    double edotu = ex[dir]*ux[ixy] + ey[dir]*uy[ixy];
                    double udotu = ux[ixy]*ux[ixy] + uy[ixy]*uy[ixy];
                    feq[index] = rh[ixy] * alpha[dir] * (1 + 3*edotu + 4.5*edotu*edotu - 1.5*udotu);
                }
            }
        }

        // streaming from interior node points
        for(int i=1;i<N-1;i++) {
            for(int j=1;j<N-1;j++) {
                for(int dir=0;dir<NDIR;dir++) {

                    int index = i+j*N+dir*N*N;   // (i,j,dir)
                    int index_new = (i+ex[dir]) + (j+ey[dir])*N + dir*N*N;
                    int index_ant = i + j*N + ant[dir]*N*N;

                    // post-collision distribution at (i,j) along "dir"
                    double f_plus = f[index] - (f[index] - feq[index])/tau;

                    if((i+ex[dir]==0) || (i+ex[dir]==N-1) || (j+ey[dir]==0) || (j+ey[dir]==N-1)) {
                        // bounce back
                        int ixy = i+ex[dir] + N*(j+ey[dir]);
                        double ubdote = ux[ixy]*ex[dir] + uy[ixy]*ey[dir];
                        f_new[index_ant] = f_plus - 6.0 * DENSITY * alpha[dir] * ubdote;
                    } 
                    else {
                        // stream to neighbor
                        f_new[index_new] = f_plus;
                    }
                }
            }
        }

        // push f_new into f
        for(int index=0;index<N*N*NDIR;index++) f[index] = f_new[index];

        // update density at interior nodes
        for(int i=1;i<N-1;i++) {
            for(int j=1;j<N-1;j++) {
                double rho=0;
                for(int dir=0;dir<NDIR;dir++) {
                    int index = i+j*N+dir*N*N;
                    rho+=f_new[index];
                }
                rh[i+N*j] = rho;
            }
        }

        // update velocity at interior nodes
        for(int i=1;i<N-1;i++) {
            for(int j=1;j<N-1;j++) {
                double velx=0;
                double vely=0;
                for(int dir=0;dir<NDIR;dir++) {
                    int index = i+j*N+dir*N*N;
                    velx+=f_new[index]*ex[dir]; 
                    vely+=f_new[index]*ey[dir]; 
                }
                ux[i+N*j] = velx/rh[i+N*j];
                uy[i+N*j] = vely/rh[i+N*j];
            }
        }

    } // while loop ends

/*
    // centerline velocity profile
    for(int i=1;i<N-1;i++) {
        std::cout << 0.5 + (double)(i-1) << "  " << uy[i + N*(N/2)] << std::endl;
    }
*/
    return 0;
}
