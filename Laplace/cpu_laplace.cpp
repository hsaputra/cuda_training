#include <iostream>
#include <cmath>

// number of nodes along the X and Y axes

const int NX = 1024;
const int NY = 1024;

// function to carry out Jacobi iteration

void Jacobi(float* T, float* T_new)
{
    const int MAX_ITER = 1000;
    int iter = 0;
//  float max_change;

    while(iter < MAX_ITER)
    {
        for(int i=1; i<NX-1; i++)
        {
            for(int j=1; j<NY-1; j++)
            {
                float T_E = T[(i+1) + NX*j];
                float T_W = T[(i-1) + NX*j];
                float T_N = T[i + NX*(j+1)];
                float T_S = T[i + NX*(j-1)];
                T_new[i+NX*j] = 0.25*(T_E + T_W + T_N + T_S);
            }
        }

        for(int i=1; i<NX-1; i++)
        {
            for(int j=1; j<NY-1; j++)
            {
                float T_E = T_new[(i+1) + NX*j];
                float T_W = T_new[(i-1) + NX*j];
                float T_N = T_new[i + NX*(j+1)];
                float T_S = T_new[i + NX*(j-1)];
                T[i+NX*j] = 0.25*(T_E + T_W + T_N + T_S);
            }
        }
        iter+=2;
    }
}

// main program

int main(int argc, char* argv[])
{
    // allocate memory on the host for storing temperature at all nodes points

    float *T     = new float [NX*NY];    // current time level
    float *T_new = new float [NX*NY];    // new time level

    // initial condition

    for(int i=0; i<NX; i++) 
    {
       	for(int j=0; j<NY; j++) 
        {
            if (i==0)
            {
                T[i+NX*j] = 1.0;
                T_new[i+NX*j] = 1.0;
            }
            else
            {
                T[i+NX*j] = 0.0;
                T_new[i+NX*j] = 0.0;
            }
        }
    }

    // run Jacobi iteration
 
    Jacobi(T,T_new);
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

    // free up memory
    delete T;
    delete T_new;

    return 0;
}
