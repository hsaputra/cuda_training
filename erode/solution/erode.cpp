// calculate throughput of image erosion

#include <stdio.h>
#include <arrayfire.h>
using namespace af;

const int n = 1024; // size of image (n-by-n)
const int time_total = 7; // total seconds to run for

int main()
{
    try {
        info();

        int bytes = n*n*sizeof(bool);
        bool *h_src = array::pinned<bool>(bytes);
        bool *h_dst = array::pinned<bool>(bytes);

        array kernel = randu(3,3) > .5; // small boolean kernel

        int niter = 0;
        timer t = timer::start();
        while (timer::stop(t) < time_total) {

            // transfer to device
            array src(n, n, h_src);

            // process
            array dst = erode(src, kernel);

            // transfer to host
            dst.host(h_dst);

            niter++; // iteration complete
        }

        printf("FPS %.3f\n", niter / timer::stop(t));


        array::free(h_src);
        array::free(h_dst);

    } catch (af::exception &e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
    }

    return 0;
}
