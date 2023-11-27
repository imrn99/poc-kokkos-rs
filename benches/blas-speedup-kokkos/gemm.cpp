// THIS CODE IS MADE FOR COMPILATION USING A PROPER KOKKOS SETUP.
// REQUIRES C++20
// COMPILE USING OPENMP BACKEND TO HAVE SOMETHING COMPARABLE TO RAYON
//
// This file is here in order to provide a comparable implementation of the blas
// benchmarks using the Kokkos library. It is not by any means the best way to 
// write such kernels.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <execution>
#include <random>

#include <Kokkos_Core.hpp>

#define DATA_SIZE 10
#define N_REPEAT 100

int main( int argc, char* argv[] )
{ 
    Kokkos::initialize(argc, argv);
    {
        // Readability
        typedef Kokkos::View<double**, Kokkos::LayoutRight>  MatRight;
        typedef Kokkos::View<double**, Kokkos::LayoutLeft>  MatLeft;

        // declare data

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        const uint64_t length = pow(2, DATA_SIZE);

        // runtime dims
        MatRight A("matA", length, length);
        MatLeft B("matB", length, length);
        MatRight C("matC", length, length);
        double alpha;
        double beta; 

        // fill with rand doubles
        alpha = dis(gen);
        beta = dis(gen);
        for (int ii = 0; ii < length; ii++) {
        for (int jj = 0; jj < length; jj++) {
            A(ii,jj) = dis(gen);
            B(ii,jj) = dis(gen);
            C(ii,jj) = dis(gen);
        }}

        // run the kernel N_REPEAT times

        std::chrono::duration<double> times[N_REPEAT];
        for (int idx = 0; idx < N_REPEAT; idx++) {
            
            const auto start{std::chrono::steady_clock::now()};   // start timer
            Kokkos::parallel_for("GEMM kernel", length, KOKKOS_LAMBDA(const uint64_t i) {
                for (uint64_t j = 0; j < length; j++) {
                    // this computation is the most costly part of the kernel
                    // trying to turn this into a proper reduction significantly
                    // obfuscate the code.
                    // I think this is pretty interesting since it can be done with 
                    // decent performances using just one line in Rust.
                    double AB_ij = 0.0; 
                    for (uint64_t k = 0; k < length; k++) { AB_ij += A(i,k) * B(k,j); }
                    // assign to C
                    C(i, j) = alpha * AB_ij + beta * C(i, j);
                }
            });
            const auto end{std::chrono::steady_clock::now()};     // end timer

            times[idx] = {end - start}; // save duration
            std::cout << "iteration " << idx << ": " << times[idx] << '\n'; // print duration
        }

        // process times
        double avg = 0.0;
        for (auto t : times) {
            avg += t.count();
        }
        avg /= (double) N_REPEAT;
        printf("average time: %fs\n", avg);

        double variance = 0.0;
        for (auto t : times) {
            variance += pow(t.count() - avg, 2.0);
        }
        variance /= (double) N_REPEAT;
        double stddev = sqrt(variance);
        printf("standard deviation: %.5fs\n", stddev);

    }
    Kokkos::finalize();
}