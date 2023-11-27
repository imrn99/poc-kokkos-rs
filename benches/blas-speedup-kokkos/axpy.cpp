#include <execution>

#include <Kokkos_Core.hpp>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <random>

#define DATA_SIZE 20
#define N_REPEAT 100


int main( int argc, char* argv[] )
{ 
    Kokkos::initialize(argc, argv);
    {
        // Readability
        typedef Kokkos::View<double*, Kokkos::LayoutLeft>   Vec; // technically layout doesn't matter here

        // declare data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        const uint64_t length = pow(2, DATA_SIZE);

        Vec x("vecx", length);
        Vec y("vecy", length);
        double alpha;

        // fill with rand doubles
        alpha = dis(gen);
        // need to parallelize this if possible
        Kokkos::parallel_for("inits", length, KOKKOS_LAMBDA(const uint64_t ii) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(0.0, 1.0);
            x(ii) = dis(gen);
            y(ii) = dis(gen);
        });

        // run the kernel N_REPEAT times

        std::chrono::duration<double> times[N_REPEAT];
        for (int idx = 0; idx < N_REPEAT; idx++) {
            
            const auto start{std::chrono::steady_clock::now()};   // start timer
            Kokkos::parallel_for("GEMM kernel", length, KOKKOS_LAMBDA(const uint64_t i) {
                // assign to y
                y(i) = alpha * x(i) + y(i);
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