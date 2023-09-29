#include "poc-kokkos-rs/include/hello.hpp"
#include "omp.h"

#include <cstdio>

void say_hello() { printf("Hello CPP!\n"); }

void say_many_hello() {
#pragma omp parallel
  {
    int id = omp_get_thread_num();
    printf("Hello CPP! from OpenMP thread #%i\n", id);
  }
}