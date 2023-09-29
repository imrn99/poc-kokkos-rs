#include "openmp-parallel/many_hello.hpp"

#include <cstdio>

void say_many_hello() {
#pragma omp parallel
  {
    int id = omp_get_thread_num();
    printf("Hello from OpenMP thread #%i\n", id);
  }
}
