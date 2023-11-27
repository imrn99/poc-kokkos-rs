#!/bin/bash

cmake -DKokkos_ENABLE_OPENMP=ON -B build/
cmake --build build --parallel
