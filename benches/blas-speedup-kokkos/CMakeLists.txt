cmake_minimum_required(VERSION 3.16)

# set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm/bin/clang")     # uncomment if using homebrew clang
# set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++") # uncomment if using homebrew clang

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE "Release")

project(KokkosBenchmarks)
add_subdirectory($KOKKOS_INSTALL_FOLDER dep/kokkos) # add kokkos files dep

add_executable(gemm gemm.cpp)
add_executable(gemv gemv.cpp)
add_executable(axpy axpy.cpp)

target_link_libraries(gemm Kokkos::kokkos)
target_link_libraries(gemv Kokkos::kokkos)
target_link_libraries(axpy Kokkos::kokkos)
