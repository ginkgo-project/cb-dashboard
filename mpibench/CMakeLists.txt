cmake_minimum_required(VERSION 3.21)
project(mpibench LANGUAGES CXX)

# Where the buffers live: device memory through the CUDA or HIP runtime (for
# GPU-aware MPI), or host memory for a test on a machine without GPUs.
set(MPIBENCH_GPU "none" CACHE STRING "Buffers: cuda, hip or none (host)")
set_property(CACHE MPIBENCH_GPU PROPERTY STRINGS cuda hip none)

find_package(MPI REQUIRED COMPONENTS CXX)

add_executable(mpibench mpibench.cpp)
target_compile_features(mpibench PRIVATE cxx_std_17)
target_link_libraries(mpibench PRIVATE MPI::MPI_CXX)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(MPIBENCH_GPU STREQUAL "cuda")
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(mpibench PRIVATE CUDA::cudart)
    target_compile_definitions(mpibench PRIVATE MPIBENCH_CUDA)
elseif(MPIBENCH_GPU STREQUAL "hip")
    find_package(hip REQUIRED)
    target_link_libraries(mpibench PRIVATE hip::host)
    target_compile_definitions(mpibench PRIVATE MPIBENCH_HIP)
elseif(NOT MPIBENCH_GPU STREQUAL "none")
    message(FATAL_ERROR "MPIBENCH_GPU must be cuda, hip or none, not ${MPIBENCH_GPU}")
endif()
