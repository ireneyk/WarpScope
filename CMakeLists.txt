# ==============================================================================
# GPU Architecture Microbenchmarking & Kernel Performance Profiler
# Author: Irene Therese Joseph (github.com/ireneyk)
#
# Hardware Performance Modeling, GPU Microarchitecture Analysis,
# Memory Subsystem Optimization, and Occupancy Exploration.
# ==============================================================================

cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(warpscope 
        VERSION 2.0.0
        DESCRIPTION "GPU Architecture Performance Modeling & Kernel Optimization Framework"
        HOMEPAGE_URL "https://github.com/ireneyk/warpscope"
        LANGUAGES CXX CUDA)

# Default to modern compute architectures (Turing, Ampere, Ada Lovelace, Hopper)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90" CACHE STRING "CUDA Architectures" FORCE)
endif()

add_subdirectory(src)

enable_testing()
add_subdirectory(tests)
