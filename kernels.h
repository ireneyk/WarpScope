// ==============================================================================
// GPU Architecture Kernel Library: Compute, Memory Hierarchy, & Warp Access
// Author: Irene Therese Joseph (github.com/ireneyk)
//
// Workloads for evaluating GPU microarchitectural behavior:
// - Memory Subsystem: Coalesced vs Strided global memory transactions
// - Memory Management: Unified Memory (demand paging) vs Explicit PCIe copies
// - Compute Hierarchy: Naive global memory GEMM vs Shared Memory Tiled GEMM
// ==============================================================================

#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>

namespace warpscope {

constexpr int kTileDim = 16;

// Vector Add kernels: Evaluate warp-stride looping vs 1:1 thread mapping
__global__ void AddStridedKernel(int n, float *x, float *y);
__global__ void AddUnstridedKernel(int n, float *x, float *y);

// Euclidean Distance: Evaluates vector types (float2) and compute vs bandwidth ratio
__global__ void EuclideanDistanceStridedKernel(int n, float2 *x, float2 *y,
                                               float *distance);
__global__ void EuclideanDistanceUnstridedKernel(int n, float2 *x, float2 *y,
                                                 float *distance);

// Naive Matrix Multiply: High global memory traffic, low arithmetic intensity
__global__ void MatrixMultiplyKernel(int matrix_dim, float *a, float *b,
                                     float *c);

// Shared-Memory Tiled GEMM: Cooperative thread-block loading, L1/Shared Memory reuse,
// and avoidance of bank conflicts for compute-bound throughput
__global__ void MatrixMultiplyTiledKernel(int matrix_dim, const float *a,
                                          const float *b, float *c);

// Memory Coalescing Benchmarks: Demonstrates memory controller cache line (32B/128B) transaction efficiency
__global__ void MemoryCoalescedKernel(int n, const float *in, float *out);
__global__ void MemoryStridedKernel(int n, int stride, const float *in, float *out);

// Kernel function pointer types
using AddKernelFunc = void (*)(int, float *, float *);
using DistKernelFunc = void (*)(int, float2 *, float2 *, float *);
using MatrixMultiplyKernelFunc = void (*)(int, float *, float *, float *);
using MemoryBenchKernelFunc = void (*)(int, const float *, float *);

// Simple matrix representation
struct Matrix {
  int width;
  int height;
  float *elements;
};

// Device utility functions
inline __device__ float2 subtract(float2 a, float2 b) {
  return make_float2(a.x - b.x, a.y - b.y);
}

inline __device__ float dot(float2 a, float2 b) {
  return a.x * b.x + a.y * b.y;
}

}  // namespace warpscope
