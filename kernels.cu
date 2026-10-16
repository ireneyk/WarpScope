// ==============================================================================
// GPU Architecture Kernel Library: Compute, Memory Hierarchy, & Warp Access
// Author: Irene Therese Joseph (github.com/ireneyk)
//
// Implementations of parallel workloads modeling hardware execution behavior.
// ==============================================================================

#include "kernels.h"

namespace warpscope {

// 1:1 Thread Mapping: Ideal when problem size fits grid dimensions evenly.
// Consecutive threads in a warp access consecutive floats -> single coalesced 128-byte transaction.
__global__ void AddUnstridedKernel(int n, float *x, float *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = x[i] + y[i];
  }
}

// Grid-Stride Loop: Decouples hardware grid size from problem size.
// Maximizes SM reuse when launching smaller thread grids, amortizing launch overhead.
__global__ void AddStridedKernel(int n, float *x, float *y) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    y[i] = x[i] + y[i];
  }
}

// Euclidean Distance Unstrided: Uses 64-bit float2 loads (8 bytes per thread).
// A full 32-thread warp reads 256 bytes (two 128-byte cache line sectors).
__global__ void EuclideanDistanceUnstridedKernel(int n, float2 *x, float2 *y,
                                                 float *distance) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float2 dp = subtract(y[i], x[i]);
    float dist = sqrtf(dot(dp, dp));
    distance[i] = dist;
  }
}

// Euclidean Distance with Grid-Stride loop.
__global__ void EuclideanDistanceStridedKernel(int n, float2 *x, float2 *y,
                                               float *distance) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride) {
    float2 dp = subtract(y[i], x[i]);
    float dist = sqrtf(dot(dp, dp));
    distance[i] = dist;
  }
}

// Naive GEMM: Bottlenecked by redundant global DRAM accesses.
// Every thread independently streams an entire row of A and column of B from global memory,
// causing heavy L2 thrashing and severe memory bandwidth saturation.
__global__ void MatrixMultiplyKernel(int matrix_dim, float *a, float *b,
                                     float *c) {
  int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  int row = thread_id / matrix_dim;
  int col = thread_id % matrix_dim;

  if (row < matrix_dim && col < matrix_dim) {
    float sum = 0.0f;
    for (int k = 0; k < matrix_dim; k++) {
      sum += a[row * matrix_dim + k] * b[k * matrix_dim + col];
    }
    c[row * matrix_dim + col] = sum;
  }
}

// Shared Memory Tiled GEMM:
// Leverages on-chip SRAM (Shared Memory / L1 cache) to stage matrix sub-blocks.
// Reduces global DRAM traffic by a factor of kTileDim (16x), raising arithmetic
// intensity from 0.25 FLOPs/byte up to ~4.0 FLOPs/byte, pushing execution into
// the compute-bound regime of the GPU Roofline.
__global__ void MatrixMultiplyTiledKernel(int matrix_dim, const float *a,
                                          const float *b, float *c) {
  __shared__ float s_a[kTileDim][kTileDim];
  __shared__ float s_b[kTileDim][kTileDim];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by * kTileDim + ty;
  int col = bx * kTileDim + tx;

  float sum = 0.0f;
  int num_tiles = (matrix_dim + kTileDim - 1) / kTileDim;

  for (int m = 0; m < num_tiles; ++m) {
    // Cooperative loading of tile A and B into on-chip shared memory
    int a_col = m * kTileDim + tx;
    if (row < matrix_dim && a_col < matrix_dim) {
      s_a[ty][tx] = a[row * matrix_dim + a_col];
    } else {
      s_a[ty][tx] = 0.0f;
    }

    int b_row = m * kTileDim + ty;
    if (col < matrix_dim && b_row < matrix_dim) {
      s_b[ty][tx] = b[b_row * matrix_dim + col];
    } else {
      s_b[ty][tx] = 0.0f;
    }

    // Barrier synchronization ensures all warp threads finished loading before computation
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < kTileDim; ++k) {
      sum += s_a[ty][k] * s_b[k][tx];
    }

    // Barrier synchronization ensures tile computation is done before loading next tile
    __syncthreads();
  }

  if (row < matrix_dim && col < matrix_dim) {
    c[row * matrix_dim + col] = sum;
  }
}

// Coalesced Memory Access:
// Adjacent threads access contiguous 4-byte words.
// Hardware memory controllers merge warp requests into minimal 32-byte / 128-byte transactions.
__global__ void MemoryCoalescedKernel(int n, const float *in, float *out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = in[i];
  }
}

// Strided Memory Access:
// Demonstrates uncoalesced transactions. As stride increases, threads in a warp
// hit different 128-byte cache lines, generating up to 32 discrete memory requests
// per warp instruction and wasting bus bandwidth on unused words.
__global__ void MemoryStridedKernel(int n, int stride, const float *in, float *out) {
  int i = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
  if (i < n) {
    out[i] = in[i];
  }
}

}  // namespace warpscope
