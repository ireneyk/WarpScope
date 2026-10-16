// ==============================================================================
// GPU Architecture Occupancy & Hardware Resource Allocation Modeler
// Author: Irene Therese Joseph (github.com/ireneyk)
//
// Models Streaming Multiprocessor (SM) warp residency, resource limits
// (registers, shared memory, warp schedulers), and theoretical occupancy.
// ==============================================================================

#pragma once

#include <cuda_runtime.h>
#include <cassert>
#include <string>

#include "./adaptive_sampler.h"
#include "./i_kernel.h"

namespace warpscope {

enum class OccupancyLimiter {
  kNone,
  kWarps,
  kRegisters,
  kSharedMemory,
  kMaxBlocksPerSM,
  kBlockSizeAlignment
};

struct OccupancyAnalysis {
  double theoretical_occupancy;
  int active_warps;
  int max_warps;
  int active_blocks_per_sm;
  OccupancyLimiter primary_limiter;
  std::string limiter_description;
};

// Returns theoretical occupancy (0.0 to 1.0) given device properties,
// kernel function pointer, and block size.
template <typename KernelFunc>
double Occupancy(const cudaDeviceProp &props, int num_blocks, int block_size,
                 KernelFunc kernel) {
  if (props.warpSize <= 0 || block_size <= 0) return 0.0;
  
  int active_blocks = 0;
  cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, kernel, block_size, 0);
  if (err != cudaSuccess || active_blocks <= 0) {
    return 0.0;
  }

  int active_warps = active_blocks * (block_size / props.warpSize);
  int max_warps = props.maxThreadsPerMultiProcessor / props.warpSize;
  if (max_warps <= 0) return 0.0;

  double occ = static_cast<double>(active_warps) / static_cast<double>(max_warps);
  return (occ > 1.0) ? 1.0 : occ;
}

// Architectural diagnosis of the limiting hardware resource on an SM.
// Analyzes register allocation, shared memory allocations, and warp scheduler constraints.
OccupancyAnalysis DiagnoseOccupancy(const cudaDeviceProp &props,
                                    int block_size,
                                    int regs_per_thread,
                                    size_t dynamic_shmem_bytes,
                                    size_t static_shmem_bytes = 0);

// Calculate the optimal num_blocks and block_size for the given kernel on
// target GPU architecture using occupancy heuristics.
template <typename KernelFunc>
void OptimizeOccupancy(const cudaDeviceProp &hardware_info, int &num_blocks,
                       int &block_size, KernelFunc kernel) {
  int min_grid_size = 0;
  cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel, 0, 0);

  int num_blocks_per_sm = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel,
                                                block_size, 0);

  int num_sms = hardware_info.multiProcessorCount;
  num_blocks = num_blocks_per_sm * num_sms;

  double current_occupancy =
      Occupancy(hardware_info, num_blocks, block_size, kernel);

  // Sweep downward in warp-sized (32-thread) increments to explore alternate
  // high-occupancy block configurations that may reduce register pressure.
  for (int bs = block_size; bs >= 32; bs -= 32) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm, kernel,
                                                  bs, 0);
    int nb = num_blocks_per_sm * num_sms;
    double occ = Occupancy(hardware_info, nb, bs, kernel);

    if (occ > current_occupancy) {
      num_blocks = nb;
      block_size = bs;
      current_occupancy = occ;
    }

    if (current_occupancy >= 0.99) break;  // Saturated SM capacity
  }
}

}  // namespace warpscope
