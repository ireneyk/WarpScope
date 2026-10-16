// ==============================================================================
// GPU Architecture Occupancy & Hardware Resource Allocation Modeler
// Author: Irene Therese Joseph (github.com/ireneyk)
//
// Implementations for SM resource partitioning, warp budgeting, and
// bottleneck diagnosis (register pressure, shared memory, block limits).
// ==============================================================================

#include "occupancy.h"

#include <algorithm>
#include <sstream>

namespace warpscope {

OccupancyAnalysis DiagnoseOccupancy(const cudaDeviceProp &props,
                                    int block_size,
                                    int regs_per_thread,
                                    size_t dynamic_shmem_bytes,
                                    size_t static_shmem_bytes) {
  OccupancyAnalysis result{};
  result.theoretical_occupancy = 0.0;
  result.active_warps = 0;
  result.max_warps = (props.warpSize > 0) ? (props.maxThreadsPerMultiProcessor / props.warpSize) : 64;
  result.active_blocks_per_sm = 0;
  result.primary_limiter = OccupancyLimiter::kNone;
  result.limiter_description = "Optimal or unlimited";

  if (block_size <= 0 || props.warpSize <= 0) {
    result.limiter_description = "Invalid block size or warp configuration";
    return result;
  }

  // Check warp alignment
  if (block_size % props.warpSize != 0) {
    result.primary_limiter = OccupancyLimiter::kBlockSizeAlignment;
    result.limiter_description = "Block size is not a multiple of warp size (32); wastes warp scheduling slots";
  }

  int warps_per_block = (block_size + props.warpSize - 1) / props.warpSize;

  // 1. Thread count limitation per SM
  int limit_by_threads = props.maxThreadsPerMultiProcessor / block_size;

  // 2. Maximum blocks per SM supported by hardware architecture
  int max_blocks_hw = props.maxBlocksPerMultiProcessor;
  if (max_blocks_hw <= 0) {
    // Fallback based on compute capability (Ampere/Ada/Hopper allow up to 32 blocks per SM, earlier allowed 16 or 32)
    max_blocks_hw = (props.major >= 8) ? 32 : 16;
  }
  int limit_by_blocks = max_blocks_hw;

  // 3. Register file limitation
  // Hardware allocates registers in chunks per warp (typically 256 regs per chunk).
  int limit_by_regs = max_blocks_hw;
  if (regs_per_thread > 0 && props.regsPerMultiprocessor > 0) {
    // Registers per thread rounded to hardware allocation quantum (typically 4 on modern SMs)
    int reg_quantum = 4;
    int aligned_regs = ((regs_per_thread + reg_quantum - 1) / reg_quantum) * reg_quantum;
    // Register allocation granularity per warp = aligned_regs * 32
    int regs_per_warp = aligned_regs * props.warpSize;
    // Granularity of 256 registers per allocation unit
    regs_per_warp = ((regs_per_warp + 255) / 256) * 256;
    int regs_per_block = regs_per_warp * warps_per_block;
    if (regs_per_block > 0) {
      limit_by_regs = props.regsPerMultiprocessor / regs_per_block;
    }
  }

  // 4. Shared memory limitation
  int limit_by_shmem = max_blocks_hw;
  size_t total_shmem_per_block = dynamic_shmem_bytes + static_shmem_bytes;
  if (total_shmem_per_block > 0 && props.sharedMemPerMultiprocessor > 0) {
    // Shared memory allocation unit (typically 256 or 512 bytes)
    size_t shmem_unit = 256;
    size_t aligned_shmem = ((total_shmem_per_block + shmem_unit - 1) / shmem_unit) * shmem_unit;
    limit_by_shmem = static_shmem_bytes > 0 ? 
      static_cast<int>(props.sharedMemPerMultiprocessor / aligned_shmem) : max_blocks_hw;
  }

  // Binding constraint
  int active_blocks = std::min({limit_by_threads, limit_by_blocks, limit_by_regs, limit_by_shmem});
  result.active_blocks_per_sm = std::max(0, active_blocks);
  result.active_warps = result.active_blocks_per_sm * warps_per_block;
  
  if (result.max_warps > 0) {
    result.theoretical_occupancy = static_cast<double>(result.active_warps) / static_cast<double>(result.max_warps);
    if (result.theoretical_occupancy > 1.0) result.theoretical_occupancy = 1.0;
  }

  // Diagnose primary bottleneck
  std::ostringstream ss;
  if (active_blocks == limit_by_regs && limit_by_regs < limit_by_threads) {
    result.primary_limiter = OccupancyLimiter::kRegisters;
    ss << "Register pressure (" << regs_per_thread << " regs/thread limits active blocks to " << limit_by_regs << " per SM)";
  } else if (active_blocks == limit_by_shmem && limit_by_shmem < limit_by_threads) {
    result.primary_limiter = OccupancyLimiter::kSharedMemory;
    ss << "Shared memory footprint (" << total_shmem_per_block << " bytes/block limits active blocks to " << limit_by_shmem << " per SM)";
  } else if (active_blocks == limit_by_blocks && limit_by_blocks < limit_by_threads) {
    result.primary_limiter = OccupancyLimiter::kMaxBlocksPerSM;
    ss << "Hardware block scheduling limit reached (" << max_blocks_hw << " blocks/SM)";
  } else if (active_blocks == limit_by_threads) {
    result.primary_limiter = OccupancyLimiter::kWarps;
    ss << "Thread/warp capacity per SM (" << block_size << " threads/block yields " << result.active_warps << "/" << result.max_warps << " max warps)";
  }

  result.limiter_description = ss.str();
  return result;
}

}  // namespace warpscope
