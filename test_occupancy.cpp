// ==============================================================================
// Unit Tests: GPU Architecture Occupancy & Hardware Limiter Diagnostics
// Author: Irene Therese Joseph (github.com/ireneyk)
// ==============================================================================

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "../src/occupancy.h"

namespace warpscope {

TEST_CASE("DiagnoseOccupancy::HardwareResourceLimits", "[Occupancy]") {
  cudaDeviceProp props{};
  props.warpSize = 32;
  props.maxThreadsPerMultiProcessor = 2048;  // Standard for Ampere / Ada Lovelace SMs
  props.maxBlocksPerMultiProcessor = 32;
  props.regsPerMultiprocessor = 65536;       // 64K 32-bit registers per SM
  props.sharedMemPerMultiprocessor = 102400; // 100 KB
  props.major = 8;
  props.minor = 6;

  SECTION("Detects non-warp-aligned block sizing") {
    auto analysis = DiagnoseOccupancy(props, 50, 32, 0, 0);
    REQUIRE(analysis.primary_limiter == OccupancyLimiter::kBlockSizeAlignment);
  }

  SECTION("Achieves 100% occupancy with optimal 256-thread blocks and balanced registers") {
    // 256 threads/block, 32 regs/thread -> 8192 regs/block -> 8 blocks fit in 65536 registers.
    // 8 blocks * 256 threads = 2048 threads -> 64 warps (100% capacity)
    auto analysis = DiagnoseOccupancy(props, 256, 32, 0, 0);
    REQUIRE(analysis.active_blocks_per_sm == 8);
    REQUIRE(analysis.active_warps == 64);
    REQUIRE(analysis.theoretical_occupancy == 1.0);
  }

  SECTION("Diagnoses register pressure as primary bottleneck under heavy register usage") {
    // 256 threads/block, 128 regs/thread -> 32,768 regs/block.
    // SM can only hold 65536 / 32768 = 2 active blocks (16 warps out of 64 max = 25% occupancy)
    auto analysis = DiagnoseOccupancy(props, 256, 128, 0, 0);
    REQUIRE(analysis.primary_limiter == OccupancyLimiter::kRegisters);
    REQUIRE(analysis.active_blocks_per_sm <= 2);
    REQUIRE(analysis.theoretical_occupancy <= 0.25);
  }

  SECTION("Diagnoses shared memory capacity bottleneck") {
    // 256 threads/block, low registers (16), but large shared memory allocation (48 KB per block)
    // SM capacity of 100 KB allows at most 2 blocks to be concurrently scheduled
    auto analysis = DiagnoseOccupancy(props, 256, 16, 0, 48 * 1024);
    REQUIRE(analysis.primary_limiter == OccupancyLimiter::kSharedMemory);
    REQUIRE(analysis.active_blocks_per_sm <= 2);
  }
}

}  // namespace warpscope
