// ==============================================================================
// GPU Architecture Microbenchmarking & Kernel Performance Profiler
// Author: Irene Therese Joseph (github.com/ireneyk)
//
// Explores GPU microarchitecture parameter spaces:
// 1. Memory Subsystem: Coalesced 128-byte warp accesses vs strided cache misses
// 2. Interconnect & Paging: Unified Memory on-demand migration vs explicit PCIe transfers
// 3. Compute Architecture: Naive global DRAM GEMM vs Shared-Memory Tiled GEMM
// 4. Multiprocessor Occupancy: Register pressure, warp residency, and latency hiding
// ==============================================================================

#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>
#include <string>

#include "examples/add_strided_managed.h"
#include "examples/add_strided_unmanaged.h"
#include "examples/add_unstrided_managed.h"
#include "examples/add_unstrided_unmanaged.h"
#include "examples/euclidean_distance_strided.h"
#include "examples/euclidean_distance_unstrided.h"
#include "examples/matrix_multiply.h"
#include "examples/matrix_multiply_tiled.h"
#include "occupancy.h"
#include "optimizer.h"

namespace ws = ::warpscope;

cudaDeviceProp QueryHardwareArchitecture() {
  int num_devices = 0;
  cudaGetDeviceCount(&num_devices);
  if (num_devices == 0) {
    std::cerr << "[ERROR] No CUDA-capable devices detected. Halting execution." << std::endl;
    exit(1);
  }

  std::cout << "\n=================================================================" << std::endl;
  std::cout << " GPU Hardware Architecture Discovery (Device Count: " << num_devices << ")" << std::endl;
  std::cout << "=================================================================" << std::endl;

  cudaDeviceProp props{};
  for (int i = 0; i < num_devices; i++) {
    cudaGetDeviceProperties(&props, i);
    std::cout << "Device [" << i << "]: " << props.name << std::endl;
    std::cout << "  Compute Capability           : " << props.major << "." << props.minor << std::endl;
    std::cout << "  Streaming Multiprocessors    : " << props.multiProcessorCount << " SMs" << std::endl;
    std::cout << "  Total Global DRAM            : " << props.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  L2 Cache Capacity            : " << props.l2CacheSize / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Memory Bus Width             : " << props.memoryBusWidth << " bits" << std::endl;
    std::cout << "  Memory Clock Rate            : " << props.memoryClockRate / 1000 << " MHz" << std::endl;

    // Peak theoretical DRAM bandwidth (GB/s): 2 * clock(Hz) * (bus_bits/8)
    double peak_bw = 2.0 * (static_cast<double>(props.memoryClockRate) * 1e3) * (props.memoryBusWidth / 8.0) / 1e9;
    std::cout << "  Theoretical Peak DRAM BW     : " << std::fixed << std::setprecision(1) << peak_bw << " GB/s" << std::endl;

    std::cout << "  Warp Execution Size          : " << props.warpSize << " threads" << std::endl;
    std::cout << "  Max Resident Warps per SM    : " << props.maxThreadsPerMultiProcessor / props.warpSize << std::endl;
    std::cout << "  Max Threads per SM           : " << props.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "  Max Threads per Block        : " << props.maxThreadsPerBlock << std::endl;
    std::cout << "  32-bit Registers per SM      : " << props.regsPerMultiprocessor << std::endl;
    std::cout << "  Shared Memory per SM         : " << props.sharedMemPerMultiprocessor / 1024 << " KB" << std::endl;
    std::cout << "  PCIe Location                : Domain " << props.pciDomainID << ", Bus " << props.pciBusID << ", Device " << props.pciDeviceID << std::endl;
  }
  std::cout << "=================================================================\n" << std::endl;
  return props;
}

int main(void) {
  auto hardware_info = QueryHardwareArchitecture();

  int max_num_blocks = hardware_info.maxThreadsDim[0] *
                       hardware_info.maxThreadsDim[1] *
                       hardware_info.maxThreadsDim[2];
  int max_block_size = hardware_info.maxThreadsPerBlock;

  std::cout << "Microarchitecture Constraints:" << std::endl;
  std::cout << "  Max Grid Dimension X*Y*Z     : " << ws::Reporter::FormatWithCommas(max_num_blocks) << std::endl;
  std::cout << "  Max Thread Block Dimension   : " << ws::Reporter::FormatWithCommas(max_block_size) << "\n" << std::endl;

  // ---------------------------------------------------------------------------
  // Step 1: Baseline Functional Verification Across Workloads
  // ---------------------------------------------------------------------------
  std::cout << "-----------------------------------------------------------------" << std::endl;
  std::cout << " 1. Functional Verification & Baseline Runs" << std::endl;
  std::cout << "-----------------------------------------------------------------" << std::endl;

  std::cout << "Running: Add (Strided, Managed Memory)..." << std::endl;
  ws::AddStridedManaged add_strided_managed(max_num_blocks, max_block_size);
  add_strided_managed.Run(4096, 256);

  std::cout << "Running: Add (Unstrided, Managed Memory)..." << std::endl;
  ws::AddUnstridedManaged add_unstrided_managed(max_num_blocks, max_block_size);
  add_unstrided_managed.Run(4096, 256);

  std::cout << "Running: Add (Strided, Explicit PCIe Memory)..." << std::endl;
  ws::AddStridedUnmanaged add_strided_unmanaged(max_num_blocks, max_block_size);
  add_strided_unmanaged.Run(4096, 256);

  std::cout << "Running: Add (Unstrided, Explicit PCIe Memory)..." << std::endl;
  ws::AddUnstridedUnmanaged add_unstrided_unmanaged(max_num_blocks, max_block_size);
  add_unstrided_unmanaged.Run(4096, 256);

  std::cout << "Running: Euclidean Distance (Strided)..." << std::endl;
  ws::EuclideanDistanceStrided dist_strided(max_num_blocks, max_block_size);
  dist_strided.Run(4096, 256);

  std::cout << "Running: Euclidean Distance (Unstrided)..." << std::endl;
  ws::EuclideanDistanceUnstrided dist_unstrided(max_num_blocks, max_block_size);
  dist_unstrided.Run(4096, 256);

  std::cout << "Running: Matrix Multiply (Naive Global Memory)..." << std::endl;
  ws::MatrixMultiply matrix_multiply_naive(max_num_blocks, max_block_size);
  matrix_multiply_naive.Run(8192, 32);

  std::cout << "Running: Matrix Multiply (Shared-Memory Tiled GEMM)..." << std::endl;
  ws::MatrixMultiplyTiled matrix_multiply_tiled(max_num_blocks, max_block_size);
  matrix_multiply_tiled.Run(1024, 256);

  // ---------------------------------------------------------------------------
  // Step 2: Microarchitectural Search - Memory Subsystems & Interconnect
  // ---------------------------------------------------------------------------
  std::cout << "\n-----------------------------------------------------------------" << std::endl;
  std::cout << " 2. Architectural Sweeps: Memory Access & Interconnect Latency" << std::endl;
  std::cout << "-----------------------------------------------------------------" << std::endl;

  ws::Optimizer<ws::AddKernelFunc> add_optimizer;
  add_optimizer.AddStrategy("Strided, Managed",
                            ws::RunStridedSearch<ws::AddKernelFunc>,
                            &add_strided_managed);
  add_optimizer.AddStrategy("Unstrided, Managed",
                            ws::RunUnstridedSearch<ws::AddKernelFunc>,
                            &add_unstrided_managed);
  add_optimizer.AddStrategy("Strided, Explicit-PCIe",
                            ws::RunStridedSearch<ws::AddKernelFunc>,
                            &add_strided_unmanaged);
  add_optimizer.AddStrategy("Unstrided, Explicit-PCIe",
                            ws::RunUnstridedSearch<ws::AddKernelFunc>,
                            &add_unstrided_unmanaged);

  // Comparison 1: Strided vs Unstrided within Managed Memory
  std::string set_name = "Vector Add: Warp Striding vs Coalesced Mapping (Managed Memory)";
  add_optimizer.CreateSet(set_name, {"Strided, Managed", "Unstrided, Managed"});
  add_optimizer.OptimizeSet(set_name, hardware_info);

  // Comparison 2: Managed Memory (Demand Paging) vs Explicit Host-to-Device Transfers
  set_name = "Interconnect: Unified Memory Demand Paging vs Explicit Host-Device Transfers";
  add_optimizer.CreateSet(set_name, {"Strided, Managed", "Strided, Explicit-PCIe"});
  add_optimizer.OptimizeSet(set_name, hardware_info);

  // Comparison 3: 4-Way Cross Matrix (Loop Structure x Memory Interconnect)
  set_name = "Memory Subsystem: 4-Way Evaluation (Loop Structure x Interconnect)";
  add_optimizer.CreateSet(set_name, {"Strided, Managed", "Unstrided, Managed",
                                     "Strided, Explicit-PCIe", "Unstrided, Explicit-PCIe"});
  add_optimizer.OptimizeSet(set_name, hardware_info);

  // ---------------------------------------------------------------------------
  // Step 3: Vector Data Types & Double-Word Loads (Euclidean Distance)
  // ---------------------------------------------------------------------------
  std::cout << "\n-----------------------------------------------------------------" << std::endl;
  std::cout << " 3. Vector Types & Coalescing: Euclidean Distance (float2 Loads)" << std::endl;
  std::cout << "-----------------------------------------------------------------" << std::endl;

  ws::Optimizer<ws::DistKernelFunc> dist_optimizer;
  dist_optimizer.AddStrategy("Strided",
                             ws::RunStridedSearch<ws::DistKernelFunc>,
                             &dist_strided);
  dist_optimizer.AddStrategy("Unstrided",
                             ws::RunUnstridedSearch<ws::DistKernelFunc>,
                             &dist_unstrided);

  set_name = "Euclidean Distance: 64-bit Vector Loads (float2)";
  dist_optimizer.CreateSet(set_name, {"Strided", "Unstrided"});
  dist_optimizer.OptimizeSet(set_name, hardware_info);

  // ---------------------------------------------------------------------------
  // Step 4: Compute Architecture Deep Dive - Global DRAM GEMM vs Tiled GEMM
  // ---------------------------------------------------------------------------
  std::cout << "\n-----------------------------------------------------------------" << std::endl;
  std::cout << " 4. Compute Architecture: Memory Hierarchy Exploitation (GEMM)" << std::endl;
  std::cout << "-----------------------------------------------------------------" << std::endl;

  ws::Optimizer<ws::MatrixMultiplyKernelFunc> gemm_optimizer;
  gemm_optimizer.AddStrategy("Naive Global-DRAM",
                             ws::RunUnstridedSearch<ws::MatrixMultiplyKernelFunc>,
                             &matrix_multiply_naive);
  gemm_optimizer.AddStrategy("Shared-Memory Tiled",
                             ws::RunUnstridedSearch<ws::MatrixMultiplyKernelFunc>,
                             &matrix_multiply_tiled);

  set_name = "GEMM Memory Wall: Naive Global Memory vs Shared Memory Tiled Staging";
  gemm_optimizer.CreateSet(set_name, {"Naive Global-DRAM", "Shared-Memory Tiled"});
  gemm_optimizer.OptimizeSet(set_name, hardware_info);

  std::cout << "\nAll architectural evaluations completed successfully." << std::endl;
  return 0;
}