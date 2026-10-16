// ==============================================================================
// GPU Architecture Kernel Interface & Workload Metadata
// Author: Irene Therese Joseph (github.com/ireneyk)
//
// Defines hardware execution interfaces, memory transfer sizes,
// floating-point operation counts, and grid configuration generators.
// ==============================================================================

#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "./generators.h"
#include "./kernels.h"

namespace warpscope {

struct KernelInfo {
  std::string name;
  int n;                // Problem dimension / element count
  size_t total_bytes;   // Memory footprint moved across memory hierarchy
  size_t total_flops;   // Floating point operations executed

  KernelInfo(const std::string& kernel_name, int problem_size, size_t bytes, size_t flops = 0)
      : name(kernel_name),
        n(problem_size),
        total_bytes(bytes),
        total_flops(flops) {}

  // Arithmetic Intensity = FLOPs / Byte (fundamental input to the Roofline Model)
  double GetArithmeticIntensity() const {
    if (total_bytes == 0) return 0.0;
    return static_cast<double>(total_flops) / static_cast<double>(total_bytes);
  }
};

template <typename KernelFunc>
class IKernel {
 public:
  virtual ~IKernel() = default;

  virtual KernelInfo GetKernelInfo() const = 0;
  virtual KernelFunc GetKernel() const = 0;
  virtual void Setup() = 0;
  virtual std::unique_ptr<IGridSizeGenerator> GetNumBlocksGenerator() const = 0;
  virtual std::unique_ptr<IGridSizeGenerator> GetBlockSizeGenerator() const = 0;
  virtual void RunKernel(int num_blocks, int block_size) = 0;
  virtual int CheckResults() = 0;
  virtual void Cleanup() = 0;

  // Single test execution helper with functional verification
  void Run(int num_blocks, int block_size) {
    Setup();
    RunKernel(num_blocks, block_size);
    int errors = CheckResults();
    if (0 == errors) {
      std::cout << "    [PASS] Functional verification passed (0 errors)" << std::endl << std::flush;
    } else {
      std::cout << "    [FAIL] Functional verification failed (" << errors << " errors)" << std::endl << std::flush;
    }
    Cleanup();
  }
};

}  // namespace warpscope