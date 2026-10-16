// ==============================================================================
// GPU Architecture Grid Search & Microbenchmark Execution Engine
// Author: Irene Therese Joseph (github.com/ireneyk)
//
// Explores the hardware parameter space:
// - Thread Block dimensions (warp multiples: 32, 64, 128, 256, 512, 1024)
// - Grid dimensions (thread block counts vs multiprocessor occupancy)
// - Statistically rigorous execution timing using adaptive sampling
// ==============================================================================

#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <string>

#include "./i_kernel.h"
#include "./metrics.h"
#include "./occupancy.h"
#include "./reporter.h"
#include "./timer.h"

namespace warpscope {

// Relative precision threshold for the adaptive sampler:
// Stops running iterations once the 95% confidence interval margin of error
// falls below 35% of the sample standard deviation.
constexpr double kRequiredPrecision = 0.35;

template <typename KernelFunc>
float TimeKernel(IKernel<KernelFunc> &ex, int num_blocks, int block_size) {
  CudaTimer timer;
  timer.Start();
  ex.RunKernel(num_blocks, block_size);
  timer.Stop();

  // Ensure GPU pipeline drains completely before reading timestamps
  cudaDeviceSynchronize();

  return timer.ElapsedMilliseconds();
}

template <typename KernelFunc>
tl::expected<AdaptiveSampler, ErrorInfo> RepeatUntil(double required_precision,
                                                     IKernel<KernelFunc> &ex,
                                                     int num_blocks,
                                                     int block_size) {
  AdaptiveSampler stats(required_precision);
  bool skip_first = true;

  while (stats.ShouldContinue()) {
    ex.Setup();

    float time = TimeKernel(ex, num_blocks, block_size);

    if (0 != ex.CheckResults()) {
      ex.Cleanup();
      return tl::make_unexpected(ErrorInfo(ErrorInfo::kUnexpectedKernelResult,
                                           "Functional verification error in kernel execution"));
    }
    ex.Cleanup();

    // Discard the initial run to eliminate cold L2/L1 cache misses,
    // unified memory first-touch overheads, and driver warmup jitter.
    if (skip_first) {
      skip_first = false;
      continue;
    }
    stats.Update(time);
  }
  return stats;
}

inline void PrintResults(const std::string& header, const Metrics& metrics) {
  Reporter::PrintResults(header + " best latency    : ",
                         metrics.get_metrics(Condition::kMinTime));
  Reporter::PrintResults(header + " best bandwidth  : ",
                         metrics.get_metrics(Condition::kMaxBandwidth));
  Reporter::PrintResults(header + " best occupancy  : ",
                         metrics.get_metrics(Condition::kMaxOccupancy));
}

// Searches grid parameter space for kernels utilizing grid-stride loops
template <typename KernelFunc>
Metrics RunStridedSearch(const cudaDeviceProp &hardware_info,
                         IKernel<KernelFunc> &ex) {
  Metrics metrics;
  auto kernel_info = ex.GetKernelInfo();

  auto block_size_gen = ex.GetBlockSizeGenerator();
  while (auto block_size_opt = block_size_gen->Next()) {
    auto num_blocks_gen = ex.GetNumBlocksGenerator();
    while (auto num_blocks_opt = num_blocks_gen->Next()) {
      int num_blocks = *num_blocks_opt;
      int block_size = *block_size_opt;

      // Avoid excessively overprovisioning thread blocks past problem size
      if (static_cast<size_t>(num_blocks) * block_size > static_cast<size_t>(kernel_info.n)) {
        num_blocks = static_cast<int>(num_blocks / 2.0 * 1.1);
      }

      Reporter::PrintResultsHeader(num_blocks, block_size);
      auto occupancy =
          Occupancy(hardware_info, num_blocks, block_size, ex.GetKernel());

      auto stats_res =
          RepeatUntil(kRequiredPrecision, ex, num_blocks, block_size);

      if (!stats_res) {
        std::cout << " [execution error or verification failure]" << std::endl;
        continue;
      }
      auto mean_res = stats_res->EstimatedMean();
      if (!mean_res || 0.0 == *mean_res) {
        std::cout << " [failed to converge on mean]" << std::endl;
        continue;
      }

      auto time_in_ms = *mean_res;
      auto time_in_seconds = time_in_ms / 1000.0;
      auto bandwidth = kernel_info.total_bytes / time_in_seconds;
      double gflops = (kernel_info.total_flops > 0 && time_in_seconds > 0)
                          ? (kernel_info.total_flops / 1e9) / time_in_seconds
                          : 0.0;
      double intensity = kernel_info.GetArithmeticIntensity();

      Data current_metrics{num_blocks, block_size, time_in_ms, bandwidth,
                           occupancy, gflops, intensity};
      metrics.UpdateAll(current_metrics);
      std::cout << ", ";
      Reporter::PrintResultsData(current_metrics, stats_res->NumSamples());

      if (static_cast<size_t>(num_blocks) * block_size > static_cast<size_t>(kernel_info.n)) {
        break;
      }
    }
    PrintResults(kernel_info.name + " current", metrics);
  }
  PrintResults(kernel_info.name + " final", metrics);
  return metrics;
}

// Searches block sizes for 1:1 mapped kernels (num_blocks derived directly from N / block_size)
template <typename KernelFunc>
Metrics RunUnstridedSearch(const cudaDeviceProp &hardware_info,
                           IKernel<KernelFunc> &ex) {
  Metrics metrics;

  int optimal_num_blocks = 0;
  int optimal_block_size = 0;
  OptimizeOccupancy(hardware_info, optimal_num_blocks, optimal_block_size, ex.GetKernel());
  std::cout << "Occupancy API baseline recommendation: <<"
            << optimal_num_blocks << ", " << optimal_block_size << ">>" << std::endl;

  auto kernel_info = ex.GetKernelInfo();
  auto block_size_gen = ex.GetBlockSizeGenerator();

  while (auto block_size_opt = block_size_gen->Next()) {
    int block_size = *block_size_opt;
    int num_blocks = (kernel_info.n + block_size - 1) / block_size;

    Reporter::PrintResultsHeader(num_blocks, block_size);
    auto occupancy =
        Occupancy(hardware_info, num_blocks, block_size, ex.GetKernel());

    auto stats_res =
        RepeatUntil(kRequiredPrecision, ex, num_blocks, block_size);

    if (!stats_res) {
      std::cout << " [execution error or verification failure]" << std::endl;
      continue;
    }
    auto mean_res = stats_res->EstimatedMean();
    if (!mean_res || 0.0 == *mean_res) {
      std::cout << " [failed to converge on mean]" << std::endl;
      continue;
    }

    auto time_in_ms = *mean_res;
    auto time_in_seconds = time_in_ms / 1000.0;
    auto bandwidth = kernel_info.total_bytes / time_in_seconds;
    double gflops = (kernel_info.total_flops > 0 && time_in_seconds > 0)
                        ? (kernel_info.total_flops / 1e9) / time_in_seconds
                        : 0.0;
    double intensity = kernel_info.GetArithmeticIntensity();

    Data current_metrics{num_blocks, block_size, time_in_ms, bandwidth,
                         occupancy, gflops, intensity};
    metrics.UpdateAll(current_metrics);
    std::cout << ", ";
    Reporter::PrintResultsData(current_metrics, stats_res->NumSamples());
  }

  PrintResults(kernel_info.name + " final", metrics);
  return metrics;
}

}  // namespace warpscope
