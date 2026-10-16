// ==============================================================================
// GPU Architecture Performance Profiling Reporter
// Author: Irene Therese Joseph (github.com/ireneyk)
//
// Formats hardware execution telemetry, metric units (SI scaling),
// thread block configurations, and benchmark statistics.
// ==============================================================================

#pragma once

#include <optional>
#include <string>

#include "./metrics.h"

namespace warpscope {

class Reporter {
 public:
  // Format numeric values with standard engineering SI prefixes (K, M, G, T, etc.)
  static std::string FormatToSI(double number);

  // Comma-separated integer formatting for readable block and thread counts
  static std::string FormatWithCommas(int n);

  // Print grid parameter header: <<numBlocks, blockSize>>
  static void PrintResultsHeader(int num_blocks, int block_size);

  // Print detailed timing, occupancy, bandwidth, and throughput data
  static void PrintResultsData(Data data, std::optional<int> num_samples);

  // Complete results block with prefix label
  static void PrintResults(std::string prefix, Data data);
};

}  // namespace warpscope
