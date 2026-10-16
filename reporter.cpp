// ==============================================================================
// GPU Architecture Performance Profiling Reporter
// Author: Irene Therese Joseph (github.com/ireneyk)
// ==============================================================================

#include "reporter.h"

#include <cstdio>
#include <iomanip>
#include <iostream>
#include <string>

#include "metrics.h"

namespace warpscope {

std::string Reporter::FormatToSI(double number) {
  struct Scale {
    double divisor;
    char prefix;
  };

  const Scale scales[] = {
      {1e24, 'Y'}, {1e21, 'Z'}, {1e18, 'E'}, {1e15, 'P'}, {1e12, 'T'},
      {1e9, 'G'},  {1e6, 'M'},  {1e3, 'K'},  {1, ' '}  // No prefix if < 1000.
  };

  Scale selected_scale = {1, ' '};
  for (const auto &scale : scales) {
    if (number >= scale.divisor) {
      selected_scale = scale;
      break;
    }
  }

  double scaled_number = number / selected_scale.divisor;
  char formatted_string[50];

  if (selected_scale.prefix == ' ') {
    snprintf(formatted_string, sizeof(formatted_string), "%.2f", scaled_number);
  } else {
    snprintf(formatted_string, sizeof(formatted_string), "%.2f%c",
             scaled_number, selected_scale.prefix);
  }

  return std::string(formatted_string);
}

std::string Reporter::FormatWithCommas(int n) {
  std::string result = std::to_string(n);
  for (int i = static_cast<int>(result.size()) - 3; i > 0; i -= 3) {
    result.insert(i, ",");
  }
  return result;
}

void Reporter::PrintResultsHeader(int num_blocks, int block_size) {
  std::cout << "<<numBlocks, blockSize>> = <<" << std::setw(10)
            << Reporter::FormatWithCommas(num_blocks) << ", " << std::setw(5)
            << Reporter::FormatWithCommas(block_size) << ">>";
}

void Reporter::PrintResultsData(Data data, std::optional<int> num_samples) {
  std::cout << "occupancy: " << std::fixed << std::setprecision(2)
            << std::setw(5) << data.occupancy;

  std::cout << ", bandwidth: " << std::setw(10)
            << Reporter::FormatToSI(data.bandwidth) << "B/s";

  std::cout << ", time: " << std::fixed << std::setprecision(2) << std::setw(7)
            << data.time_ms << " ms";

  if (data.gflops > 0.0) {
    std::cout << ", throughput: " << std::setw(7) << std::fixed
              << std::setprecision(1) << data.gflops << " GFLOP/s";
  }

  if (num_samples.has_value()) {
    std::cout << " (over " << *num_samples << " runs)";
  }

  std::cout << std::endl;
}

void Reporter::PrintResults(std::string prefix, Data data) {
  std::cout << prefix;
  Reporter::PrintResultsHeader(data.num_blocks, data.block_size);
  std::cout << std::endl;
  std::cout << "                  ";
  PrintResultsData(data, std::nullopt);
}

}  // namespace warpscope
