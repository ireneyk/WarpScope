// ==============================================================================
// GPU Architecture Kernel Optimization & Strategy Evaluator
// Author: Irene Therese Joseph (github.com/ireneyk)
//
// Manages multi-strategy benchmarking suites, executes parameter space sweeps,
// cross-compares memory hierarchy strategies, and computes architectural speedup ratios.
// ==============================================================================

#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "./grid_searchers.h"
#include "./metrics.h"
#include "./reporter.h"

namespace warpscope {

template <typename KernelFunc>
class Optimizer {
 public:
  using SearchFunction = Metrics (*)(const cudaDeviceProp&, IKernel<KernelFunc>&);

  Optimizer() : name_("GPU Architectural Optimizer") {}
  explicit Optimizer(std::string name) : name_(std::move(name)) {}

  void AddStrategy(const std::string& name, SearchFunction func,
                   IKernel<KernelFunc>* kernel) {
    searches_[name] = {name, func,
                       std::unique_ptr<IKernel<KernelFunc>>(kernel), std::nullopt};
  }

  void CreateSet(const std::string& set_name,
                 const std::vector<std::string>& strategy_names) {
    sets_[set_name] = strategy_names;
  }

  const std::string& GetName() const { return name_; }

  void OptimizeSet(const std::string& set_name, const cudaDeviceProp& hardware_info) {
    std::cout << "\n=======================================================" << std::endl;
    std::cout << " Evaluating Architectural Optimization Set: " << set_name << std::endl;
    std::cout << "=======================================================" << std::endl;

    const auto& strategy_names = sets_[set_name];
    for (const auto& name : strategy_names) {
      auto& search = searches_[name];
      if (!search.result.has_value()) {
        std::cout << "\n--> Launching strategy: [" << name << "]..." << std::endl;
        search.result = search.func(hardware_info, *search.kernel);
      } else {
        std::cout << "\n--> Reusing cached profile for strategy: [" << name << "]" << std::endl;
      }
      PrintCurrentResults("Current ", name, *search.result);
    }
    PrintBestResults(set_name);
    PrintComparisonMatrix(set_name);
  }

 private:
  struct Search {
    std::string name;
    SearchFunction func;
    std::unique_ptr<IKernel<KernelFunc>> kernel;
    std::optional<Metrics> result;
  };

  void PrintCurrentResults(const std::string& header, const std::string& name,
                           const Metrics& result) {
    Reporter::PrintResults(header + "[" + name + "] Best Latency  : ",
                           result.get_metrics(Condition::kMinTime));
    Reporter::PrintResults(header + "[" + name + "] Best Bandwidth: ",
                           result.get_metrics(Condition::kMaxBandwidth));
    Reporter::PrintResults(header + "[" + name + "] Best Occupancy: ",
                           result.get_metrics(Condition::kMaxOccupancy));
  }

  void PrintBestResults(const std::string& set_name) const {
    std::cout << "\n-------------------------------------------------------" << std::endl;
    std::cout << " SUMMARY REPORT: " << set_name << std::endl;
    std::cout << " Strategies evaluated: " << std::endl;
    for (const auto& name : sets_.at(set_name)) {
      std::cout << "   * " << name << std::endl;
    }
    std::cout << "-------------------------------------------------------" << std::endl;

    PrintBestResult(set_name, "Lowest Latency", Condition::kMinTime);
    PrintBestResult(set_name, "Peak Effective Bandwidth", Condition::kMaxBandwidth);
    PrintBestResult(set_name, "Highest SM Warp Occupancy", Condition::kMaxOccupancy);
    PrintBestResult(set_name, "Peak Compute Throughput", Condition::kMaxGflops);
  }

  void PrintBestResult(const std::string& set_name, const std::string& label,
                       Condition condition) const {
    const auto& strategy_names = sets_.at(set_name);
    auto it = std::max_element(
        strategy_names.begin(), strategy_names.end(),
        [this, condition](const std::string& a, const std::string& b) {
          const auto& result_a = searches_.at(a).result.value();
          const auto& result_b = searches_.at(b).result.value();
          return !result_a.IsBetter(result_a.get_metrics(condition),
                                    result_b.get_metrics(condition), condition);
        });

    if (it != strategy_names.end()) {
      const auto& best_data = searches_.at(*it).result.value().get_metrics(condition);
      // Only print if metric has meaningful recorded data
      if (condition == Condition::kMaxGflops && best_data.gflops <= 0.0) return;
      std::cout << label << " champion: [" << *it << "]" << std::endl;
      Reporter::PrintResults("  ", best_data);
    }
  }

  // Cross-comparison matrix showing speedup and bandwidth ratios across evaluated strategies
  void PrintComparisonMatrix(const std::string& set_name) const {
    const auto& strategies = sets_.at(set_name);
    if (strategies.size() < 2) return;

    std::cout << "\n--- Relative Speedup Matrix (Time Ratio: Row / Column) ---" << std::endl;
    std::cout << std::left << std::setw(25) << "Strategy";
    for (const auto& s : strategies) {
      std::string short_name = s.substr(0, 12);
      std::cout << std::right << std::setw(14) << short_name;
    }
    std::cout << std::endl;

    for (const auto& row_strat : strategies) {
      double t_row = searches_.at(row_strat).result.value().get_metrics(Condition::kMinTime).time_ms;
      std::cout << std::left << std::setw(25) << row_strat.substr(0, 24);
      for (const auto& col_strat : strategies) {
        double t_col = searches_.at(col_strat).result.value().get_metrics(Condition::kMinTime).time_ms;
        if (t_row > 0.0 && t_col > 0.0) {
          // Speedup of row over column: t_col / t_row (> 1.0 means row is faster than col)
          double speedup = t_col / t_row;
          std::cout << std::right << std::setw(13) << std::fixed << std::setprecision(2) << speedup << "x";
        } else {
          std::cout << std::right << std::setw(14) << "N/A";
        }
      }
      std::cout << std::endl;
    }
    std::cout << "(Values > 1.00x indicate the row strategy is faster than the column strategy)\n" << std::endl;
  }

  std::string name_;
  std::map<std::string, Search> searches_;
  std::map<std::string, std::vector<std::string>> sets_;
};

}  // namespace warpscope
