// ==============================================================================
// GPU Architecture Performance Metrics & Roofline Modeler
// Author: Irene Therese Joseph (github.com/ireneyk)
//
// Tracks execution latency, effective memory bandwidth, SM occupancy,
// computational throughput (GFLOP/s), and Roofline operational regime.
// ==============================================================================

#pragma once

#include <cstddef>
#include <array>
#include <limits>
#include <string>
#include <string_view>

namespace warpscope {

enum class Condition {
  kMinTime,
  kMaxBandwidth,
  kMaxOccupancy,
  kMaxGflops,
  kLastCondition  // Sentinel for counting conditions.
};

struct Data {
  int num_blocks = 0;
  int block_size = 0;
  double time_ms = 0.0;
  double bandwidth = 0.0;             // Effective Memory Bandwidth in GB/s
  double occupancy = 0.0;             // Theoretical SM Occupancy (0.0 - 1.0)
  double gflops = 0.0;                // Computational Throughput in GFLOP/s
  double arithmetic_intensity = 0.0;  // FLOPs per byte transferred
};

class Metrics {
 public:
  Metrics() {
    (*this)[Condition::kMinTime].time_ms = std::numeric_limits<double>::max();
    (*this)[Condition::kMaxBandwidth].bandwidth = std::numeric_limits<double>::lowest();
    (*this)[Condition::kMaxOccupancy].occupancy = std::numeric_limits<double>::lowest();
    (*this)[Condition::kMaxGflops].gflops = std::numeric_limits<double>::lowest();
  }

  // Update metrics for a given condition.
  void Update(Condition cond, const Data& new_metrics);

  // Update metrics across all tracked conditions.
  void UpdateAll(const Data& new_data);

  // Evaluates whether new_metrics improves on current for condition cond,
  // using tie-breaker heuristics across throughput and occupancy.
  bool IsBetter(const Data& new_metrics, const Data& current, Condition cond) const;

  // Retrieve metrics for a condition.
  const Data& get_metrics(Condition cond) const {
    return data[static_cast<size_t>(cond)];
  }

  // Roofline model diagnostic: Determines if kernel is Memory-Bound or Compute-Bound.
  static std::string ClassifyRoofline(double arithmetic_intensity,
                                      double peak_bandwidth_gbs,
                                      double peak_gflops);

  void print_all() const;

  Data& operator[](Condition c) { return data[static_cast<size_t>(c)]; }
  const Data& operator[](Condition c) const { return data[static_cast<size_t>(c)]; }

 private:
  std::array<Data, static_cast<size_t>(Condition::kLastCondition)> data;

  constexpr std::string_view ConditionToString(Condition cond) const;
};

}  // namespace warpscope
