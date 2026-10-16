// ==============================================================================
// GPU Architecture Performance Metrics & Roofline Modeler
// Author: Irene Therese Joseph (github.com/ireneyk)
//
// Implementations for multi-objective hardware performance tracking.
// ==============================================================================

#include "metrics.h"

#include <iomanip>
#include <iostream>

namespace warpscope {

void Metrics::Update(Condition cond, const Data& new_metrics) {
  auto& current = data[static_cast<size_t>(cond)];
  if (IsBetter(new_metrics, current, cond)) {
    current = new_metrics;
  }
}

void Metrics::UpdateAll(const Data& new_data) {
  for (size_t i = 0; i < static_cast<size_t>(Condition::kLastCondition); ++i) {
    Update(static_cast<Condition>(i), new_data);
  }
}

bool Metrics::IsBetter(const Data& new_metrics, const Data& current,
                       Condition cond) const {
  switch (cond) {
    case Condition::kMinTime:
      if (new_metrics.time_ms < current.time_ms) return true;
      if (new_metrics.time_ms > current.time_ms) return false;
      return new_metrics.bandwidth > current.bandwidth ||
             (new_metrics.bandwidth == current.bandwidth &&
              new_metrics.occupancy > current.occupancy);

    case Condition::kMaxBandwidth:
      if (new_metrics.bandwidth > current.bandwidth) return true;
      if (new_metrics.bandwidth < current.bandwidth) return false;
      return new_metrics.time_ms < current.time_ms ||
             (new_metrics.time_ms == current.time_ms &&
              new_metrics.occupancy > current.occupancy);

    case Condition::kMaxOccupancy:
      if (new_metrics.occupancy > current.occupancy) return true;
      if (new_metrics.occupancy < current.occupancy) return false;
      return new_metrics.bandwidth > current.bandwidth ||
             (new_metrics.time_ms < current.time_ms &&
              new_metrics.bandwidth == current.bandwidth);

    case Condition::kMaxGflops:
      if (new_metrics.gflops > current.gflops) return true;
      if (new_metrics.gflops < current.gflops) return false;
      return new_metrics.time_ms < current.time_ms;

    default:
      return false;
  }
}

std::string Metrics::ClassifyRoofline(double arithmetic_intensity,
                                      double peak_bandwidth_gbs,
                                      double peak_gflops) {
  if (peak_bandwidth_gbs <= 0.0 || peak_gflops <= 0.0) {
    return "Unknown (Hardware limits uncalibrated)";
  }
  // The ridge point is the balance between compute throughput and memory bandwidth
  double ridge_point = peak_gflops / peak_bandwidth_gbs;
  if (arithmetic_intensity < ridge_point) {
    return "Memory Bandwidth Bound (Latency/DRAM Bus Bottleneck)";
  } else {
    return "Compute Bound (ALU / Tensor Core Pipeline Bottleneck)";
  }
}

constexpr std::string_view Metrics::ConditionToString(Condition cond) const {
  switch (cond) {
    case Condition::kMinTime:
      return "Minimum Latency";
    case Condition::kMaxBandwidth:
      return "Maximum Memory Bandwidth";
    case Condition::kMaxOccupancy:
      return "Maximum SM Warp Occupancy";
    case Condition::kMaxGflops:
      return "Maximum Compute Throughput (GFLOP/s)";
    default:
      return "Unknown Metric";
  }
}

void Metrics::print_all() const {
  for (size_t i = 0; i < static_cast<size_t>(Condition::kLastCondition); ++i) {
    Condition cond = static_cast<Condition>(i);
    const auto& m = data[i];

    std::cout << "Metric Target: " << ConditionToString(cond) << "\n"
              << "  Grid Config: <<" << m.num_blocks << ", " << m.block_size << ">>\n"
              << "  Latency    : " << m.time_ms << " ms\n"
              << "  Bandwidth  : " << m.bandwidth << " GB/s\n"
              << "  Occupancy  : " << std::fixed << std::setprecision(2) << (m.occupancy * 100.0) << "%\n";
    if (m.gflops > 0.0) {
      std::cout << "  Throughput : " << m.gflops << " GFLOP/s\n";
    }
    if (m.arithmetic_intensity > 0.0) {
      std::cout << "  Intensity  : " << m.arithmetic_intensity << " FLOPs/byte\n";
    }
    std::cout << "\n";
  }
}

}  // namespace warpscope