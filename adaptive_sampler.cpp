// ==============================================================================
// Statistical Adaptive Sampler for GPU Microbenchmarks
// Author: Irene Therese Joseph (github.com/ireneyk)
//
// Implementations for online statistical convergence testing on GPU kernels.
// ==============================================================================

#include "adaptive_sampler.h"

#include <cstddef>
#include <cmath>
#include <utility>
#include <vector>

#include "./errors.h"
#include "tl/expected.hpp"

namespace warpscope {

// Computes the two-tailed 95% critical value of the Student's t-distribution
// for degrees of freedom df = n - 1. Uses a lookup table for df in [1, 40]
// and piecewise linear interpolation beyond 40.
ExpectedDouble AdaptiveSampler::TwoTailed95PercentStudentsT(int df) {
  if (df <= 0) {
    return tl::make_unexpected(
        ErrorInfo(ErrorInfo::kInvalidDegreesOfFreedom,
                  "Degrees of freedom must be strictly positive"));
  }

  // Exact Student's t critical values for alpha = 0.05 (df 1 through 40)
  const std::vector<double> table_values = {
      12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
      2.201,  2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
      2.080,  2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
      2.040,  2.037, 2.035, 2.032, 2.030, 2.028, 2.026, 2.024, 2.023, 2.021};

  if (df <= static_cast<int>(table_values.size())) {
    return table_values[df - 1];
  }

  // Linear interpolation anchor points for large df approaching standard normal (z = 1.960)
  const std::vector<std::pair<int, double>> interpolation_points = {
      {45, 2.014},  {50, 2.009},  {55, 2.004},  {60, 2.000},  {70, 1.994},
      {80, 1.990},  {90, 1.987},  {100, 1.984}, {150, 1.976}, {200, 1.972},
      {250, 1.969}, {300, 1.968}, {400, 1.966}, {500, 1.965}, {600, 1.964},
      {700, 1.963}, {800, 1.963}, {900, 1.963}, {1000, 1.962}};

  auto interpolate = [](int x1, double y1, int x2, double y2, int x) {
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1);
  };

  for (size_t i = 0; i < interpolation_points.size() - 1; ++i) {
    if (df <= interpolation_points[i + 1].first) {
      return interpolate(interpolation_points[i].first,
                         interpolation_points[i].second,
                         interpolation_points[i + 1].first,
                         interpolation_points[i + 1].second, df);
    }
  }
  return interpolation_points.back().second;
}

void AdaptiveSampler::Update(double x) {
  alpha_ += x;
  num_samples_++;
}

ExpectedDouble AdaptiveSampler::EstimatedMean() {
  if (0 == num_samples_) {
    return tl::make_unexpected(
        ErrorInfo(ErrorInfo::kTooFewSamples,
                  "Sample count must be greater than zero to compute mean"));
  }
  return alpha_ / static_cast<double>(num_samples_);
}

// Determines if additional kernel launches are required.
// Evaluates whether the 95% confidence margin of error (MoE) on the sample mean
// has decayed below relative_precision_ * sample standard deviation.
bool AdaptiveSampler::ShouldContinue() {
  // Need at least 2 samples to compute sample variance
  if (num_samples_ <= 1) {
    return true;
  }

  double n = static_cast<double>(num_samples_);
  double variance = alpha_ / (n * n * (n - 1.0));

  // Standard error of the mean = sqrt(variance / n)
  double std_error = std::sqrt(variance / n);
  auto t_result = TwoTailed95PercentStudentsT(num_samples_ - 1);
  if (!t_result) {
    return true;
  }

  double std_dev = std::sqrt(variance);
  double margin_of_error = *t_result * std_error;
  double required_margin_of_error = relative_precision_ * std_dev;

  // Margin of error decays proportionally to 1/sqrt(n); continue until threshold met
  return margin_of_error > required_margin_of_error;
}

}  // namespace warpscope
