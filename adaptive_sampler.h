// ==============================================================================
// Statistical Adaptive Sampler for GPU Microbenchmarks
// Author: Irene Therese Joseph (github.com/ireneyk)
//
// Dynamically adjusts sample count using Student's t-distribution to achieve
// a 95% confidence interval, filtering out DVFS clock frequency shifts,
// cache line cold-start effects, and thermal throttling noise.
// ==============================================================================

#pragma once

#include "./errors.h"

namespace warpscope {

class AdaptiveSampler {
  friend class AdaptiveSamplerTest;

 private:
  double alpha_ = 0.0;
  int num_samples_ = 0;
  double relative_precision_;

  ExpectedDouble TwoTailed95PercentStudentsT(int df);

 public:
  explicit AdaptiveSampler(double rp = 0.30) : relative_precision_(rp) {}

  void Update(double x);
  bool ShouldContinue();
  ExpectedDouble EstimatedMean();
  int NumSamples() const { return num_samples_; }
};

}  // namespace warpscope
