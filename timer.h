// ==============================================================================
// GPU Hardware Timer & High-Resolution CUDA Event Profiler
// Author: Irene Therese Joseph (github.com/ireneyk)
//
// Measures GPU execution latency directly on the hardware command queue timeline
// with sub-microsecond resolution (~0.5 us), eliminating CPU host driver polling latency.
// ==============================================================================

#pragma once

#include <cuda_runtime.h>
#include <iostream>

namespace warpscope {

#define gpuErrchk(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    std::cerr << "CUDA Runtime API Error: " << cudaGetErrorString(code)
              << " in " << file << ":" << line << std::endl;
    if (abort) exit(code);
  }
}

class CudaEvent {
 public:
  CudaEvent() { gpuErrchk(cudaEventCreate(&event_)); }
  ~CudaEvent() { gpuErrchk(cudaEventDestroy(event_)); }

  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;

  CudaEvent(CudaEvent&& other) noexcept : event_(other.event_) {
    other.event_ = nullptr;
  }
  CudaEvent& operator=(CudaEvent&& other) noexcept {
    if (this != &other) {
      if (event_) cudaEventDestroy(event_);
      event_ = other.event_;
      other.event_ = nullptr;
    }
    return *this;
  }

  void Record(cudaStream_t stream = 0) {
    gpuErrchk(cudaEventRecord(event_, stream));
  }

  void Synchronize() { gpuErrchk(cudaEventSynchronize(event_)); }

  // Returns elapsed hardware time in milliseconds between two events.
  float ElapsedTime(const CudaEvent &other) const {
    float time_ms = 0.0f;
    gpuErrchk(cudaEventElapsedTime(&time_ms, event_, other.event_));
    return time_ms;
  }

 private:
  cudaEvent_t event_ = nullptr;
};

class CudaTimer {
 public:
  void Start(cudaStream_t stream = 0) { start_event_.Record(stream); }

  void Stop(cudaStream_t stream = 0) {
    stop_event_.Record(stream);
    stop_event_.Synchronize();
  }

  float ElapsedMilliseconds() const { return start_event_.ElapsedTime(stop_event_); }

 private:
  CudaEvent start_event_, stop_event_;
};

}  // namespace warpscope
