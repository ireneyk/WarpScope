// ==============================================================================
// GPU Architecture Grid & Block Size Search Space Generators
// Author: Irene Therese Joseph (github.com/ireneyk)
//
// Generates systematic exploration steps for GPU microarchitecture parameter tuning:
// - Warp-sized increments (multiples of 32 threads)
// - Power-of-two grid sizing (SM saturation curves)
// ==============================================================================

#pragma once

#include <optional>

namespace warpscope {

// Base interface for grid parameter iterators
class IGridSizeGenerator {
 public:
  virtual ~IGridSizeGenerator() = default;
  virtual std::optional<int> Next() = 0;
};

// Generates block counts by powers of 2 (1, 2, 4, 8, ... up to max_num_blocks)
class DoublingGenerator : public IGridSizeGenerator {
 public:
  explicit DoublingGenerator(int mnb) : max_num_blocks_(mnb) {}

  std::optional<int> Next() override {
    if (current_ > max_num_blocks_) {
      return std::nullopt;
    }
    int result = current_;
    current_ *= 2;
    return result;
  }

 private:
  int current_ = 1;
  const int max_num_blocks_;
};

// Generates block sizes in warp multiples (32, 64, 96, 128, ... up to max_block_size)
// to prevent warp fragmentation on modern Streaming Multiprocessors.
class IncrementBy32Generator : public IGridSizeGenerator {
 public:
  explicit IncrementBy32Generator(int mbs) : max_block_size_(mbs) {}

  std::optional<int> Next() override {
    if (block_size_ > max_block_size_) {
      return std::nullopt;
    }
    int result = block_size_;
    i_++;
    block_size_ = 32 * i_;
    return result;
  }

 private:
  int i_ = 0;
  int block_size_ = 1;
  const int max_block_size_;
};

}  // namespace warpscope
