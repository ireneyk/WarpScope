// ==============================================================================
// Unit Tests: Grid & Block Size Search Space Generators
// Author: Irene Therese Joseph (github.com/ireneyk)
// ==============================================================================

#include <catch2/catch_test_macros.hpp>
#include <optional>

#include "../src/generators.h"

namespace warpscope {

TEST_CASE("DoublingGenerator with max_num_blocks of 0", "[DoublingGenerator]") {
  DoublingGenerator generator(0);

  SECTION("Immediately returns std::nullopt") {
    REQUIRE(generator.Next() == std::nullopt);
  }
}

TEST_CASE("DoublingGenerator with max_num_blocks of 1", "[DoublingGenerator]") {
  DoublingGenerator generator(1);

  SECTION("Generates only one value") {
    REQUIRE(generator.Next() == 1);
    REQUIRE(generator.Next() == std::nullopt);
  }
}

TEST_CASE("DoublingGenerator generates correct sequence",
          "[DoublingGenerator]") {
  DoublingGenerator generator(8);

  SECTION("Generates correct sequence") {
    REQUIRE(generator.Next() == 1);
    REQUIRE(generator.Next() == 2);
    REQUIRE(generator.Next() == 4);
    REQUIRE(generator.Next() == 8);
    REQUIRE(generator.Next() == std::nullopt);
  }
}

TEST_CASE("DoublingGenerator with max_num_blocks not a power of 2",
          "[DoublingGenerator]") {
  DoublingGenerator generator(6);

  SECTION("Generates correct sequence and stops at last valid value") {
    REQUIRE(generator.Next() == 1);
    REQUIRE(generator.Next() == 2);
    REQUIRE(generator.Next() == 4);
    REQUIRE(generator.Next() == std::nullopt);
  }
}

TEST_CASE("DoublingGenerator multiple calls after reaching max",
          "[DoublingGenerator]") {
  DoublingGenerator generator(4);

  SECTION("Consistently returns std::nullopt after reaching max") {
    REQUIRE(generator.Next() == 1);
    REQUIRE(generator.Next() == 2);
    REQUIRE(generator.Next() == 4);
    REQUIRE(generator.Next() == std::nullopt);
    REQUIRE(generator.Next() == std::nullopt);
    REQUIRE(generator.Next() == std::nullopt);
  }
}

TEST_CASE("IncrementBy32Generator with max_block_size of 0",
          "[IncrementBy32Generator]") {
  IncrementBy32Generator generator(0);

  SECTION("Immediately returns std::nullopt") {
    REQUIRE(generator.Next() == std::nullopt);
  }
}

TEST_CASE("IncrementBy32Generator generates correct sequence",
          "[IncrementBy32Generator]") {
  IncrementBy32Generator generator(100);

  SECTION("Generates correct sequence") {
    REQUIRE(generator.Next() == 1);
    REQUIRE(generator.Next() == 32);
    REQUIRE(generator.Next() == 64);
    REQUIRE(generator.Next() == 96);
    REQUIRE(generator.Next() == std::nullopt);
  }
}

TEST_CASE("IncrementBy32Generator with max_block_size less than 32",
          "[IncrementBy32Generator]") {
  IncrementBy32Generator generator(30);

  SECTION("Generates only the initial value") {
    REQUIRE(generator.Next() == 1);
    REQUIRE(generator.Next() == std::nullopt);
  }
}

TEST_CASE("IncrementBy32Generator with max_block_size not a multiple of 32",
          "[IncrementBy32Generator]") {
  IncrementBy32Generator generator(50);

  SECTION("Generates correct sequence and stops at last valid value") {
    REQUIRE(generator.Next() == 1);
    REQUIRE(generator.Next() == 32);
    REQUIRE(generator.Next() == std::nullopt);
  }
}

TEST_CASE("IncrementBy32Generator multiple calls after reaching max",
          "[IncrementBy32Generator]") {
  IncrementBy32Generator generator(64);

  SECTION("Consistently returns std::nullopt after reaching max") {
    REQUIRE(generator.Next() == 1);
    REQUIRE(generator.Next() == 32);
    REQUIRE(generator.Next() == 64);
    REQUIRE(generator.Next() == std::nullopt);
    REQUIRE(generator.Next() == std::nullopt);
    REQUIRE(generator.Next() == std::nullopt);
  }
}

}  // namespace warpscope
