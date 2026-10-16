// ==============================================================================
// Unit Tests: GPU Architecture Reporter & Numeric Formatting
// Author: Irene Therese Joseph (github.com/ireneyk)
// ==============================================================================

#include <catch2/catch_test_macros.hpp>
#include <string>

#include "../src/reporter.h"

namespace warpscope {

TEST_CASE("FormatToSI", "[Reporter]") {
  REQUIRE(Reporter::FormatToSI(-22) == "-22.00");
  REQUIRE(Reporter::FormatToSI(0) == "0.00");
  REQUIRE(Reporter::FormatToSI(-6e-6) == "-0.00");
  REQUIRE(Reporter::FormatToSI(0.9e-2) == "0.01");
  REQUIRE(Reporter::FormatToSI(1e-2) == "0.01");
  REQUIRE(Reporter::FormatToSI(22) == "22.00");
  REQUIRE(Reporter::FormatToSI(3e3) == "3.00K");
  REQUIRE(Reporter::FormatToSI(2e6) == "2.00M");
  REQUIRE(Reporter::FormatToSI(6.235e9) == "6.24G");
  REQUIRE(Reporter::FormatToSI(6.235e10) == "62.35G");
  REQUIRE(Reporter::FormatToSI(6.235e11) == "623.50G");
  REQUIRE(Reporter::FormatToSI(6.235e12) == "6.24T");
  REQUIRE(Reporter::FormatToSI(1.2e13) == "12.00T");
  REQUIRE(Reporter::FormatToSI(1.5e15) == "1.50P");
  REQUIRE(Reporter::FormatToSI(1.8e18) == "1.80E");
  REQUIRE(Reporter::FormatToSI(2.1e21) == "2.10Z");
  REQUIRE(Reporter::FormatToSI(2.4e24) == "2.40Y");
  REQUIRE(Reporter::FormatToSI(2.8e27) == "2800.00Y");
}

TEST_CASE("FormatWithCommas", "[Reporter]") {
  REQUIRE(Reporter::FormatWithCommas(0) == "0");
  REQUIRE(Reporter::FormatWithCommas(-0) == "0");
  REQUIRE(Reporter::FormatWithCommas(0) == "0");
  REQUIRE(Reporter::FormatWithCommas(00) == "0");
  REQUIRE(Reporter::FormatWithCommas(1) == "1");
  REQUIRE(Reporter::FormatWithCommas(-1) == "-1");
  REQUIRE(Reporter::FormatWithCommas(22) == "22");
  REQUIRE(Reporter::FormatWithCommas(333) == "333");
  REQUIRE(Reporter::FormatWithCommas(4444) == "4,444");
  REQUIRE(Reporter::FormatWithCommas(-4444) == "-4,444");
  REQUIRE(Reporter::FormatWithCommas(55555) == "55,555");
  REQUIRE(Reporter::FormatWithCommas(666666) == "666,666");
  REQUIRE(Reporter::FormatWithCommas(7777777) == "7,777,777");
  REQUIRE(Reporter::FormatWithCommas(88888888) == "88,888,888");
  REQUIRE(Reporter::FormatWithCommas(999999999) == "999,999,999");
  REQUIRE(Reporter::FormatWithCommas(1111111111) == "1,111,111,111");
  REQUIRE(Reporter::FormatWithCommas(-1111111111) == "-1,111,111,111");
}

}  // namespace warpscope
