# ==============================================================================
# GPU Architecture Microbenchmark Unit & Integration Tests
# Author: Irene Therese Joseph (github.com/ireneyk)
# ==============================================================================

include(FetchContent)

FetchContent_Declare(
  catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.7.0)
FetchContent_MakeAvailable(catch2)
list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/contrib")

set(TEST_FILES
    test_adaptive_sampler.cpp
    test_generators.cpp
    test_metrics.cpp
    test_occupancy.cpp
    test_reporter.cpp)

add_executable(warpscope_tests ${TEST_FILES})
target_link_libraries(warpscope_tests PRIVATE Catch2::Catch2WithMain core_lib)

include(CTest)
include(Catch)
catch_discover_tests(warpscope_tests)
