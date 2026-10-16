# WarpScope: GPU Architecture Profiler & Kernel Optimization Engine

**Author:** [Irene Therese Joseph](https://github.com/ireneyk)  
**Contact:** [ireneeetherese@gmail.com](mailto:ireneeetherese@gmail.com) | [LinkedIn](https://linkedin.com/in/irenetjoseph) | [GitHub](https://github.com/ireneyk)

A C++ and CUDA performance modeling framework for profiling Streaming Multiprocessor (SM) resource allocation, memory hierarchy bottlenecks, and kernel execution dynamics.

---

## Why I Built This

Writing functionally correct CUDA kernels is relatively straightforward, but getting them to actually saturate hardware capabilities is an entirely different challenge. How a kernel accesses memory, how thread blocks are dimensioned, and how many registers each thread consumes directly dictate whether the Streaming Multiprocessors (SMs) spend their cycles executing math or stalling on memory pipelines.

When I started profiling common parallel patterns, I noticed that conventional developer intuitions often fail:
- Grid-stride loops are widely recommended for flexibility, yet on fixed problem sizes, 1:1 mapped grids frequently achieve higher sustained bandwidth by avoiding loop counter registers and stride branch evaluations.
- Unified Memory is convenient, but page migration faults over the PCIe bus can completely overshadow raw kernel execution time on smaller problem sets.
- Maximizing occupancy to 100% does not automatically guarantee maximum speedup if instruction-level parallelism (ILP) or memory transaction coalescing is neglected.

I built this framework to systematically benchmark, model, and explain these microarchitectural trade-offs with empirical data.

---

## What the Framework Explores

```
+-----------------------------------------------------------------------------------+
|                             GPU Architecture                                      |
+-----------------------------------------------------------------------------------+
|  Host CPU / Host Memory                                                           |
|       |                                                                           |
|       |  PCIe Gen3 / Gen4 / Gen5 Bus Interconnect (16 - 64 GB/s)                  |
|       v                                                                           |
|  Device Memory (HBM / GDDR: 500 - 3,350 GB/s)                                     |
|       |                                                                           |
|       |  L2 Cache Crossbar Interconnect (32 - 128 Byte Sectors)                   |
|       v                                                                           |
|  Streaming Multiprocessors (SMs)                                                  |
|  +-----------------------------------------------------------------------------+  |
|  |  Warp Schedulers & Dispatch Units (4 per SM)                                |  |
|  |  Register File (64K 32-bit registers per SM)                                |  |
|  |  L1 Data Cache / Shared Memory SRAM (64 KB - 228 KB per SM)                 |  |
|  |  ALUs / FP32 Units / Tensor Cores                                           |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

### 1. Memory Subsystem & Warp Coalescing
GPU global memory is serviced in 32-byte or 128-byte cache line transactions. When the 32 threads in a warp access contiguous 4-byte memory addresses, the memory controller coalesces the entire warp request into a minimal number of bus transactions.
- **Coalesced 1:1 Mapping:** Consecutive threads access contiguous array indices, achieving peak memory bus utilization.
- **Grid-Stride Loops:** Evaluates loop reuse across fixed-size grids against pure 1:1 mapped blocks.
- **Strided Access Penalty:** Demonstrates how non-contiguous striding fragments warp requests into up to 32 separate transactions per instruction, saturating the memory bus while transferring mostly wasted bytes.

### 2. Interconnect & Memory Management: Unified Memory vs Explicit PCIe Transfers
- **Unified Memory (`cudaMallocManaged`):** Page migration engine brings pages to the device on demand. I wanted to quantify the exact boundary where on-demand page fault overheads drop below the upfront latency overhead of explicit buffer setup.
- **Explicit Device Memory (`cudaMalloc` + `cudaMemcpy`):** Pre-allocates buffers on device DRAM and pipelines transfers over the PCIe interconnect.
- The framework measures both isolated kernel compute latency and end-to-end data movement over the bus, highlighting how PCIe bandwidth caps throughput on streaming workloads.

### 3. Compute Architecture: Breaking the GEMM Memory Wall
General Matrix Multiplication (GEMM, $C = A \cdot B$) is the backbone of deep learning and numerical computing.
- **Naive Global DRAM GEMM:** Every thread computes one output element by independently streaming an entire row of $A$ and column of $B$ directly from global memory. For an $N \times N$ matrix, this requires $2N^3$ global memory reads, producing an arithmetic intensity of just $0.25$ FLOPs/byte. The kernel is completely memory-bandwidth bound.
- **Shared-Memory Tiled GEMM:** Cooperative thread blocks stage $16 \times 16$ sub-matrix tiles into fast on-chip SRAM (Shared Memory). All threads synchronize via `__syncthreads()`, compute partial dot products from on-chip memory, and repeat across tiles. Global memory traffic is reduced by a factor of $16\times$, elevating arithmetic intensity to $\sim 4.0$ FLOPs/byte and moving the execution profile up the Roofline curve toward compute saturation.

### 4. SM Resource Partitioning & Occupancy Bottlenecks
Theoretical occupancy is the ratio of active resident warps to the maximum possible warps an SM can support. 

I wrote an analytical diagnostic engine (`DiagnoseOccupancy`) in [`occupancy.cu`](src/occupancy.cu) that inspects hardware device properties and kernel attributes to report exactly what restricts occupancy:
1. **Thread/Warp Limit:** Block size does not provide enough warps to populate all warp schedulers.
2. **Register Pressure:** High register allocation per thread exceeds the SM's 64K register file budget, capping the number of active resident blocks.
3. **Shared Memory Capacity:** Per-block shared memory allocations exceed SM SRAM capacity.
4. **Warp Alignment:** Block sizes that are not multiples of 32 waste scheduling slots in the dispatch pipeline.

### 5. Statistical Adaptive Sampling (No More Arbitrary Run Counts)
Benchmarking kernels with fixed iteration counts (e.g. "run 10 times and average") is unreliable:
- Dynamic core clock boosting (DVFS) shifts execution speed across iterations as thermals fluctuate.
- Initial cold-cache runs distort sample means.
- PCIe bus arbitration introduces run-to-run timing jitter.

To solve this, I implemented an `AdaptiveSampler` that discards warmup runs and continuously updates sample variance. It stops running iterations once the margin of error at a **95% confidence interval** (derived from the Student's t-distribution for $n-1$ degrees of freedom) falls below a strict fraction ($35\%$) of the sample standard deviation.

---

## Repository Architecture

```
warpscope/
|-- CMakeLists.txt                    # Top-level build configuration (CUDA Standard 17)
|-- LICENSE-MIT                       # MIT License (Irene Therese Joseph)
|-- README.md                         # Architecture documentation & benchmarks
|-- scripts/
|   \-- warpscope_analyzer.py         # Python Roofline model & SM occupancy profiler
|-- src/
|   |-- CMakeLists.txt                # Core library and executable build rules
|   |-- warpscope.cu             # Main hardware profiling suite & device discovery
|   |-- optimizer.h                   # Strategy manager & relative speedup matrix engine
|   |-- i_kernel.h                    # Abstract workload interface & FLOP metadata
|   |-- kernels.h                     # CUDA kernel signatures & device helper functions
|   |-- kernels.cu                    # Kernel implementations (Vector Add, Distance, GEMM, Tiled)
|   |-- grid_searchers.h              # Parameter sweep iterators & execution harnesses
|   |-- generators.h                  # Block-size (warp-multiple) and grid-size generators
|   |-- occupancy.h                   # SM resource limiters & theoretical occupancy modeler
|   |-- occupancy.cu                  # Hardware bottleneck diagnostic implementation
|   |-- metrics.h                     # Data structures for latency, BW, GFLOP/s & Roofline
|   |-- metrics.cpp                   # Multi-condition optimization & comparison logic
|   |-- adaptive_sampler.h            # Online statistical sampling engine header
|   |-- adaptive_sampler.cpp          # Student's t-distribution online convergence engine
|   |-- reporter.h                    # SI unit scaling & terminal telemetry formatter
|   |-- reporter.cpp                  # Formatted terminal output implementation
|   |-- timer.h                       # Sub-microsecond CUDA event hardware timer
|   |-- errors.h                      # Error handling types (tl::expected monad)
|   \-- examples/
|       |-- add_strided_managed.h / .cu       # Grid-stride loop with Unified Memory
|       |-- add_strided_unmanaged.h / .cu     # Grid-stride loop with explicit PCIe copies
|       |-- add_unstrided_managed.h / .cu     # 1:1 coalesced loop with Unified Memory
|       |-- add_unstrided_unmanaged.h / .cu   # 1:1 coalesced loop with explicit PCIe copies
|       |-- euclidean_distance_strided.h / .cu    # 64-bit vector load stride benchmark
|       |-- euclidean_distance_unstrided.h / .cu  # 64-bit vector load coalesced benchmark
|       |-- matrix_multiply.h / .cu               # Naive global memory GEMM
|       \-- matrix_multiply_tiled.h / .cu         # Shared-memory tiled GEMM
\-- tests/
    |-- CMakeLists.txt                # Catch2 test suite build definition
    |-- test_adaptive_sampler.cpp     # Unit tests for Student's t statistical engine
    |-- test_generators.cpp           # Unit tests for parameter grid search generators
    |-- test_metrics.cpp              # Unit tests for metric tracking & condition updates
    |-- test_occupancy.cpp            # Unit tests for SM resource bottleneck diagnostics
    \-- test_reporter.cpp             # Unit tests for formatting and SI unit scaling
```

---

## Build and Execution Guide

### Prerequisites
- **CUDA Toolkit:** 11.8 or later
- **CMake:** Version 3.18 or higher
- **C++ Compiler:** Supporting C++17 (`g++`, `clang++`, or MSVC)
- **Python:** Version 3.8+ (for architectural analysis tools)

### 1. Building the Project

```bash
# Clone the repository
git clone https://github.com/ireneyk/warpscope.git
cd warpscope

# Configure build directory with CMake
mkdir build
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release

# Compile all libraries, benchmarks, and test binaries
cmake --build build --config Release
```

### 2. Running Unit & Integration Tests

The test suite validates statistical convergence, metric tie-breakers, generator sequences, and SM resource diagnosis:

```bash
# Execute Catch2 test executable
./build/tests/warpscope_tests
```

Output:
```text
Randomness seeded to: 1729165243
===============================================================================
All tests passed (38 assertions in 12 test cases)
```

### 3. Executing the Hardware Benchmark Profiler

Run the main profiling suite to execute the architectural sweeps and generate comparison matrices:

```bash
./build/src/warpscope
```

To capture results to a log file for post-processing:

```bash
./build/src/warpscope | tee profile_run.log
```

### 4. Running the Python Roofline & Architecture Analyzer

I also wrote a standalone Python tool to model theoretical Roofline limits and SM resource allocation curves across hardware architectures:

```bash
# Run Roofline and SM occupancy modeling for RTX 4090
python scripts/warpscope_analyzer.py --arch RTX-4090

# Or evaluate against an A100 Tensor Core configuration
python scripts/warpscope_analyzer.py --arch A100-SXM4-40GB

# Parse captured benchmark log output
python scripts/warpscope_analyzer.py --log profile_run.log
```

---

## Experimental Observations & Takeaways

### 1. Vector Addition: Coalesced 1:1 Mapping vs Grid-Stride Loop
- **Observation:** In vector addition ($N = 1,048,576$ elements), the unstrided 1:1 thread-to-element mapping with 256 threads per block consistently outpaced the strided loop variation.
- **Hardware Rationale:** In a 1:1 mapping, contiguous threads within each warp access strictly contiguous memory addresses. The memory controller consolidates 32 4-byte accesses into exactly one 128-byte cache line transaction. In contrast, while grid-stride loops offer flexibility when handling dynamic problem sizes with fixed grid allocations, the loop indexing and stride branching logic introduce slight register overhead and branch execution cycles without improving cache line reuse on simple streaming arithmetic.

### 2. Memory Subsystem: Unified Memory Demand Paging vs Explicit Host-Device Transfers
- **Observation:** Unmanaged explicit memory transfers demonstrated lower kernel-only compute time, but including `cudaMemcpy` transfers over the PCIe link substantially increased end-to-end execution latency.
- **Hardware Rationale:** Unified Memory on modern architectures leverages the GPU Page Migration Engine. On initial kernel touch, pages fault and migrate across the PCIe bus in 4 KB or 2 MB granularity. For small or single-pass streaming workloads, page fault handling introduces observable latency. However, for iterative workloads where memory remains resident on the device after the first touch, Unified Memory achieves near-native device DRAM bandwidth.

### 3. Compute Architecture: The Memory Wall in Matrix Multiplication
- **Observation:** Staging matrix tiles through shared memory (`MatrixMultiplySharedTiled`) delivered over **$8\times$ to $12\times$ higher throughput** than the naive implementation (`MatrixMultiplyNaive`).
- **Hardware Rationale:** In naive GEMM, each thread reads $N$ values from matrix $A$ and $N$ values from matrix $B$ directly from global DRAM. Because neighboring threads in a 1D grid read different rows or columns simultaneously, memory accesses cannot be fully coalesced, leading to excessive L2 cache line evictions and DRAM bus saturation. In the tiled version, each 16x16 tile is cooperatively loaded into fast on-chip shared memory (SRAM) once per thread block, and reused 16 times across the threads in that block. This effectively bypasses the DRAM memory wall and dramatically increases floating-point pipeline saturation.

### 4. Speedup Summary Matrix (Generated by Optimizer Engine)

```text
======================================================================
 Architectural Optimization Set: Vector Add (Loop Structure x Memory Subsystem)
======================================================================
Strategies evaluated:
   * Strided, Managed
   * Unstrided, Managed
   * Strided, Explicit-PCIe
   * Unstrided, Explicit-PCIe
----------------------------------------------------------------------
Lowest Latency champion            : [Unstrided, Managed]
Peak Effective Bandwidth champion  : [Unstrided, Managed]
Highest SM Warp Occupancy champion : [Unstrided, Managed]

--- Relative Speedup Matrix (Time Ratio: Row / Column) ---
Strategy                     Strided, Man    Unstrided, M    Strided, Exp    Unstrided, E
Strided, Managed                    1.00x           0.46x           2.15x           2.08x
Unstrided, Managed                  2.16x           1.00x           4.65x           4.49x
Strided, Explicit-PCIe              0.47x           0.22x           1.00x           0.97x
Unstrided, Explicit-PCIe            0.48x           0.22x           1.03x           1.00x

(Values > 1.00x indicate the row strategy is faster than the column strategy)
```

---

## Key Hardware Architecture & Systems Engineering Highlights

| Core Focus Area | Hardware Architecture & Systems Engineering Topics Covered |
|---|---|
| **GPU Microarchitecture** | Microarchitectural performance modeling; SM resource partitioning (register budgeting, shared memory sizing); warp residency & occupancy analysis; memory subsystem benchmarking (coalescing, cache-line sector transactions, DRAM bandwidth bounds). |
| **Parallel Kernel Optimization** | CUDA C/C++ parallel kernel optimization; identifying computation and memory access bottlenecks; shared-memory GEMM tiling; Roofline performance evaluation; high-precision profiling using hardware CUDA events. |
| **Compute Systems Modeling** | Workload characterization for AI/GEMM compute platforms; empirical vs theoretical roofline modeling; developing profiling and analysis tooling in both C++ and Python; architectural speedup evaluation. |
| **ASIC / Microbenchmarking Verification** | Functional verification testbenches with Catch2; boundary condition checks across block/grid dimensions; PCIe interconnect throughput and data transfer modeling; structured modular C++ architecture. |

---

## License

Distributed under the MIT License. See [LICENSE-MIT](LICENSE-MIT) for details.

Copyright (c) 2026 **Irene Therese Joseph**
