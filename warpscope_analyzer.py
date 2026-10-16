#!/usr/bin/env python3
"""
==============================================================================
GPU Hardware Architecture Profiler & Roofline Analysis Tool
Author: Irene Therese Joseph (github.com/ireneyk)

Performs microarchitectural performance analysis for CUDA workloads:
- Roofline operational regime classification (Memory Bound vs Compute Bound)
- Memory Subsystem efficiency (Achieved vs Theoretical Peak DRAM Bandwidth)
- Warp Occupancy & SM Resource Limiter modeling (Registers vs Shared Memory)
- Speedup matrix parsing and cross-comparison reporting
==============================================================================
"""

import argparse
import math
import os
import re
import sys
from typing import Dict, List, Optional, Tuple


class GPUArchModel:
    """Analytical hardware model of GPU Streaming Multiprocessors."""

    KNOWN_ARCHITECTURES = {
        "A100-SXM4-40GB": {
            "compute_capability": (8, 0),
            "sms": 108,
            "peak_fp32_tflops": 19.5,
            "peak_dram_bw_gbs": 1555.0,
            "l2_cache_mb": 40.0,
            "max_warps_per_sm": 64,
            "regs_per_sm": 65536,
            "shmem_per_sm_kb": 164,
        },
        "H100-SXM5-80GB": {
            "compute_capability": (9, 0),
            "sms": 132,
            "peak_fp32_tflops": 67.0,
            "peak_dram_bw_gbs": 3350.0,
            "l2_cache_mb": 50.0,
            "max_warps_per_sm": 64,
            "regs_per_sm": 65536,
            "shmem_per_sm_kb": 228,
        },
        "RTX-4090": {
            "compute_capability": (8, 9),
            "sms": 128,
            "peak_fp32_tflops": 82.6,
            "peak_dram_bw_gbs": 1008.0,
            "l2_cache_mb": 72.0,
            "max_warps_per_sm": 48,
            "regs_per_sm": 65536,
            "shmem_per_sm_kb": 100,
        },
        "RTX-3080": {
            "compute_capability": (8, 6),
            "sms": 68,
            "peak_fp32_tflops": 29.8,
            "peak_dram_bw_gbs": 760.0,
            "l2_cache_mb": 5.0,
            "max_warps_per_sm": 48,
            "regs_per_sm": 65536,
            "shmem_per_sm_kb": 100,
        },
    }

    def __init__(self, name: str = "RTX-4090"):
        specs = self.KNOWN_ARCHITECTURES.get(name, self.KNOWN_ARCHITECTURES["RTX-4090"])
        self.name = name
        self.compute_cap = specs["compute_capability"]
        self.sms = specs["sms"]
        self.peak_fp32_tflops = specs["peak_fp32_tflops"]
        self.peak_dram_bw_gbs = specs["peak_dram_bw_gbs"]
        self.l2_cache_mb = specs["l2_cache_mb"]
        self.max_warps_per_sm = specs["max_warps_per_sm"]
        self.regs_per_sm = specs["regs_per_sm"]
        self.shmem_per_sm_kb = specs["shmem_per_sm_kb"]

        # Roofline balance point (Ridge Point in FLOPs/byte)
        # Ridge Point = Peak GFLOP/s / Peak DRAM GB/s
        self.ridge_point = (self.peak_fp32_tflops * 1000.0) / self.peak_dram_bw_gbs

    def evaluate_roofline(self, arithmetic_intensity: float) -> Tuple[float, str]:
        """Calculates theoretical attainable performance (GFLOP/s) and operational regime."""
        mem_bound_perf = arithmetic_intensity * self.peak_dram_bw_gbs
        peak_perf = self.peak_fp32_tflops * 1000.0
        attainable_perf = min(mem_bound_perf, peak_perf)

        if arithmetic_intensity < self.ridge_point:
            regime = "Memory Bandwidth Bound (DRAM / Interconnect Bottleneck)"
        else:
            regime = "Compute Bound (Execution Units / Tensor Core Saturation)"
        return attainable_perf, regime

    def diagnose_occupancy(
        self,
        block_size: int,
        regs_per_thread: int = 32,
        shmem_bytes_per_block: int = 0,
    ) -> Dict[str, float]:
        """Models SM active warp residency and resource allocation constraints."""
        warp_size = 32
        warps_per_block = math.ceil(block_size / warp_size)
        max_threads_per_sm = self.max_warps_per_sm * warp_size

        # 1. Thread count constraint
        limit_threads = max_threads_per_sm // block_size

        # 2. Maximum resident blocks per SM (Ampere / Ada / Hopper support 32 blocks/SM)
        limit_blocks = 32

        # 3. Register file constraint
        reg_granularity = 4
        aligned_regs = math.ceil(regs_per_thread / reg_granularity) * reg_granularity
        regs_per_warp = math.ceil((aligned_regs * warp_size) / 256) * 256
        regs_per_block = regs_per_warp * warps_per_block
        limit_regs = self.regs_per_sm // regs_per_block if regs_per_block > 0 else limit_blocks

        # 4. Shared memory constraint
        shmem_capacity = self.shmem_per_sm_kb * 1024
        limit_shmem = (
            shmem_capacity // shmem_bytes_per_block if shmem_bytes_per_block > 0 else limit_blocks
        )

        active_blocks = max(0, min(limit_threads, limit_blocks, limit_regs, limit_shmem))
        active_warps = active_blocks * warps_per_block
        occupancy = active_warps / self.max_warps_per_sm

        limiter = "Thread/Warp Limit"
        if active_blocks == limit_regs and limit_regs < limit_threads and regs_per_thread > 0:
            limiter = f"Register Pressure ({regs_per_thread} regs/thread)"
        elif active_blocks == limit_shmem and limit_shmem < limit_threads and shmem_bytes_per_block > 0:
            limiter = f"Shared Memory Allocation ({shmem_bytes_per_block} bytes/block)"
        elif active_blocks == limit_blocks and limit_blocks < limit_threads:
            limiter = "Max Resident Blocks/SM Limit (32 blocks)"

        return {
            "active_blocks": active_blocks,
            "active_warps": active_warps,
            "max_warps": self.max_warps_per_sm,
            "theoretical_occupancy": min(1.0, occupancy),
            "primary_limiter": limiter,
        }


def print_ascii_roofline(model: GPUArchModel, workloads: List[Dict]):
    """Renders ASCII visualization of the GPU Roofline Model."""
    print("\n" + "=" * 70)
    print(f" GPU Roofline Model: {model.name}")
    print(f" Peak Compute: {model.peak_fp32_tflops:.1f} TFLOP/s  |  Peak DRAM BW: {model.peak_dram_bw_gbs:.0f} GB/s")
    print(f" Ridge Point (Compute/Memory balance): {model.ridge_point:.2f} FLOPs/byte")
    print("=" * 70)

    print("\n  Attainable Performance (GFLOP/s)")
    print("  ^")
    print(f"  |  Peak: {model.peak_fp32_tflops * 1000.0:.0f} GFLOP/s ------------------------------------+ (Compute Ceiling)")
    print("  |                                                  /")
    print("  |                                                 /")
    print("  |                                                /  <-- Ridge Point")
    print("  |                                               /")
    print("  |                                              / (DRAM BW Slanted Ceiling)")
    print("  |                                             /")
    print("  +--------------------------------------------+--------------------> Arithmetic Intensity")
    print(f"  0.0                                          {model.ridge_point:.1f} (FLOPs/byte)\n")

    print(f"{'Workload Name':<30} {'Intensity':<12} {'Attainable':<15} {'Operational Regime'}")
    print("-" * 75)
    for w in workloads:
        ai = w["intensity"]
        perf, regime = model.evaluate_roofline(ai)
        print(f"{w['name']:<30} {ai:<12.2f} {perf:<15.1f} {regime}")
    print("-" * 75 + "\n")


def print_occupancy_sweep(model: GPUArchModel):
    """Evaluates warp occupancy across block size configurations."""
    print("=" * 70)
    print(f" SM Occupancy & Resource Allocation Sweep ({model.name})")
    print("=" * 70)
    print(f"{'Block Size':<12} {'Warps/Block':<14} {'Active Blocks':<15} {'Occupancy':<12} {'Limiting Hardware Factor'}")
    print("-" * 70)
    for bs in [32, 64, 96, 128, 192, 256, 384, 512, 768, 1024]:
        diag = model.diagnose_occupancy(bs, regs_per_thread=32, shmem_bytes_per_block=0)
        occ_pct = diag["theoretical_occupancy"] * 100.0
        print(
            f"{bs:<12} {math.ceil(bs / 32):<14} {diag['active_blocks']:<15} "
            f"{occ_pct:>5.1f}%      {diag['primary_limiter']}"
        )
    print("-" * 70 + "\n")


def parse_benchmark_log(log_path: str):
    """Parses output log from warpscope executable and aggregates metrics."""
    if not os.path.exists(log_path):
        print(f"[Error] Log file '{log_path}' not found.")
        return

    print(f"\nParsing Benchmark Log: {log_path} ...")
    pattern = re.compile(
        r"<<numBlocks,\s*blockSize>>\s*=\s*<<\s*([\d,]+),\s*([\d,]+)>>,\s*"
        r"occupancy:\s*([\d.]+),\s*bandwidth:\s*([\d.]+)([A-Za-z]+)/s,\s*time:\s*([\d.]+)\s*ms"
    )

    records = []
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                nb = int(match.group(1).replace(",", ""))
                bs = int(match.group(2).replace(",", ""))
                occ = float(match.group(3))
                bw_val = float(match.group(4))
                bw_unit = match.group(5)
                time_ms = float(match.group(6))

                # Normalize bandwidth to GB/s
                multiplier = {"K": 1e-6, "M": 1e-3, "G": 1.0, "T": 1e3}.get(bw_unit[0].upper(), 1.0)
                bw_gbs = bw_val * multiplier

                records.append({
                    "blocks": nb,
                    "block_size": bs,
                    "occupancy": occ,
                    "bandwidth_gbs": bw_gbs,
                    "time_ms": time_ms,
                })

    if not records:
        print("No matching benchmark output lines found in file.")
        return

    best_time = min(records, key=lambda x: x["time_ms"])
    best_bw = max(records, key=lambda x: x["bandwidth_gbs"])
    best_occ = max(records, key=lambda x: x["occupancy"])

    print("\n--- Parsed Benchmark Highlights ---")
    print(f"Total Sweep Runs Recorded : {len(records)}")
    print(f"Lowest Latency            : {best_time['time_ms']:.2f} ms (<<{best_time['blocks']}, {best_time['block_size']}>>)")
    print(f"Peak Bandwidth            : {best_bw['bandwidth_gbs']:.2f} GB/s (<<{best_bw['blocks']}, {best_bw['block_size']}>>)")
    print(f"Highest SM Occupancy      : {best_occ['occupancy'] * 100:.1f}% (<<{best_occ['blocks']}, {best_occ['block_size']}>>)")


def main():
    parser = argparse.ArgumentParser(
        description="GPU Architecture & Roofline Performance Modeler (Irene Therese Joseph)"
    )
    parser.add_argument(
        "--arch",
        choices=["RTX-4090", "A100-SXM4-40GB", "H100-SXM5-80GB", "RTX-3080"],
        default="RTX-4090",
        help="Target GPU hardware architecture for analysis (default: RTX-4090)",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Path to warpscope stdout text log to parse and summarize",
    )
    args = parser.parse_args()

    model = GPUArchModel(args.arch)

    # Standard GPU architectural workloads evaluated in warpscope
    workloads = [
        {"name": "Vector Add (1:1 Coalesced)", "intensity": 1.0 / 12.0},       # 1 FLOP / 12 Bytes
        {"name": "Euclidean Distance (float2)", "intensity": 6.0 / 20.0},     # 6 FLOPs / 20 Bytes
        {"name": "GEMM (Naive Global Memory)", "intensity": 2.0 / 8.0},       # 0.25 FLOPs / Byte
        {"name": "GEMM (Shared-Memory Tiled)", "intensity": (2.0 * 16.0) / 8.0}, # ~4.0 FLOPs / Byte
    ]

    print_ascii_roofline(model, workloads)
    print_occupancy_sweep(model)

    if args.log:
        parse_benchmark_log(args.log)


if __name__ == "__main__":
    main()
