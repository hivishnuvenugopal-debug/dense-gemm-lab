# Shared-Memory Tiled GEMM — Baseline Results

## Overview

This experiment implements a generic CUDA shared-memory tiled GEMM for row-major
single-precision matrices:

\[
C_{M\times N} = A_{M\times K} B_{K\times N}
\]

The program accepts matrix dimensions, tile dimensions, and CUDA block dimensions
at runtime. Block dimensions may differ from tile dimensions, but each block
dimension must not exceed the corresponding tile dimension.

The register-accumulation variant keeps each thread's output accumulators across
the full K loop and writes each valid output element to global memory once.

## Test environment

- **GPU:** NVIDIA A100-SXM4-80GB
- **Matrix dimensions:** M = 1024, N = 1024, K = 1024
- **Data type:** `float` (FP32)
- **Timing:** 10 warm-up launches, followed by 100 timed launches using CUDA events
- **Throughput calculation:** `2 * M * N * K / kernel_time`
- **Validation:** Compared against a CPU reference with absolute tolerance
  `1e-4` and relative tolerance `1e-4`

## Best result observed

| Metric | Value |
|---|---:|
| Tile dimensions (`tile_x × tile_y`) | 32 × 32 |
| Block dimensions (`block_x × block_y`) | 32 × 32 |
| Threads per block | 1024 |
| Grid dimensions | 32 × 32 |
| Dynamic shared memory per block | 8192 bytes (8 KiB) |
| Average kernel time | 0.816456 ms |
| Effective throughput | 2.630251 TFLOP/s |
| Maximum absolute error | 1.90734863e-05 |
| Maximum relative error | 1.50373809e-01 |
| Correctness | PASS |

The maximum relative error can be large for output elements whose reference values
are close to zero. The combined absolute/relative tolerance is used for the PASS/FAIL
decision.

## Benchmark configurations

All rows below use the same GPU and 1024 × 1024 × 1024 matrices. Results are from
100 timed launches after 10 warm-up launches.

| Tile (`x × y`) | Block (`x × y`) | Threads/block | Shared memory | Time (ms) | Throughput (TFLOP/s) | Correctness |
|---|---|---:|---:|---:|---:|---|
| 32 × 32 | 32 × 32 | 1024 | 8 KiB | **0.816456** | **2.630251** | PASS |
| 16 × 32 | 16 × 16 | 256 | 3 KiB | 0.861942 | 2.491449 | PASS |
| 32 × 16 | 16 × 16 | 256 | 6 KiB | 0.912947 | 2.352254 | PASS |
| 32 × 32 | 16 × 16 | 256 | 8 KiB | 0.919808 | 2.334709 | PASS |
| 16 × 16 | 16 × 16 | 256 | 2 KiB | 0.955218 | 2.248161 | PASS |
| 16 × 16 | 8 × 8 | 64 | 2 KiB | 1.135350 | 1.891473 | PASS |
| 32 × 32 | 8 × 8 | 64 | 8 KiB | 1.538417 | 1.395905 | PASS |

### Earlier global-partial-sum baseline

The earlier generic tiled kernel wrote each K-tile partial sum to global memory.
Its measured result for the same matrix size and a 16 × 16 tile / 8 × 8 block was:

| Kernel variant | Tile | Block | Time (ms) | Throughput (TFLOP/s) | Correctness |
|---|---|---|---:|---:|---|
| Global partial sums | 16 × 16 | 8 × 8 | 1.257441 | 1.707820 | PASS |
| Register accumulation | 16 × 16 | 8 × 8 | 1.135350 | 1.891473 | PASS |

For this matched configuration, register accumulation reduced the observed kernel
time by approximately **9.71%** and increased measured throughput by approximately
**10.75%**.

## Observations

- The fastest configuration tested so far is a 32 × 32 tile with a 32 × 32 block.
- For a 32 × 32 tile, the 1024-thread block outperformed the 256-thread and 64-thread
  blocks in these measurements.
- Tile orientation affected performance: the 16 × 32 tile was faster than the
  32 × 16 tile for the tested 16 × 16 block.
- The 32 × 32 / 32 × 32 result is about 5.6% faster than the 16 × 32 / 16 × 16
  result, so repeated runs are recommended to confirm the ranking.
- These are measurements of a generic scalar-FP32 CUDA kernel, not a Tensor Core
  GEMM implementation. They should not be compared directly with Tensor Core peak
  performance or optimized GEMM libraries without accounting for implementation
  and precision differences.

## Reproduction

Example build command:

```bash
nvcc -O2 src/tiled_gemm_register_accum.cu -o build/tiled_gemm_register_accum
```

Run the fastest tested configuration:

```bash
./build/tiled_gemm_register_accum 1024 1024 1024 32 32 32 32
```

Run the 16 × 32 tile alternative:

```bash
./build/tiled_gemm_register_accum 1024 1024 1024 16 32 16 16
```

## Scope and next step

This is the baseline for the generic shared-memory tiled GEMM stage. The next
stage is a dedicated register-tiled kernel that maps a small output fragment to
each thread and reuses loaded operands across multiple accumulators. The current
kernel's fixed per-thread accumulator capacity and scalar arithmetic are retained
as baseline constraints for comparison.

