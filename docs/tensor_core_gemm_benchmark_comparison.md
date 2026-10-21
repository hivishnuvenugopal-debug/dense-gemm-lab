# Tensor Core GEMM Benchmark Comparison

## Objective

Compare the existing register-tiled FP32 GEMM variants with the first WMMA Tensor Core GEMM implementation on an NVIDIA A100-SXM4-80GB (compute capability 8.0).

The goal is to establish a reproducible baseline, understand observed performance trends, and identify next optimization experiments.

## Hardware and workload

- **GPU:** NVIDIA A100-SXM4-80GB
- **Compute capability:** 8.0 (Ampere)
- **Matrix dimensions:** \(M=K=N=1024\)
- **Operation:** \(C = A \times B\)
- **Nominal work:** \(2MNK = 2,147,483,648\) floating-point operations
- **Register-tiled implementation:** FP32 GEMM, `TILE_K = 32`, output tile 32 × 32
- **WMMA implementation:** FP16 inputs, FP32 accumulation and FP32 output; 16 × 16 × 16 WMMA operations
- **Benchmark repetitions:** 100 for register-tiled template and 100 timed kernel launches for WMMA

> **Precision caveat:** These are not precision-equivalent implementations. The register-tiled kernel uses FP32 inputs, whereas WMMA uses FP16 inputs and accumulates in FP32. WMMA's performance advantage therefore reflects both the implementation/hardware path and the different input precision. A TF32 or carefully controlled alternative benchmark is needed for a fairer comparison at FP32 input storage.

## Results

### Register-tiled GEMM configuration sweep

All results below are from the measured `register_tiled_gemm_template` run.

| Register tile (RM × RN) | Block dimensions (X × Y) | Threads/block | Average time (ms) | Throughput (TFLOP/s) | Correctness |
|---|---:|---:|---:|---:|---|
| 1 × 1 | 32 × 32 | 1024 | 0.881551 | 2.436028 | PASS |
| 1 × 2 | 16 × 32 | 512 | 0.738314 | 2.908631 | PASS |
| 2 × 2 | 16 × 16 | 256 | 0.530504 | 4.048009 | PASS |
| 4 × 2 | 16 × 8 | 128 | 0.586793 | 3.659696 | PASS |
| 4 × 4 | 8 × 8 | 64 | 0.602481 | 3.564403 | PASS |

All variants used `TILE_K = 32`, shared tiles A[32 × 32] and B[32 × 32], and 100 benchmark iterations.

### WMMA Tensor Core GEMM

| Matrix dimensions | Input type | Accumulator/output type | Average time (ms) | Throughput (TFLOP/s) | Correctness |
|---|---|---|---:|---:|---|
| 1024 × 1024 × 1024 | FP16 | FP32 / FP32 | 0.145644 | 14.744794 | PASS |

The WMMA kernel used one warp per 16 × 16 output tile and iterated through the K dimension in chunks of 16. This is a deliberately simple baseline without explicit shared-memory staging or advanced pipelining.

### Summary comparison

| Metric | Best register-tiled variant | WMMA |
|---|---:|---:|
| Configuration | RM × RN = 2 × 2 | 16 × 16 output tile per warp |
| Average time (ms) | 0.530504 | 0.145644 |
| Throughput (TFLOP/s) | 4.048009 | 14.744794 |
| Relative to best register-tiled throughput | 1.00× | 3.64× |
| Relative to best register-tiled latency | 1.00× | 3.64× faster |

The WMMA result is about **3.64× the throughput** and has about **72.5% lower latency** than the best register-tiled configuration in these measurements. This comparison is indicative rather than precision-equivalent because the input types differ.

## Observations and possible explanations

### 1. Register tiling improves performance up to a point

Moving from RM × RN = 1 × 1 to 2 × 2 improved throughput from 2.44 to 4.05 TFLOP/s. Giving each thread multiple output accumulators can increase data reuse and amortize some indexing and loop overhead.

Larger register tiles are not automatically better. The 4 × 2 and 4 × 4 variants were slower than 2 × 2 in this run. Possible contributors include:
- Higher register demand per thread.
- Fewer threads per block and changed scheduling/occupancy behavior.
- Different memory-access and instruction-level parallelism characteristics.
- Resource allocation or compiler code-generation differences.

These are hypotheses, not confirmed causes. Check compiler resource reports and profile the kernels before attributing the slowdown to a particular factor.

### 2. The 1 × 1 configuration is especially slow

The 1 × 1 variant uses 1024 threads per block. That is the architectural maximum threads per block on many CUDA GPUs, including the A100, leaving little flexibility for scheduling a block and potentially limiting occupancy/resource efficiency. It also provides less per-thread output reuse than larger register tiles.

The measured result supports testing more moderate block sizes; it does not by itself establish which resource is the dominant bottleneck.

### 3. WMMA has substantially higher throughput

WMMA reached 14.74 TFLOP/s, compared with 4.05 TFLOP/s for the best tested register-tiled FP32 kernel. Tensor Cores are specialized for matrix multiply-accumulate operations, and WMMA expresses work in matrix tiles rather than scalar output calculations.

The simple WMMA implementation still has limitations:
- It reloads A and B fragments each K iteration.
- It uses one warp per output tile.
- It has no explicit shared-memory staging to promote reuse across neighboring output tiles.
- It does not use double buffering or a more advanced pipeline.
- It has not been tuned for multiple warps or multiple output tiles per block.

The current measurement does not identify which limitation dominates. Profiling is needed to confirm the bottleneck.

### 4. Relative error can look large when the reference is near zero

The register-tiled benchmark reported maximum absolute error 0.000072 and RMSE 0.000006, with zero elements above an absolute error of 1e-3. Its maximum relative error was 0.181515 using a denominator floor of 1e-6.

WMMA reported maximum absolute error 0.000164 and maximum relative error 3.099442, while passing its combined absolute/relative tolerance.

A large maximum relative error can arise for reference elements close to zero. It should not be interpreted without absolute error, the tolerance rule, and aggregate metrics. The two programs also use different correctness criteria, so their PASS statuses alone are not a strict numerical comparison.

## Potential improvements

### Priority 1 — Make the comparison more controlled

1. Implement a TF32 Tensor Core path using FP32 inputs and FP32 accumulation, where supported.
2. Use identical input matrices or deterministic generation and compare against a common FP32 reference.
3. Report latency, throughput, max absolute error, RMSE, and a clearly defined normalized error metric for every implementation.
4. Keep iteration counts, warm-up policy, and timing methodology consistent.
5. Repeat runs and report variability (for example, median and spread) to reduce the effect of GPU clock and system variation.

### Priority 2 — Profile before changing the kernel

Collect:
- Registers per thread and shared memory per block from compiler resource reports.
- Occupancy and active warps.
- Tensor Core / matrix instruction utilization for WMMA.
- Memory throughput and cache behavior.
- Warp stalls and instruction-pipeline utilization.

Use Nsight Compute if available on the cluster and permitted by the environment. Profiling should guide optimization rather than relying on assumptions from elapsed time alone.

### Priority 3 — Improve the WMMA implementation

Potential experiments, in incremental order:

1. Increase matrix dimensions to see how throughput scales.
2. Add explicit shared-memory staging to improve reuse of A and B tiles.
3. Let a block compute multiple output tiles where appropriate, reusing staged operands.
4. Explore multiple warps per block and different output-tile arrangements.
5. Investigate double buffering and pipelined data movement after a simpler shared-memory version is correct.
6. Inspect generated PTX/SASS to verify the intended matrix instructions are emitted.

Each change should be benchmarked independently and validated numerically.

### Priority 4 — Tune the register-tiled baseline

1. Sweep block dimensions and register tile sizes systematically.
2. Examine compiler register counts and occupancy for each configuration.
3. Test `TILE_K` values such as 16, 32, and 64, subject to kernel constraints and shared-memory limits.
4. Check for register spilling and whether the compiler vectorizes or unrolls loops as expected.
5. Keep the best measured configuration as the FP32 CUDA-core baseline.

## Suggested next experiment

**Implement a TF32 Tensor Core GEMM and compare it with the existing FP32 register-tiled kernel.**

This keeps FP32 input storage while using the A100's Tensor Core path, making the comparison more informative than FP16-input WMMA alone. Confirm the selected TF32 API's numerical behavior and supported architecture in the CUDA documentation, and validate all results against the same reference.

## Reproducibility notes

The WMMA benchmark reported:

```text
GPU: NVIDIA A100-SXM4-80GB
Matrix shape (M,K,N): 1024, 1024, 1024
Input type: FP16
Accumulator/output type: FP32
Average kernel time: 0.145644 ms
Effective throughput: 14.744794 TFLOP/s
Max absolute error: 0.000164
Max relative error: 3.099442
Correctness: PASS
```

The register-tiled template benchmark used the following interface:

```text
M K N RM RN TILE_K BLOCK_X BLOCK_Y [ITERS]
```

Example commands used for the measured configurations:

```bash
./build/register_tiled_gemm_template 1024 1024 1024 1 1 32 32 32 100
./build/register_tiled_gemm_template 1024 1024 1024 1 2 32 16 32 100
./build/register_tiled_gemm_template 1024 1024 1024 2 2 32 16 16 100
./build/register_tiled_gemm_template 1024 1024 1024 4 2 32 16 8 100
./build/register_tiled_gemm_template 1024 1024 1024 4 4 32 8 8 100
./build/wmma_gemm 1024 1024 1024
```

