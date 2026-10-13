# Naive CUDA GEMM Baseline — A100

## TL;DR

- **Best measured configuration:** `block=(32, 4)` (128 threads per block).
- **Mean kernel time:** **0.641559 ms** for a `1024 × 1024 × 1024` FP32 GEMM.
- **Effective throughput:** approximately **3.347 TFLOP/s**.
- All tested configurations passed the correctness check.
- The timing differences are small: `32 × 4` was only about **0.21% faster** than `32 × 8` and **0.79% faster** than `32 × 16` by their five-run means. Treat `32 × 4` as the current winner for this workload, not a universally optimal block shape.

## Summary

We benchmarked the naive CUDA GEMM kernel on an NVIDIA A100-SXM4-80GB using square matrices with:

- `M = 1024`
- `N = 1024`
- `K = 1024`
- FP32 inputs and output
- 100 timed kernel iterations per program execution
- 5 separate program executions per block configuration

The kernel computes one output element per CUDA thread. These tests compare launch configurations; they do not yet include shared-memory tiling or Tensor Core acceleration.

### Results

| Block dimensions (`x × y`) | Threads per block | Mean kernel time (ms) | Std. dev. (ms) | Effective throughput (TFLOP/s) | Correctness |
|---|---:|---:|---:|---:|---|
| **32 × 4** | 128 | **0.641559** | 0.000098 | **3.3473** | PASS |
| 32 × 8 | 256 | 0.642937 | 0.001131 | 3.3401 | PASS |
| 32 × 16 | 512 | 0.646607 | 0.000060 | 3.3212 | PASS |

Mean time and standard deviation are calculated from the five reported execution times for each configuration. Throughput is calculated from the mean time using `2 × M × N × K` FLOPs per GEMM.

### Observations

1. **`32 × 4` had the lowest mean time.** Its mean was approximately 0.21% lower than `32 × 8` and 0.79% lower than `32 × 16`.
2. **`32 × 8` was close in performance.** One run was slower than the other four, which increased its measured run-to-run variation.
3. **`32 × 16` was the most consistent in these five runs**, based on the smallest standard deviation, but it had the highest mean time among these three configurations.
4. **Correctness passed in every run.** For the `1024³` case, the reported maximum absolute error was approximately `5.36e-7`, and zero output elements exceeded the configured tolerance.
5. **These results are specific to this kernel, GPU, and matrix size.** The small performance differences should not be generalized without more measurements and workloads.

## Benchmark configuration

- GPU: NVIDIA A100-SXM4-80GB
- Compute capability: 8.0
- Operation: `C[M,N] = A[M,K] × B[K,N]`
- Data type: FP32
- Matrix dimensions: `1024 × 1024 × 1024`
- Timing: CUDA events around 100 repeated kernel launches per run

## Conclusion and next step

Use `block=(32, 4)` as the **current naive-kernel baseline** for this workload. Next, implement a shared-memory tiled GEMM and compare its correctness, kernel time, and effective throughput against this baseline under the same measurement procedure.
