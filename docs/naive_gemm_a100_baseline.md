# Naive CUDA GEMM Baseline — NVIDIA A100

## TL;DR

- **GPU:** NVIDIA A100-SXM4-80GB (compute capability 8.0).
- **Workload:** FP32 GEMM, `C[M,N] = A[M,K] × B[K,N]`, with `M=N=K=1024`.
- **Best measured configuration:** `block=(32,4)`, with 128 threads per block.
- **Mean kernel time:** 0.641559 ms across five independent runs.
- **Effective throughput:** approximately 3.347 TFLOP/s.
- **Correctness:** All tested configurations passed.
- **Key finding:** Thread-block geometry affects performance even when the total number of launched threads remains constant. However, the performance differences among the three leading configurations are small.
- **Next step:** Implement shared-memory tiled GEMM and compare it against this naive baseline.

## Summary

We implemented a parameterized naive CUDA GEMM kernel and benchmarked it on an NVIDIA A100.

Each CUDA thread computes one output element. The kernel uses a conventional loop over the reduction dimension `K` and does not explicitly use shared-memory tiling or Tensor Cores.

The experiments proceeded in two stages:

1. **Initial sweep:** Tested seven thread-block configurations to identify promising candidates.
2. **Repeated measurements:** Executed the three leading configurations five times each to assess their performance consistency.

Each program execution performs 100 timed kernel iterations using CUDA events. The reported execution time is the average over those iterations.

## 1. Initial Configuration Sweep

### Workload

- `M = 1024`
- `N = 1024`
- `K = 1024`
- Data type: FP32
- GPU: NVIDIA A100-SXM4-80GB
- Kernel: Naive GEMM, one output element per thread

### Results

The following results are from individual benchmark executions.

| Block dimensions (`x × y`) | Threads per block | Mean kernel time (ms) | Throughput (TFLOP/s) | Correctness |
|---|---:|---:|---:|---|
| 8 × 32 | 256 | 1.247048 | 1.7221 | PASS |
| 16 × 8 | 128 | 0.754371 | 2.8467 | PASS |
| 16 × 16 | 256 | 0.761979 | 2.8183 | PASS |
| 16 × 32 | 512 | 0.834877 | 2.5722 | PASS |
| **32 × 4** | **128** | **0.641618** | **3.3470** | **PASS** |
| 32 × 8 | 256 | 0.641751 | 3.3463 | PASS |
| 32 × 16 | 512 | 0.646400 | 3.3222 | PASS |

Throughput is calculated using:

`Throughput = (2 × M × N × K) / kernel_time`

The calculation counts one multiplication and one addition as two floating-point operations.

### Initial Sweep: Conclusions

**1. Configurations with 32 threads along the X dimension performed best.**

The `32 × 4`, `32 × 8`, and `32 × 16` configurations were faster than the other configurations in this sweep.

**2. The `8 × 32` configuration was particularly slow.**

It achieved approximately 1.722 TFLOP/s, compared with approximately 3.347 TFLOP/s for `32 × 4`.

In the kernel, adjacent threads access adjacent columns of the matrices. Changing the block dimensions changes how threads map to rows and columns, which can affect memory-access patterns and execution efficiency.

The measurements indicate that block geometry matters, although profiling would be needed to establish the precise causes.

**3. More threads per block did not guarantee better performance.**

The `32 × 16` configuration uses 512 threads per block but was slower than `32 × 4`, which uses 128 threads.

Thread count alone is therefore insufficient to predict performance.

**4. The leading configurations were close enough to warrant repeated measurements.**

The initial times for `32 × 4` and `32 × 8` differed by only about 0.02%. We therefore selected `32 × 4`, `32 × 8`, and `32 × 16` for further benchmarking.

## 2. Repeated Measurements of the Three Leading Configurations

Each configuration was executed five times independently. Each execution measured 100 kernel iterations.

### Results

| Block dimensions | Threads per block | Mean kernel time (ms) | Std. dev. (ms) | Throughput (TFLOP/s) |
|---|---:|---:|---:|---:|
| **32 × 4** | **128** | **0.641559** | 0.000098 | **3.3473** |
| 32 × 8 | 256 | 0.642937 | 0.001131 | 3.3401 |
| 32 × 16 | 512 | 0.646607 | 0.000060 | 3.3212 |

Mean and standard deviation are calculated from the five independently reported execution times. Throughput is calculated from the mean time.

### Repeated Measurements: Conclusions

**1. `32 × 4` had the lowest mean execution time.**

Its mean time was approximately:

- 0.21% lower than `32 × 8`.
- 0.79% lower than `32 × 16`.

This makes `32 × 4` the current best measured configuration for this workload.

**2. `32 × 8` remained competitive.**

Its mean execution time was only about 1.38 microseconds higher than `32 × 4`.

One of its five runs was slower than the other four, increasing its observed standard deviation.

**3. `32 × 16` had the smallest observed variation.**

Its standard deviation was approximately 0.000060 ms, compared with 0.000098 ms for `32 × 4` and 0.001131 ms for `32 × 8`.

However, it also had the highest mean execution time of the three configurations.

**4. The winner should not be overinterpreted.**

The performance differences are small. These measurements identify `32 × 4` as the current leading configuration, but more extensive benchmarking would be required to establish whether the difference is statistically robust across different GPU states and workloads.

## 3. Correctness Validation

All tested configurations passed the correctness checks.

For the `1024 × 1024 × 1024` workload, the reported errors were:

- Maximum absolute error: approximately `5.36e-7`.
- Maximum relative error: approximately `4.94e-6`.
- Output elements exceeding the configured absolute tolerance: `0 / 1048576`.

A separate test with non-multiple dimensions,

`M = 257, N = 193, K = 129`

also passed correctness validation. This confirms that the current implementation handles matrix dimensions that are not exact multiples of the thread-block dimensions.

## 4. Benchmark Configuration

| Property | Value |
|---|---|
| GPU | NVIDIA A100-SXM4-80GB |
| Compute capability | 8.0 |
| Operation | `C[M,N] = A[M,K] × B[K,N]` |
| Data type | FP32 |
| Main workload | `1024 × 1024 × 1024` |
| Kernel implementation | Naive CUDA GEMM |
| Work per thread | One output element |
| Timing method | CUDA events |
| Timed iterations per execution | 100 |
| Repeated executions for leading configurations | 5 |

The reported timing measures kernel execution, excluding host-to-device and device-to-host transfers.

## 5. Limitations

- Only one main matrix size was used for the block-configuration comparison.
- The repeated measurements cover only the three leading configurations.
- The kernel does not explicitly reuse input tiles through shared memory.
- The implementation does not use Tensor Cores.
- These results do not establish performance across different matrix shapes or data types.
- The small timing differences among the leading configurations should be interpreted cautiously.

## 6. Conclusion and Next Steps

The experiments establish a reproducible naive CUDA GEMM baseline on an NVIDIA A100.

The initial sweep showed that block geometry materially affects performance. The repeated measurements identified `block=(32,4)` as the fastest of the three leading configurations for the tested workload, achieving approximately 3.347 TFLOP/s.

The next optimization is **shared-memory tiled GEMM**. By loading tiles of A and B into shared memory and reusing those values across multiple output computations, the next kernel can reduce redundant global-memory accesses.

The next experiment will compare:

1. Naive GEMM using `block=(32,4)`.
2. Shared-memory tiled GEMM.
3. A production library implementation using cuBLAS SGEMM.

Each implementation will be evaluated for correctness, kernel execution time, and effective throughput using the same matrix dimensions and measurement procedure.
