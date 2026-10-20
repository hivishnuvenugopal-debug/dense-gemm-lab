# CUDA GEMM Exercise: Register-Mapped Tiling

## Objective

Implement and benchmark FP32 general matrix multiplication (GEMM) on an NVIDIA A100 GPU, progressing from runtime-configurable register tiling to compile-time template-specialized register tiling.

The operation is:

\[
C_{M\times N} = A_{M\times K}B_{K\times N}
\]

The exercise investigates how per-thread register accumulator tiles and CUDA thread-block geometry affect kernel performance, while validating numerical correctness against a CPU reference.

## Hardware and workload

- **GPU:** NVIDIA A100-SXM4-80GB
- **Compute capability:** 8.0 (Ampere)
- **Data type:** FP32
- **Matrix dimensions:** \(M=K=N=1024\)
- **Operation count:** \(2MNK = 2,147,483,648\) floating-point operations
- **Shared-memory reduction tile:** `TILE_K = 32` for the recorded template-specialized runs
- **Benchmark iterations:** 100 timed kernel launches
- **Correctness reference:** CPU GEMM with double-precision accumulation, converted to FP32 for comparison

The reported throughput is effective GEMM throughput, calculated as \(2MNK/t\). Kernel timing excludes memory allocation, host-to-device copies, device-to-host copies, and CPU reference computation.

## Kernel design

### Shared-memory tiling

Each thread block stages tiles of A and B in shared memory. The output tile dimensions are derived from the per-thread register tile and block dimensions:

\[
TILE_M = RM \times BLOCK_Y
\]

\[
TILE_N = RN \times BLOCK_X
\]

The shared-memory tiles are:

- A tile: `TILE_M × TILE_K`
- B tile: `TILE_K × TILE_N`

For the recorded configurations, the output tile is always \(32\times32\), and `TILE_K = 32`. The dynamic shared-memory allocation is therefore:

\[
(32\times32 + 32\times32)\times4 = 8192\text{ bytes/block}
\]

### Register-mapped output tiling

Each CUDA thread computes an \(RM\times RN\) tile of C and maintains its partial sums in an accumulator array.

The initial implementation used a fixed maximum-size accumulator array and runtime tile dimensions:

```cpp
float acc[MAX_REG_TILE * MAX_REG_TILE] = {0.0f};
```

This implementation passed correctness checks but showed lower performance.

The revised implementation uses compile-time template parameters:

```cpp
template <int RM, int RN, int TILE_K>
__global__ void register_tiled_gemm_kernel(/* ... */)
{
    float acc[RM][RN] = {};
    // GEMM computation
}
```

The program dispatches to separate kernel specializations. This allows the compiler to optimize accumulator dimensions and loops for each register-tile configuration.

Compile-time specialization does **not** guarantee that every source-level variable is held in a register; compiler resource reports are used to inspect register usage and spill traffic.

## Build and run

Compile for the A100 (`sm_80`) and request verbose PTXAS resource reports:

```bash
mkdir -p build

nvcc -O3 -std=c++17 -arch=sm_80 \
    -Xptxas=-v \
    src/register_tiled_gemm_template.cu \
    -o build/register_tiled_gemm_template \
    2>&1 | tee compile_report.txt
```

Command-line interface:

```text
register_tiled_gemm_template M K N RM RN TILE_K BLOCK_X BLOCK_Y [ITERS]
```

Example for the 2×2 register tile:

```bash
./build/register_tiled_gemm_template \
    1024 1024 1024 \
    2 2 \
    32 \
    16 16 \
    100
```

The supported template variants in this exercise are `RM ∈ {1, 2, 4}`, `RN ∈ {1, 2, 4}`, and `TILE_K = 32`.

## Results: Initial runtime-configurable implementation

The original runtime-configurable kernel was benchmarked with a 2×2 register accumulator tile, 32×32 shared-memory tiles, `TILE_K = 32`, and a 16×16 thread block.

| Metric | Result |
|---|---:|
| Average kernel time | 2.410660 ms (first recorded run) |
| Effective throughput | 0.890828 TFLOP/s |
| Maximum absolute error | 0.000072 |
| RMSE | 0.000006 |
| Correctness | PASS |

A subsequent run of the same configuration measured 2.527846 ms and 0.849531 TFLOP/s. These variations show why repeated measurements and stable GPU conditions matter.

## Results: Template-specialized implementation

All results below are measured on the NVIDIA A100-SXM4-80GB for 1024×1024×1024 FP32 GEMM. The output tile per block is 32×32 and `TILE_K = 32`.

| Register tile (RM×RN) | Block dimensions (X×Y) | Threads/block | Average time (ms) | Throughput (TFLOP/s) | Speedup vs. 1×1 |
|---|---:|---:|---:|---:|---:|
| 1×1 | 32×32 | 1024 | 0.878930 | 2.443293 | 1.00× |
| 1×2 | 16×32 | 512 | 0.735713 | 2.918914 | 1.19× |
| 2×2 | 16×16 | 256 | 0.528353 | 4.064484 | 1.66× |
| 4×2 | 16×8 | 128 | 0.527145 | 4.073801 | 1.67× |
| 4×4 | 8×8 | 64 | 0.601958 | 3.567495 | 1.46× |

All five configurations passed the correctness test.

### Numerical accuracy

All five template-specialized configurations reported the same summary metrics:

| Metric | Result |
|---|---:|
| Maximum absolute error | 0.000072 |
| Maximum relative error (denominator floor \(10^{-6}\)) | 0.181515 |
| RMSE | 0.000006 |
| Elements with absolute error > \(10^{-3}\) | 0 / 1,048,576 |
| Worst-error index | (793, 625) |
| GPU value at worst-error index | 38.721436 |
| CPU reference at worst-error index | 38.721363 |
| Correctness threshold | Maximum absolute error ≤ \(10^{-3}\) |
| Result | PASS |

The maximum relative error is larger than the absolute error might suggest because relative error can become large for reference values close to zero. For this workload, the maximum absolute error, RMSE, and count of elements exceeding the absolute-error threshold provide useful complementary checks.

### Compiler register and spill report

The build with `-Xptxas=-v` emitted all nine combinations of `RM ∈ {1,2,4}` and `RN ∈ {1,2,4}`, with `TILE_K = 32`.

| RM×RN specialization | Registers/thread | Stack frame | Spill stores | Spill loads |
|---|---:|---:|---:|---:|
| 1×1 | 56 | 0 bytes | 0 bytes | 0 bytes |
| 1×2 | 56 | 0 bytes | 0 bytes | 0 bytes |
| 1×4 | 50 | 0 bytes | 0 bytes | 0 bytes |
| 2×1 | 54 | 0 bytes | 0 bytes | 0 bytes |
| 2×2 | 44 | 0 bytes | 0 bytes | 0 bytes |
| 2×4 | 30 | 0 bytes | 0 bytes | 0 bytes |
| 4×1 | 44 | 0 bytes | 0 bytes | 0 bytes |
| 4×2 | 30 | 0 bytes | 0 bytes | 0 bytes |
| 4×4 | 38 | 0 bytes | 0 bytes | 0 bytes |

No register spill loads or stores were reported for these compiled specializations. This indicates that PTXAS did not generate spill traffic for them. It does not by itself prove that all source-level variables are always held in registers, nor does it identify the kernel's primary performance bottleneck.

## Analysis

### Template specialization improved performance

For the 2×2 configuration, the runtime-configurable implementation achieved approximately 0.85–0.89 TFLOP/s, while the template-specialized implementation achieved approximately 4.06 TFLOP/s.

Using the later runtime-kernel result of 2.527846 ms as the comparison:

\[
\text{Speedup} = \frac{2.527846}{0.528353} \approx 4.78\times
\]

The corresponding kernel time fell by approximately 79%. Compile-time specialization allows more aggressive optimization of accumulator arrays and loops. The performance improvement is consistent with this, although it cannot be attributed exclusively to register allocation without further compiler and profiler analysis.

### Best measured configuration

The 4×2 register tile was the fastest recorded configuration at 4.073801 TFLOP/s. It was only about 0.23% faster than the 2×2 configuration:

\[
\frac{0.528353 - 0.527145}{0.528353}\times100\% \approx 0.23\%
\]

This difference is small and may be within measurement variability. Repeated runs with controlled GPU conditions are needed before identifying a definitive winner.

The 4×4 configuration was slower than both 2×2 and 4×2. Increasing the number of output accumulators per thread does not guarantee higher throughput because instruction scheduling, occupancy, register use, and other factors affect performance.

### Headroom for further optimization

The A100's nominal peak FP32 CUDA-core throughput is approximately 19.5 TFLOP/s. The best measured result of 4.074 TFLOP/s is approximately 20.9% of that nominal peak. This comparison is indicative, not a complete measure of kernel efficiency.

## Next experiments

1. **Repeat benchmarks.** Run each configuration multiple times and compare medians or distributions to quantify variability.
2. **Vary `TILE_K`.** Test 16, 32, and 64 while keeping the register tile and output tile fixed. Add template instantiations for each `TILE_K` before testing values other than 32.
3. **Profile with Nsight Compute.** Inspect achieved occupancy, memory throughput, instruction utilization, and stall reasons to identify the dominant bottleneck.
4. **Compare with an optimized baseline.** Benchmark against a trusted GEMM implementation, such as cuBLAS, using equivalent precision and timing methodology.
5. **Explore Tensor Cores.** After establishing a robust CUDA-core baseline, investigate Tensor Core GEMM using appropriate data types and matrix layouts on the A100.

## Conclusion

This exercise demonstrated that compile-time register-tile specialization can substantially improve a straightforward shared-memory-tiled CUDA GEMM implementation. On an NVIDIA A100, the best recorded configuration achieved 4.074 TFLOP/s for a 1024³ FP32 GEMM, with all configurations passing the numerical correctness threshold and no PTXAS-reported spill loads or stores.

The next priority is to establish repeatable performance measurements, explore reduction tile depth, and use profiler evidence to guide further optimization.

