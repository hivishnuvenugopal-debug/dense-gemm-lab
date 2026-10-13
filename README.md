# Dense GEMM Lab

A CUDA C++ project focused on developing, optimizing, and benchmarking
dense matrix multiplication kernels on NVIDIA GPUs.

## Objectives

Develop, optimize, and benchmark dense matrix multiplication kernels
in CUDA C++, comparing custom implementations against cuBLAS and
analyzing performance, numerical accuracy, and hardware utilization.

## Scope

The project will progressively investigate:

- Naive CUDA matrix multiplication
- Tiled GEMM with shared-memory reuse
- Performance comparisons against cuBLAS
- Numerical accuracy and precision trade-offs
- GPU profiling and performance bottlenecks
- Tensor Core acceleration
- Architecture-specific optimization, including NVIDIA Blackwell

## Experimental Methodology

Each implementation will be evaluated using:

- Correctness against a trusted reference
- Numerical error
- Execution latency
- Throughput in TFLOP/s
- GPU architecture and precision
- Matrix dimensions and benchmark configuration

Performance claims will be accompanied by reproducible benchmark
conditions and clearly identified hardware.

## Hardware

Initial development may use an NVIDIA A100 GPU.

Blackwell-specific optimization and performance claims will require
access to compatible Blackwell hardware.

## Repository Structure

- `src/`: CUDA kernel implementations
- `include/`: Shared interfaces
- `benchmarks/`: Benchmark drivers and results
- `tests/`: Correctness and accuracy tests
- `docs/`: Technical documentation and analysis

## Status

Initial repository setup.
