// register_tiled_gemm.cu
//
// Configurable FP32 GEMM using shared-memory input tiles and per-thread
// register accumulators.
//
// Usage:
//   ./register_tiled_gemm M K N RM RN TILE_M TILE_N TILE_K BLOCK_X BLOCK_Y [ITERS]
//
// Constraints:
//   TILE_M == RM * BLOCK_Y
//   TILE_N == RN * BLOCK_X
//   1 <= RM,RN <= MAX_REG_TILE (8)
//   TILE_M, TILE_N, TILE_K > 0
//
// Example (A100-friendly starting point):
//   ./register_tiled_gemm 1024 1024 1024 2 2 32 32 32 16 16 100
//
// Each thread computes an RM x RN output tile. A block computes
// (BLOCK_Y*RM) x (BLOCK_X*RN) output elements. TILE_K is the reduction
// depth loaded per shared-memory iteration.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__     \
                      << ": " << cudaGetErrorString(err__) << '\n';          \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

constexpr int MAX_REG_TILE = 8;

__global__ void register_tiled_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N,
    int RM, int RN,
    int TILE_M, int TILE_N, int TILE_K)
{
    // Dynamic shared memory:
    // As: TILE_M x TILE_K
    // Bs: TILE_K x TILE_N
    extern __shared__ float smem[];
    float* As = smem;
    float* Bs = As + static_cast<size_t>(TILE_M) * TILE_K;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int threads = blockDim.x * blockDim.y;

    const int row0 = blockIdx.y * TILE_M + threadIdx.y * RM;
    const int col0 = blockIdx.x * TILE_N + threadIdx.x * RN;

    // Fixed-size backing storage allows RM/RN to remain runtime arguments.
    // Only the first RM*RN entries are used.
    float acc[MAX_REG_TILE * MAX_REG_TILE] = {0.0f};

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        const size_t a_count = static_cast<size_t>(TILE_M) * TILE_K;
        const size_t b_count = static_cast<size_t>(TILE_K) * TILE_N;

        for (size_t idx = tid; idx < a_count; idx += threads) {
            const int r = static_cast<int>(idx / TILE_K);
            const int kk = static_cast<int>(idx % TILE_K);
            const int gr = blockIdx.y * TILE_M + r;
            const int gk = k0 + kk;
            As[idx] = (gr < M && gk < K) ? A[static_cast<size_t>(gr) * K + gk] : 0.0f;
        }

        for (size_t idx = tid; idx < b_count; idx += threads) {
            const int kk = static_cast<int>(idx / TILE_N);
            const int c = static_cast<int>(idx % TILE_N);
            const int gk = k0 + kk;
            const int gc = blockIdx.x * TILE_N + c;
            Bs[idx] = (gk < K && gc < N) ? B[static_cast<size_t>(gk) * N + gc] : 0.0f;
        }

        __syncthreads();

        const int valid_k = min(TILE_K, K - k0);
        for (int kk = 0; kk < valid_k; ++kk) {
            float a_reg[MAX_REG_TILE];
            float b_reg[MAX_REG_TILE];

            #pragma unroll
            for (int i = 0; i < MAX_REG_TILE; ++i) {
                if (i < RM) {
                    a_reg[i] = As[(threadIdx.y * RM + i) * TILE_K + kk];
                }
                if (i < RN) {
                    b_reg[i] = Bs[kk * TILE_N + threadIdx.x * RN + i];
                }
            }

            #pragma unroll
            for (int i = 0; i < MAX_REG_TILE; ++i) {
                if (i < RM) {
                    #pragma unroll
                    for (int j = 0; j < MAX_REG_TILE; ++j) {
                        if (j < RN) {
                            const int p = i * RN + j;
                            acc[p] = fmaf(a_reg[i], b_reg[j], acc[p]);
                        }
                    }
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < MAX_REG_TILE; ++i) {
        if (i < RM) {
            #pragma unroll
            for (int j = 0; j < MAX_REG_TILE; ++j) {
                if (j < RN) {
                    const int row = row0 + i;
                    const int col = col0 + j;
                    if (row < M && col < N) {
                        C[static_cast<size_t>(row) * N + col] = acc[i * RN + j];
                    }
                }
            }
        }
    }
}

static bool parse_positive_int(const char* s, int& value) {
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0' || v <= 0 || v > std::numeric_limits<int>::max()) {
        return false;
    }
    value = static_cast<int>(v);
    return true;
}

int main(int argc, char** argv) {
    if (argc < 11 || argc > 12) {
        std::cerr
            << "Usage: " << argv[0]
            << " M K N RM RN TILE_M TILE_N TILE_K BLOCK_X BLOCK_Y [ITERS]\n"
            << "Example: " << argv[0]
            << " 1024 1024 1024 2 2 32 32 32 16 16 100\n";
        return EXIT_FAILURE;
    }

    int M, K, N, RM, RN, TILE_M, TILE_N, TILE_K, BLOCK_X, BLOCK_Y;
    int* values[] = {&M, &K, &N, &RM, &RN, &TILE_M, &TILE_N, &TILE_K, &BLOCK_X, &BLOCK_Y};
    for (int i = 0; i < 10; ++i) {
        if (!parse_positive_int(argv[i + 1], *values[i])) {
            std::cerr << "Invalid positive integer argument: " << argv[i + 1] << '\n';
            return EXIT_FAILURE;
        }
    }

    int iters = 100;
    if (argc == 12 && !parse_positive_int(argv[11], iters)) {
        std::cerr << "ITERS must be a positive integer.\n";
        return EXIT_FAILURE;
    }

    if (RM > MAX_REG_TILE || RN > MAX_REG_TILE) {
        std::cerr << "RM and RN must be <= " << MAX_REG_TILE << ".\n";
        return EXIT_FAILURE;
    }
    if (TILE_M != RM * BLOCK_Y || TILE_N != RN * BLOCK_X) {
        std::cerr << "Configuration mismatch: require TILE_M == RM*BLOCK_Y and "
                  << "TILE_N == RN*BLOCK_X.\n";
        return EXIT_FAILURE;
    }

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    if (BLOCK_X * BLOCK_Y > prop.maxThreadsPerBlock ||
        BLOCK_X > prop.maxThreadsDim[0] || BLOCK_Y > prop.maxThreadsDim[1]) {
        std::cerr << "Invalid block dimensions for this GPU.\n";
        return EXIT_FAILURE;
    }

    const size_t shared_bytes =
        (static_cast<size_t>(TILE_M) * TILE_K +
         static_cast<size_t>(TILE_K) * TILE_N) * sizeof(float);
    if (shared_bytes > prop.sharedMemPerBlock) {
        std::cerr << "Requested shared memory (" << shared_bytes
                  << " bytes) exceeds the default per-block limit ("
                  << prop.sharedMemPerBlock << " bytes).\n";
        return EXIT_FAILURE;
    }

    const size_t sizeA = static_cast<size_t>(M) * K;
    const size_t sizeB = static_cast<size_t>(K) * N;
    const size_t sizeC = static_cast<size_t>(M) * N;

    std::vector<float> A(sizeA), B(sizeB), C(sizeC), Cref(sizeC, 0.0f);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& x : A) x = dist(rng);
    for (float& x : B) x = dist(rng);

    // CPU reference: straightforward FP32 GEMM, accumulated in double
    // to make the reference less sensitive to FP32 summation error.
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < K; ++k) {
                sum += static_cast<double>(A[static_cast<size_t>(i) * K + k]) *
                       static_cast<double>(B[static_cast<size_t>(k) * N + j]);
            }
            Cref[static_cast<size_t>(i) * N + j] = static_cast<float>(sum);
        }
    }

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, sizeA * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, sizeB * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, sizeC * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), sizeA * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), sizeB * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(BLOCK_X, BLOCK_Y);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    register_tiled_gemm_kernel<<<grid, block, shared_bytes>>>(
        dA, dB, dC, M, K, N, RM, RN, TILE_M, TILE_N, TILE_K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark only the kernel; allocation, copies, and CPU reference are excluded.
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < 10; ++i) {
        register_tiled_gemm_kernel<<<grid, block, shared_bytes>>>(
            dA, dB, dC, M, K, N, RM, RN, TILE_M, TILE_N, TILE_K);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        register_tiled_gemm_kernel<<<grid, block, shared_bytes>>>(
            dA, dB, dC, M, K, N, RM, RN, TILE_M, TILE_N, TILE_K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const double avg_ms = static_cast<double>(total_ms) / iters;

    CUDA_CHECK(cudaMemcpy(C.data(), dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

    double max_abs_err = 0.0;
    double sum_sq_err = 0.0;
    double max_rel_err = 0.0;
    size_t max_err_idx = 0;
    size_t mismatches_1e_3 = 0;
    for (size_t i = 0; i < sizeC; ++i) {
        const double got = C[i];
        const double ref = Cref[i];
        const double abs_err = std::abs(got - ref);
        const double rel_err = abs_err / std::max(std::abs(ref), 1.0e-6);
        if (abs_err > max_abs_err) {
            max_abs_err = abs_err;
            max_err_idx = i;
        }
        max_rel_err = std::max(max_rel_err, rel_err);
        sum_sq_err += abs_err * abs_err;
        if (abs_err > 1.0e-3) ++mismatches_1e_3;
    }
    const double rmse = std::sqrt(sum_sq_err / static_cast<double>(sizeC));
    const double gflops = (2.0 * static_cast<double>(M) * N * K) / (avg_ms * 1.0e6);
    const double tflops = gflops / 1000.0;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\n=== Register-tiled GEMM benchmark ===\n";
    std::cout << "GPU: " << prop.name << " (compute capability "
              << prop.major << "." << prop.minor << ")\n";
    std::cout << "Matrix shape (M,K,N): " << M << ", " << K << ", " << N << '\n';
    std::cout << "Register tile per thread (RM x RN): " << RM << " x " << RN << '\n';
    std::cout << "Shared-memory tiles: A[" << TILE_M << " x " << TILE_K
              << "], B[" << TILE_K << " x " << TILE_N << "]\n";
    std::cout << "Block dimensions (X x Y): " << BLOCK_X << " x " << BLOCK_Y
              << " (" << BLOCK_X * BLOCK_Y << " threads)\n";
    std::cout << "Grid dimensions (X x Y): " << grid.x << " x " << grid.y << '\n';
    std::cout << "Dynamic shared memory: " << shared_bytes << " bytes/block\n";
    std::cout << "Iterations: " << iters << '\n';
    std::cout << "Average kernel time: " << avg_ms << " ms\n";
    std::cout << "Effective throughput: " << tflops << " TFLOP/s\n";
    std::cout << "Maximum absolute error: " << max_abs_err << '\n';
    std::cout << "Maximum relative error (denominator floor 1e-6): " << max_rel_err << '\n';
    std::cout << "RMSE: " << rmse << '\n';
    std::cout << "Elements with abs error > 1e-3: " << mismatches_1e_3
              << " / " << sizeC << '\n';
    std::cout << "Worst-error index: (" << (max_err_idx / N) << ", "
              << (max_err_idx % N) << "), GPU=" << C[max_err_idx]
              << ", CPU=" << Cref[max_err_idx] << '\n';
    std::cout << "Correctness (max abs error <= 1e-3): "
              << (max_abs_err <= 1.0e-3 ? "PASS" : "FAIL") << '\n';

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return max_abs_err <= 1.0e-3 ? EXIT_SUCCESS : EXIT_FAILURE;
}

