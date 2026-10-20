// register_tiled_gemm_template.cu
//
// Configurable FP32 GEMM with:
//   * compile-time RM/RN register accumulator tile dimensions
//   * compile-time TILE_K reduction depth
//   * runtime M, K, N
//   * runtime BLOCK_X/BLOCK_Y (validated against compile-time tile dimensions)
//   * shared-memory A/B tiles
//   * CPU double-accumulation reference and numerical/performance metrics
//
// Build:
//   nvcc -O3 -std=c++17 -arch=sm_80 -Xptxas=-v \
//       register_tiled_gemm_template.cu -o build/register_tiled_gemm_template
//
// Usage:
//   ./build/register_tiled_gemm_template M K N RM RN TILE_K BLOCK_X BLOCK_Y [ITERS]
//
// Supported compile-time variants are selected at runtime by dispatching to
// separately instantiated kernels. RM/RN and TILE_K are therefore constants
// inside each kernel specialization, so the compiler can optimize the
// accumulator array and loops for that variant.
//
// Mapping constraints:
//   TILE_M = RM * BLOCK_Y
//   TILE_N = RN * BLOCK_X
//
// Example:
//   ./build/register_tiled_gemm_template 1024 1024 1024 2 2 32 16 16 100

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
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

template <int RM, int RN, int TILE_K>
__global__ void register_tiled_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int K, int N)
{
    // Block output tile dimensions are derived from launch dimensions:
    // TILE_M = RM * blockDim.y; TILE_N = RN * blockDim.x.
    constexpr int MAX_RM = 8;
    constexpr int MAX_RN = 8;
    static_assert(RM >= 1 && RM <= MAX_RM, "RM must be in [1,8]");
    static_assert(RN >= 1 && RN <= MAX_RN, "RN must be in [1,8]");
    static_assert(TILE_K > 0, "TILE_K must be positive");

    const int TILE_M = RM * blockDim.y;
    const int TILE_N = RN * blockDim.x;

    extern __shared__ float smem[];
    float* As = smem; // TILE_M x TILE_K
    float* Bs = As + static_cast<size_t>(TILE_M) * TILE_K; // TILE_K x TILE_N

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int threads = blockDim.x * blockDim.y;

    const int row0 = blockIdx.y * TILE_M + threadIdx.y * RM;
    const int col0 = blockIdx.x * TILE_N + threadIdx.x * RN;

    // Compile-time-sized accumulator tile: optimized per specialization.
    float acc[RM][RN] = {};

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        const size_t a_count = static_cast<size_t>(TILE_M) * TILE_K;
        const size_t b_count = static_cast<size_t>(TILE_K) * TILE_N;

        // Cooperative load of A tile.
        for (size_t idx = tid; idx < a_count; idx += threads) {
            const int r = static_cast<int>(idx / TILE_K);
            const int kk = static_cast<int>(idx % TILE_K);
            const int gr = blockIdx.y * TILE_M + r;
            const int gk = k0 + kk;
            As[idx] = (gr < M && gk < K)
                    ? A[static_cast<size_t>(gr) * K + gk]
                    : 0.0f;
        }

        // Cooperative load of B tile.
        for (size_t idx = tid; idx < b_count; idx += threads) {
            const int kk = static_cast<int>(idx / TILE_N);
            const int c = static_cast<int>(idx % TILE_N);
            const int gk = k0 + kk;
            const int gc = blockIdx.x * TILE_N + c;
            Bs[idx] = (gk < K && gc < N)
                    ? B[static_cast<size_t>(gk) * N + gc]
                    : 0.0f;
        }

        __syncthreads();

        const int valid_k = min(TILE_K, K - k0);
        #pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            if (kk < valid_k) {
                float a_reg[RM];
                float b_reg[RN];

                #pragma unroll
                for (int i = 0; i < RM; ++i)
                    a_reg[i] = As[(threadIdx.y * RM + i) * TILE_K + kk];

                #pragma unroll
                for (int j = 0; j < RN; ++j)
                    b_reg[j] = Bs[kk * TILE_N + threadIdx.x * RN + j];

                #pragma unroll
                for (int i = 0; i < RM; ++i) {
                    #pragma unroll
                    for (int j = 0; j < RN; ++j)
                        acc[i][j] = fmaf(a_reg[i], b_reg[j], acc[i][j]);
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < RM; ++i) {
        #pragma unroll
        for (int j = 0; j < RN; ++j) {
            const int row = row0 + i;
            const int col = col0 + j;
            if (row < M && col < N)
                C[static_cast<size_t>(row) * N + col] = acc[i][j];
        }
    }
}

static bool parse_positive_int(const char* s, int& value) {
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (end == s || *end != '\0' || v <= 0 ||
        v > std::numeric_limits<int>::max()) return false;
    value = static_cast<int>(v);
    return true;
}

template <int RM, int RN, int TILE_K>
static void launch_kernel(
    const float* dA, const float* dB, float* dC,
    int M, int K, int N,
    int BLOCK_X, int BLOCK_Y,
    size_t shared_bytes,
    dim3 grid)
{
    dim3 block(BLOCK_X, BLOCK_Y);
    register_tiled_gemm_kernel<RM, RN, TILE_K>
        <<<grid, block, shared_bytes>>>(dA, dB, dC, M, K, N);
}

int main(int argc, char** argv) {
    if (argc < 9 || argc > 10) {
        std::cerr << "Usage: " << argv[0]
                  << " M K N RM RN TILE_K BLOCK_X BLOCK_Y [ITERS]\n"
                  << "Example: " << argv[0]
                  << " 1024 1024 1024 2 2 32 16 16 100\n";
        return EXIT_FAILURE;
    }

    int M, K, N, RM, RN, TILE_K, BLOCK_X, BLOCK_Y;
    int* vals[] = {&M, &K, &N, &RM, &RN, &TILE_K, &BLOCK_X, &BLOCK_Y};
    for (int i = 0; i < 8; ++i) {
        if (!parse_positive_int(argv[i + 1], *vals[i])) {
            std::cerr << "Invalid positive integer: " << argv[i + 1] << '\n';
            return EXIT_FAILURE;
        }
    }
    int iters = 100;
    if (argc == 10 && !parse_positive_int(argv[9], iters)) {
        std::cerr << "ITERS must be a positive integer.\n";
        return EXIT_FAILURE;
    }

    if (RM > 8 || RN > 8) {
        std::cerr << "RM and RN must be <= 8.\n";
        return EXIT_FAILURE;
    }

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    if (BLOCK_X * BLOCK_Y > prop.maxThreadsPerBlock ||
        BLOCK_X > prop.maxThreadsDim[0] ||
        BLOCK_Y > prop.maxThreadsDim[1]) {
        std::cerr << "Invalid block dimensions for this GPU.\n";
        return EXIT_FAILURE;
    }

    const int TILE_M = RM * BLOCK_Y;
    const int TILE_N = RN * BLOCK_X;
    const size_t shared_bytes =
        (static_cast<size_t>(TILE_M) * TILE_K +
         static_cast<size_t>(TILE_K) * TILE_N) * sizeof(float);

    // sharedMemPerBlock is the default per-block limit. Opt-in larger dynamic
    // shared memory is not requested by this sample.
    if (shared_bytes > prop.sharedMemPerBlock) {
        std::cerr << "Requested shared memory " << shared_bytes
                  << " bytes exceeds default per-block limit "
                  << prop.sharedMemPerBlock << " bytes.\n";
        return EXIT_FAILURE;
    }

    const size_t sizeA = static_cast<size_t>(M) * K;
    const size_t sizeB = static_cast<size_t>(K) * N;
    const size_t sizeC = static_cast<size_t>(M) * N;

    std::vector<float> A(sizeA), B(sizeB), C(sizeC), Cref(sizeC);
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& x : A) x = dist(rng);
    for (float& x : B) x = dist(rng);

    // CPU reference using double accumulation.
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < K; ++k)
                sum += static_cast<double>(A[static_cast<size_t>(i) * K + k]) *
                       static_cast<double>(B[static_cast<size_t>(k) * N + j]);
            Cref[static_cast<size_t>(i) * N + j] = static_cast<float>(sum);
        }
    }

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, sizeA * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, sizeB * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, sizeC * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), sizeA * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), sizeB * sizeof(float), cudaMemcpyHostToDevice));

    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    // Dispatch to a distinct compile-time specialization for every RM/RN/TILE_K
    // choice. BLOCK_X/Y remain launch-time values, with tile dimensions derived
    // from them.
    #define LAUNCH(RM_, RN_, TK_) \
        launch_kernel<RM_, RN_, TK_>(dA, dB, dC, M, K, N, BLOCK_X, BLOCK_Y, shared_bytes, grid)

    bool supported = true;
    #define DISPATCH_RN(RM_) \
        if (RN == 1) LAUNCH(RM_, 1, 32); \
        else if (RN == 2) LAUNCH(RM_, 2, 32); \
        else if (RN == 4) LAUNCH(RM_, 4, 32); \
        else supported = false

    // TILE_K is templated too. This dispatch supports TILE_K=32 to keep the
    // compiled variants explicit and comparable. Other values are rejected.
    if (TILE_K != 32) {
        std::cerr << "This build currently supports TILE_K=32 only (template dispatch).\n";
        CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
        return EXIT_FAILURE;
    }

    if (RM == 1) DISPATCH_RN(1);
    else if (RM == 2) DISPATCH_RN(2);
    else if (RM == 4) DISPATCH_RN(4);
    else supported = false;

    #undef DISPATCH_RN
    #undef LAUNCH

    if (!supported) {
        std::cerr << "Supported template variants: RM in {1,2,4}, RN in {1,2,4}.\n";
        CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
        return EXIT_FAILURE;
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timing dispatch helper uses the same specialized variant as above.
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    auto run_once = [&]() {
        #define LAUNCH(RM_, RN_) \
            launch_kernel<RM_, RN_, 32>(dA, dB, dC, M, K, N, BLOCK_X, BLOCK_Y, shared_bytes, grid)
        if (RM == 1 && RN == 1) LAUNCH(1, 1);
        else if (RM == 1 && RN == 2) LAUNCH(1, 2);
        else if (RM == 1 && RN == 4) LAUNCH(1, 4);
        else if (RM == 2 && RN == 1) LAUNCH(2, 1);
        else if (RM == 2 && RN == 2) LAUNCH(2, 2);
        else if (RM == 2 && RN == 4) LAUNCH(2, 4);
        else if (RM == 4 && RN == 1) LAUNCH(4, 1);
        else if (RM == 4 && RN == 2) LAUNCH(4, 2);
        else if (RM == 4 && RN == 4) LAUNCH(4, 4);
        #undef LAUNCH
    };

    for (int i = 0; i < 10; ++i) run_once();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) run_once();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const double avg_ms = static_cast<double>(total_ms) / iters;

    CUDA_CHECK(cudaMemcpy(C.data(), dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

    double max_abs_err = 0.0, max_rel_err = 0.0, sum_sq_err = 0.0;
    size_t max_idx = 0, count_gt_1e3 = 0;
    for (size_t i = 0; i < sizeC; ++i) {
        const double got = C[i], ref = Cref[i];
        const double abs_err = std::abs(got - ref);
        const double rel_err = abs_err / std::max(std::abs(ref), 1.0e-6);
        if (abs_err > max_abs_err) { max_abs_err = abs_err; max_idx = i; }
        max_rel_err = std::max(max_rel_err, rel_err);
        sum_sq_err += abs_err * abs_err;
        if (abs_err > 1.0e-3) ++count_gt_1e3;
    }
    const double rmse = std::sqrt(sum_sq_err / static_cast<double>(sizeC));
    const double tflops = (2.0 * static_cast<double>(M) * N * K) / (avg_ms * 1.0e9);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\n=== Template register-tiled GEMM benchmark ===\n";
    std::cout << "GPU: " << prop.name << " (compute capability "
              << prop.major << "." << prop.minor << ")\n";
    std::cout << "Matrix shape (M,K,N): " << M << ", " << K << ", " << N << '\n';
    std::cout << "Compile-time register tile (RM x RN): " << RM << " x " << RN << '\n';
    std::cout << "Compile-time TILE_K: " << TILE_K << '\n';
    std::cout << "Derived shared tiles: A[" << TILE_M << " x " << TILE_K
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
    std::cout << "Elements with abs error > 1e-3: " << count_gt_1e3
              << " / " << sizeC << '\n';
    std::cout << "Worst-error index: (" << max_idx / N << ", " << max_idx % N
              << "), GPU=" << C[max_idx] << ", CPU=" << Cref[max_idx] << '\n';
    std::cout << "Correctness (max abs error <= 1e-3): "
              << (max_abs_err <= 1.0e-3 ? "PASS" : "FAIL") << '\n';

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return max_abs_err <= 1.0e-3 ? EXIT_SUCCESS : EXIT_FAILURE;
}

