#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace nvcuda;

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = (call);                                       \
        if (err != cudaSuccess) {                                       \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)       \
                      << " at " << __FILE__ << ":" << __LINE__ << '\n'; \
            std::exit(EXIT_FAILURE);                                    \
        }                                                               \
    } while (0)

// One warp computes one 16x16 output tile.
__global__ void wmma_gemm(
    const half* A,
    const half* B,
    float* C,
    int M,
    int N,
    int K)
{
    const int tile_row = blockIdx.y;
    const int tile_col = blockIdx.x;

    // Tensor Core matrix fragments.
    wmma::fragment<
        wmma::matrix_a, 16, 16, 16,
        half, wmma::row_major> a_frag;

    wmma::fragment<
        wmma::matrix_b, 16, 16, 16,
        half, wmma::row_major> b_frag;

    wmma::fragment<
        wmma::accumulator, 16, 16, 16,
        float> c_frag;

    // Start with C_tile = 0.
    wmma::fill_fragment(c_frag, 0.0f);

    // Accumulate A_tile * B_tile over the K dimension.
    for (int k = 0; k < K; k += 16) {
        const half* a_ptr =
            A + tile_row * 16 * K + k;

        const half* b_ptr =
            B + k * N + tile_col * 16;

        wmma::load_matrix_sync(a_frag, a_ptr, K);
        wmma::load_matrix_sync(b_frag, b_ptr, N);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store the FP32 output tile.
    float* c_ptr =
        C + tile_row * 16 * N + tile_col * 16;

    wmma::store_matrix_sync(
        c_ptr, c_frag, N, wmma::mem_row_major);
}

int main(int argc, char** argv)
{
    int M = 256;
    int N = 256;
    int K = 256;

    if (argc == 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    } else if (argc != 1) {
        std::cerr << "Usage: " << argv[0]
                  << " [M N K]\n";
        return EXIT_FAILURE;
    }

    // This baseline handles dimensions divisible by 16.
    if (M <= 0 || N <= 0 || K <= 0 ||
        M % 16 != 0 || N % 16 != 0 || K % 16 != 0) {
        std::cerr
            << "M, N, and K must be positive multiples of 16.\n";
        return EXIT_FAILURE;
    }

    int device = 0;
    cudaDeviceProp prop{};

    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    if (prop.major < 7) {
        std::cerr << "This WMMA Tensor Core example requires "
                  << "a compatible GPU (compute capability >= 7.0).\n";
        return EXIT_FAILURE;
    }

    std::cout << "GPU: " << prop.name << '\n';
    std::cout << "Matrix shape (M,K,N): "
              << M << ", " << K << ", " << N << '\n';
    std::cout << "Input type: FP16\n";
    std::cout << "Accumulator/output type: FP32\n";

    const size_t size_A = static_cast<size_t>(M) * K;
    const size_t size_B = static_cast<size_t>(K) * N;
    const size_t size_C = static_cast<size_t>(M) * N;

    std::vector<half> h_A(size_A);
    std::vector<half> h_B(size_B);
    std::vector<float> h_C(size_C);
    std::vector<float> h_ref(size_C);

    // Deterministic input generation.
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < size_A; ++i)
        h_A[i] = __float2half(dist(rng));

    for (size_t i = 0; i < size_B; ++i)
        h_B[i] = __float2half(dist(rng));

    half *d_A = nullptr, *d_B = nullptr;
    float* d_C = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, size_B * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, size_C * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        d_A, h_A.data(), size_A * sizeof(half),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_B, h_B.data(), size_B * sizeof(half),
        cudaMemcpyHostToDevice));

    dim3 block(32, 1, 1);
    dim3 grid(N / 16, M / 16, 1);

    // Warm-up and initial error check.
    wmma_gemm<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Time repeated kernel executions using CUDA events.
    constexpr int iterations = 100;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < iterations; ++i)
        wmma_gemm<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    const float avg_ms = elapsed_ms / iterations;

    CUDA_CHECK(cudaMemcpy(
        h_C.data(), d_C, size_C * sizeof(float),
        cudaMemcpyDeviceToHost));

    // CPU reference using the same FP16-rounded input values.
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;

            for (int k = 0; k < K; ++k) {
                const float a =
                    __half2float(h_A[static_cast<size_t>(i) * K + k]);

                const float b =
                    __half2float(h_B[static_cast<size_t>(k) * N + j]);

                sum += a * b;
            }

            h_ref[static_cast<size_t>(i) * N + j] = sum;
        }
    }

    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;
    bool correct = true;

    constexpr float atol = 1.0e-2f;
    constexpr float rtol = 1.0e-2f;

    for (size_t i = 0; i < size_C; ++i) {
        const float actual = h_C[i];
        const float expected = h_ref[i];

        const float abs_error = std::fabs(actual - expected);
        const float rel_error =
            abs_error / std::max(std::fabs(expected), 1.0e-6f);

        max_abs_error = std::max(max_abs_error, abs_error);
        max_rel_error = std::max(max_rel_error, rel_error);

        if (abs_error > atol + rtol * std::fabs(expected))
            correct = false;
    }

    // Two floating-point operations per multiply-accumulate.
    const double flops =
        2.0 * static_cast<double>(M) * N * K;

    const double tflops =
        flops / (static_cast<double>(avg_ms) * 1.0e9);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Average kernel time: "
              << avg_ms << " ms\n";

    std::cout << "Effective throughput: "
              << tflops << " TFLOP/s\n";

    std::cout << "Max absolute error: "
              << max_abs_error << '\n';

    std::cout << "Max relative error: "
              << max_rel_error << '\n';

    std::cout << "Correctness: "
              << (correct ? "PASS" : "FAIL") << '\n';

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
