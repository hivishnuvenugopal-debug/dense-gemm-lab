// src/tiled_gemm.cu
//
// Generic shared-memory tiled GEMM for row-major matrices:
//   A: M x K, B: K x N, C: M x N
//   C = A * B
//
// Usage:
//   ./build/tiled_gemm M N K tile_x tile_y block_x block_y
//
// tile_x: output tile width and K-reduction tile size
// tile_y: output tile height
// block_x/block_y: CUDA threads per block in x/y; each must be <= its
// corresponding tile dimension. They do not need to equal tile dimensions.
//
// Example:
//   ./build/tiled_gemm 1024 1024 1024 16 16 8 8

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,      \
                         __LINE__, cudaGetErrorString(err__));                \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

__global__ void tiled_gemm(const float* A, const float* B, float* C,
                           int M, int N, int K,
                           int tile_x, int tile_y) {
    extern __shared__ float shared[];
    float* As = shared;                         // tile_y x tile_x
    float* Bs = As + (size_t)tile_y * tile_x;   // tile_x x tile_x

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int block_col = blockIdx.x * tile_x;
    const int block_row = blockIdx.y * tile_y;

    // Threads cooperatively load each shared-memory tile. Strided loops
    // allow block dimensions to be smaller than tile dimensions.
    for (int k0 = 0; k0 < K; k0 += tile_x) {
        for (int r = ty; r < tile_y; r += blockDim.y) {
            for (int kk = tx; kk < tile_x; kk += blockDim.x) {
                const int global_r = block_row + r;
                const int global_k = k0 + kk;
                As[(size_t)r * tile_x + kk] =
                    (global_r < M && global_k < K)
                        ? A[(size_t)global_r * K + global_k]
                        : 0.0f;
            }
        }

        for (int kk = ty; kk < tile_x; kk += blockDim.y) {
            for (int c = tx; c < tile_x; c += blockDim.x) {
                const int global_k = k0 + kk;
                const int global_c = block_col + c;
                Bs[(size_t)kk * tile_x + c] =
                    (global_k < K && global_c < N)
                        ? B[(size_t)global_k * N + global_c]
                        : 0.0f;
            }
        }

        __syncthreads();

        // Each thread computes all output elements assigned to it by
        // striding across the output tile. C stores partial sums after each
        // K tile, avoiding a compile-time limit on per-thread accumulator
        // arrays. The first K tile initializes C; subsequent tiles add to it.
        for (int r = ty; r < tile_y; r += blockDim.y) {
            const int global_r = block_row + r;
            if (global_r >= M) continue;

            for (int c = tx; c < tile_x; c += blockDim.x) {
                const int global_c = block_col + c;
                if (global_c >= N) continue;

                float partial = 0.0f;
                for (int kk = 0; kk < tile_x; ++kk) {
                    partial += As[(size_t)r * tile_x + kk] *
                               Bs[(size_t)kk * tile_x + c];
                }

                const size_t out_idx = (size_t)global_r * N + global_c;
                if (k0 == 0)
                    C[out_idx] = partial;
                else
                    C[out_idx] += partial;
            }
        }

        // Ensure every thread has finished reading shared memory and writing
        // its output partial sums before the next tile is loaded.
        __syncthreads();
    }
}

int main(int argc, char** argv) {
    if (argc != 8) {
        std::fprintf(stderr,
            "Usage: %s M N K tile_x tile_y block_x block_y\n"
            "  A is MxK, B is KxN, C is MxN (row-major floats)\n"
            "  tile_x = output tile width and K tile size\n"
            "  tile_y = output tile height\n"
            "  block_x <= tile_x; block_y <= tile_y\n"
            "Example: %s 1024 1024 1024 16 16 8 8\n",
            argv[0], argv[0]);
        return EXIT_FAILURE;
    }

    char* end = nullptr;
    auto parse_positive_int = [&](int index, const char* name) -> int {
        end = nullptr;
        const long value = std::strtol(argv[index], &end, 10);
        if (end == argv[index] || *end != '\0' || value <= 0 ||
            value > std::numeric_limits<int>::max()) {
            std::fprintf(stderr, "Invalid %s: '%s' (must be a positive integer)\n",
                         name, argv[index]);
            std::exit(EXIT_FAILURE);
        }
        return static_cast<int>(value);
    };

    const int M       = parse_positive_int(1, "M");
    const int N       = parse_positive_int(2, "N");
    const int K       = parse_positive_int(3, "K");
    const int tile_x  = parse_positive_int(4, "tile_x");
    const int tile_y  = parse_positive_int(5, "tile_y");
    const int block_x = parse_positive_int(6, "block_x");
    const int block_y = parse_positive_int(7, "block_y");

    if (block_x > tile_x || block_y > tile_y) {
        std::fprintf(stderr,
            "Invalid block dimensions: require block_x <= tile_x and "
            "block_y <= tile_y.\n");
        return EXIT_FAILURE;
    }

    int device = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    if (block_x > prop.maxThreadsDim[0] ||
        block_y > prop.maxThreadsDim[1] ||
        (long long)block_x * block_y > prop.maxThreadsPerBlock) {
        std::fprintf(stderr,
            "Invalid block dimensions for device %s (max x=%d, max y=%d, "
            "max threads/block=%d).\n",
            prop.name, prop.maxThreadsDim[0], prop.maxThreadsDim[1],
            prop.maxThreadsPerBlock);
        return EXIT_FAILURE;
    }

    const size_t a_tile_elems = (size_t)tile_y * tile_x;
    const size_t b_tile_elems = (size_t)tile_x * tile_x;
    const size_t shared_bytes =
        (a_tile_elems + b_tile_elems) * sizeof(float);

    if (shared_bytes > prop.sharedMemPerBlock) {
        std::fprintf(stderr,
            "Requested shared memory (%zu bytes) exceeds the device's "
            "default per-block limit (%zu bytes).\n",
            shared_bytes, prop.sharedMemPerBlock);
        return EXIT_FAILURE;
    }

    const size_t a_elems = (size_t)M * K;
    const size_t b_elems = (size_t)K * N;
    const size_t c_elems = (size_t)M * N;

    std::vector<float> h_A(a_elems), h_B(b_elems), h_C(c_elems);
    std::vector<float> h_ref(c_elems, 0.0f);

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& x : h_A) x = dist(rng);
    for (float& x : h_B) x = dist(rng);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, a_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, b_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, c_elems * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), a_elems * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), b_elems * sizeof(float),
                          cudaMemcpyHostToDevice));

    const dim3 block(block_x, block_y);
    const dim3 grid((N + tile_x - 1) / tile_x,
                    (M + tile_y - 1) / tile_y);

    auto launch = [&]() {
        tiled_gemm<<<grid, block, shared_bytes>>>(
            d_A, d_B, d_C, M, N, K, tile_x, tile_y);
        CUDA_CHECK(cudaGetLastError());
    };

    // Warm-up before timing.
    for (int i = 0; i < 10; ++i) launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    constexpr int iterations = 100;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; ++i) launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const double avg_ms = total_ms / iterations;
    const double flops = 2.0 * (double)M * N * K;
    const double tflops = flops / (avg_ms * 1.0e-3) / 1.0e12;

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, c_elems * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // CPU reference for correctness.
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            const float a = h_A[(size_t)i * K + k];
            for (int j = 0; j < N; ++j)
                h_ref[(size_t)i * N + j] += a * h_B[(size_t)k * N + j];
        }
    }

    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    bool correct = true;
    constexpr double abs_tol = 1.0e-4;
    constexpr double rel_tol = 1.0e-4;
    for (size_t i = 0; i < c_elems; ++i) {
        const double actual = h_C[i];
        const double expected = h_ref[i];
        const double abs_error = std::abs(actual - expected);
        const double rel_error =
            abs_error / std::max(std::abs(expected), 1.0e-12);
        max_abs_error = std::max(max_abs_error, abs_error);
        max_rel_error = std::max(max_rel_error, rel_error);
        if (abs_error > abs_tol + rel_tol * std::abs(expected))
            correct = false;
    }

    std::printf("Device:                  %s\n", prop.name);
    std::printf("Matrix dimensions:       M=%d N=%d K=%d\n", M, N, K);
    std::printf("Tile dimensions:         tile_x=%d tile_y=%d\n",
                tile_x, tile_y);
    std::printf("Block dimensions:        block_x=%d block_y=%d (%d threads)\n",
                block_x, block_y, block_x * block_y);
    std::printf("Grid dimensions:         (%u, %u)\n", grid.x, grid.y);
    std::printf("Dynamic shared memory:   %zu bytes\n", shared_bytes);
    std::printf("Average kernel time:     %.6f ms (%d iterations)\n",
                avg_ms, iterations);
    std::printf("Effective throughput:    %.6f TFLOP/s\n", tflops);
    std::printf("Max absolute error:      %.8e\n", max_abs_error);
    std::printf("Max relative error:      %.8e\n", max_rel_error);
    std::printf("Correctness:             %s\n", correct ? "PASS" : "FAIL");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}

