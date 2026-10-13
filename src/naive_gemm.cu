#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <climits>
#include <vector>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(err));      \
        std::exit(EXIT_FAILURE);                                    \
    }                                                               \
} while (0)

// C[M,N] = A[M,K] * B[K,N]
// Row-major storage; one CUDA thread computes one C element.
__global__ void naive_gemm_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;

        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = sum;
    }
}

// Parse a positive integer argument.
static int parse_positive_int(const char* text, const char* name)
{
    char* end = nullptr;
    long value = std::strtol(text, &end, 10);

    if (text[0] == '\0' ||
        end == text ||
        *end != '\0' ||
        value <= 0 ||
        value > INT_MAX) {
        fprintf(stderr, "Invalid %s: '%s'. Expected a positive integer.\n",
                name, text);
        std::exit(EXIT_FAILURE);
    }

    return static_cast<int>(value);
}

int main(int argc, char* argv[])
{
    if (argc != 6) {
        fprintf(stderr,
                "Usage: %s M N K block_x block_y\n\n"
                "Example:\n"
                "  %s 257 193 129 16 16\n"
                "  %s 1024 1024 1024 32 8\n",
                argv[0], argv[0], argv[0]);
        return EXIT_FAILURE;
    }

    // Matrix dimensions.
    const int M = parse_positive_int(argv[1], "M");
    const int N = parse_positive_int(argv[2], "N");
    const int K = parse_positive_int(argv[3], "K");

    // CUDA thread-block dimensions.
    const int block_x = parse_positive_int(argv[4], "block_x");
    const int block_y = parse_positive_int(argv[5], "block_y");

    // Validate block dimensions against CUDA device limits.
    int device = 0;
    cudaDeviceProp prop;

    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    if (block_x > prop.maxThreadsDim[0] ||
        block_y > prop.maxThreadsDim[1] ||
        static_cast<long long>(block_x) * block_y >
            prop.maxThreadsPerBlock) {
        fprintf(stderr,
                "Invalid block dimensions: %d x %d.\n"
                "Device limits: x <= %d, y <= %d, "
                "threads per block <= %d.\n",
                block_x, block_y,
                prop.maxThreadsDim[0],
                prop.maxThreadsDim[1],
                prop.maxThreadsPerBlock);
        return EXIT_FAILURE;
    }

    // Calculate the CUDA grid dimensions.
    const int grid_x = static_cast<int>(
        (static_cast<long long>(N) + block_x - 1) / block_x);

    const int grid_y = static_cast<int>(
        (static_cast<long long>(M) + block_y - 1) / block_y);

    const dim3 block(block_x, block_y);
    const dim3 grid(grid_x, grid_y);

    // Matrix sizes.
    const size_t sizeA = static_cast<size_t>(M) * K;
    const size_t sizeB = static_cast<size_t>(K) * N;
    const size_t sizeC = static_cast<size_t>(M) * N;

    // Host memory.
    std::vector<float> A(sizeA);
    std::vector<float> B(sizeB);
    std::vector<float> C(sizeC, 0.0f);
    std::vector<float> C_ref(sizeC, 0.0f);

    // Deterministic input values.
    for (size_t i = 0; i < sizeA; ++i) {
        A[i] = static_cast<float>(
            static_cast<int>(i % 17) - 8) / 17.0f;
    }

    for (size_t i = 0; i < sizeB; ++i) {
        B[i] = static_cast<float>(
            static_cast<int>(i % 13) - 6) / 13.0f;
    }

    // CPU reference.
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;

            for (int k = 0; k < K; ++k) {
                sum += A[static_cast<size_t>(row) * K + k] *
                       B[static_cast<size_t>(k) * N + col];
            }

            C_ref[static_cast<size_t>(row) * N + col] = sum;
        }
    }

    // Allocate device memory.
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, sizeA * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, sizeB * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, sizeC * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(
        d_A, A.data(), sizeA * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_B, B.data(), sizeB * sizeof(float), cudaMemcpyHostToDevice));

    // Initial kernel launch.
    naive_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Warm-up and timing.
    cudaEvent_t start, stop;

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    naive_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const int iterations = 100;

    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < iterations; ++i) {
        naive_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    const double avg_ms = elapsed_ms / iterations;
    const double flops = 2.0 * static_cast<double>(M) * N * K;
    const double tflops = flops / (avg_ms * 1.0e-3) / 1.0e12;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Copy the final result back to the host.
    CUDA_CHECK(cudaMemcpy(
        C.data(), d_C, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

    // Validate against the CPU reference.
    float max_abs_error = 0.0f;
    float max_rel_error = 0.0f;
    size_t mismatch_count = 0;

    for (size_t i = 0; i < sizeC; ++i) {
        const float abs_error = std::fabs(C[i] - C_ref[i]);
        const float rel_error =
            abs_error / (std::fabs(C_ref[i]) + 1.0e-6f);

        if (abs_error > max_abs_error)
            max_abs_error = abs_error;

        if (rel_error > max_rel_error)
            max_rel_error = rel_error;

        if (abs_error > 1.0e-4f)
            ++mismatch_count;
    }

    // Print configuration and results.
    printf("\n========== Naive CUDA GEMM ==========\n");

    printf("GPU                    : %s\n", prop.name);
    printf("Compute capability     : %d.%d\n",
           prop.major, prop.minor);

    printf("\nMatrix dimensions:\n");
    printf("M                      : %d\n", M);
    printf("N                      : %d\n", N);
    printf("K                      : %d\n", K);

    printf("\nCUDA launch configuration:\n");
    printf("Block dimensions       : %d x %d\n", block_x, block_y);
    printf("Threads per block      : %d\n", block_x * block_y);
    printf("Warps per block        : %.2f\n",
           (block_x * block_y) / 32.0);
    printf("Grid dimensions        : %d x %d\n", grid_x, grid_y);
    printf("Total blocks           : %lld\n",
           static_cast<long long>(grid_x) * grid_y);
    printf("Total launched threads : %lld\n",
           static_cast<long long>(grid_x) * grid_y *
           block_x * block_y);

    printf("\nPerformance:\n");
    printf("Iterations             : %d\n", iterations);
    printf("Average kernel time    : %.6f ms\n", avg_ms);
    printf("Effective throughput   : %.6f TFLOP/s\n", tflops);

    printf("\nCorrectness:\n");
    printf("Maximum absolute error : %.8e\n", max_abs_error);
    printf("Maximum relative error : %.8e\n", max_rel_error);
    printf("Elements exceeding tol.: %zu / %zu\n",
           mismatch_count, sizeC);
    printf("Correctness            : %s\n",
           mismatch_count == 0 ? "PASS" : "FAIL");

    printf("=====================================\n");

    // Release device memory.
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return mismatch_count == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
