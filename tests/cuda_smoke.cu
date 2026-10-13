#include <cuda_runtime.h>
#include <cstdio>

#define CUDA_CHECK(call) do {                                  \
    cudaError_t err = (call);                                  \
    if (err != cudaSuccess) {                                  \
        fprintf(stderr, "CUDA error: %s\n",                    \
                cudaGetErrorString(err));                     \
        return 1;                                               \
    }                                                           \
} while (0)

__global__ void test_kernel(int *result) {
    result[0] = 42;
}

int main() {
    int device = 0;
    cudaDeviceProp prop;

    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("GPU: %s\n", prop.name);
    printf("Compute capability: %d.%d\n",
           prop.major, prop.minor);
    printf("Global memory: %.2f GiB\n",
           prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

    int *d_result = nullptr;
    int h_result = 0;

    CUDA_CHECK(cudaMalloc(&d_result, sizeof(int)));

    test_kernel<<<1, 1>>>(d_result);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(int),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_result));

    if (h_result != 42) {
        fprintf(stderr, "FAIL: unexpected result %d\n", h_result);
        return 1;
    }

    printf("Kernel result: %d\n", h_result);
    printf("CUDA smoke test: PASS\n");

    return 0;
}
