#include <cstdio>
#include <cuda_runtime.h>


__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    // Compute the position in C that a thread is responsible for

    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure we don't access memory outside the bounds of this thread
    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[i * N + y]; // Since A and B are represented as a 1-D matrix
        }
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

int main() {
    int M = 512;
    int N = 512;
    int K = 512;
    float alpha = 1.0f, beta = 1.0f;

    // Allocate memory on host (CPU)
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float); // C is the output of matmul between matrix A and B
    float *h_A = (float*)malloc(size_A); // malloc allocates memory in CPU heap
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);

    // Flatten 2-D matrix into 1-D representation
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = 1.0f;
    }

    for (int i = 0; i < N * K; i++) {
        h_B[i] = 1.0f;
    }

    // Allocate memory on GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A); // cudaMalloc allocates memory on GPU
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // Move data from host to GPU
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Set block and grid dimensions
    dim3 blockDim(32, 32); // Number of threads in a block
    dim3 gridDim((M + 31) / 32, (N + 31) / 32); // Number of blocks in a grid
    sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C); // Instantiate naive matmul

    // Copy results from GPU to host
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    printf("Result at [0]: %f\n", h_C[0]);

    // Free memory in GPU and CPU
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}