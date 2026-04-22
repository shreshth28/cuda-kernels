#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024

__global__ void matmul_naive(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float acc = 0.0f;
        for (int k = 0; k < n; k++) {
            acc += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = acc;
    }
}

int main() {
    const int n = N;
    const size_t bytes = (size_t)n * n * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    srand(42);
    for (int i = 0; i < n * n; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((n + 15) / 16, (n + 15) / 16);

    // warmup pass
    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_naive<<<grid, block>>>(d_A, d_B, d_C, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    double flops  = 2.0 * n * n * n;
    double gflops = flops / ((double)ms / 1000.0) / 1e9;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("GPU:        %s\n",    prop.name);
    printf("Matrix:     %dx%d\n", n, n);
    printf("Time:       %.3f ms\n", ms);
    printf("GFLOPS:     %.2f\n",  gflops);
    printf("C[0][0]:    %.6f\n",  h_C[0]);
    printf("C[N/2][N/2]: %.6f\n", h_C[(n / 2) * n + (n / 2)]);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
// test
