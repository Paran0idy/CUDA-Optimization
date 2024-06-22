#include <cuda.h>
#include <iostream>

#define OFFSET(i, j, N) (i) * (N) + (j)

// blockTiling
__global__ void sgemv_v1(float *a, float *b, float *c ,int K, int N) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = 0; i < K; i++)
        c[index] += a[i] * b[OFFSET(i, N, index)];
}

// warpAllReduce
__device__ float warpAllReduce(float val) {
    for (int i = 16; i > 0; i >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, i, 32);
    return val;
}

__global__ void sgemv_v2(float *a, float *b, float *c ,int K, int N) {
    int warpNum = blockDim.x >> 5;
    int warpId = threadIdx.x >> 5;
    int laneId = threadIdx.x % 32;

    for (int w = warpId; w < N; w += warpNum) {
        float val = 0;
        for (int k = laneId; k < K; k += 32)
            val += a[k] * b[OFFSET(k, N, w)];
        // warpAllReduce
        val = warpAllReduce(val);
        if (laneId == 0) c[w] = val;
    }
}

int main() {
    const int K = 32, N = 1024;
    const int BLOCK_SIZE = 128;

    int size_a = sizeof(float) * K;
    int size_b = sizeof(float) * K * N;
    int size_c = sizeof(float) * N;

    float *a = (float *) malloc(size_a);
    float *b = (float *) malloc(size_b);
    float *c = (float *) malloc(size_c);
    for (int i = 0; i < K; i++) a[i] = 1.0;
    for (int i = 0; i < K * N; i++) b[i] = 1.0;

    float *da, *db, *dc;
    cudaMalloc(&da, size_a);
    cudaMalloc(&db, size_b);
    cudaMalloc(&dc, size_c);

    dim3 block(N / BLOCK_SIZE);
    dim3 thread(BLOCK_SIZE);

    cudaMemcpy(da, a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, size_b, cudaMemcpyHostToDevice);

    sgemv_v2<<<block, thread>>>(da, db, dc, K, N);

    cudaMemcpy(c, dc, size_c, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
        std::cout << c[i] << " ";
    std::cout << std::endl;

    return 0;
}