#include <iostream>
#include "sgemm.cuh"

__device__ float warpAllReduce_max(float val) {
    for (int i = 16; i > 0; i >>= 1)
        val = max(__shfl_xor_sync(0xffffffff, val, i, 32), val);
    return val;
}

__device__ float warpAllReduce_sum(float val) {
    for (int i = 16; i > 0; i >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, i, 32);
    return val;
}


#define OFFSET(i, j, N) ((i) * (N) + (j))

__global__ void softmax_warpAllReduce(float *src, float *out, int M, int N) {
    const int BLOCK = 4;

    for (int r = blockIdx.y * BLOCK; r < M; r += gridDim.y * BLOCK) {
        for (int m = threadIdx.y; m < BLOCK; m += blockDim.y) {
            float local_val = 0.0;
            for (int c = threadIdx.x; c < N; c += blockDim.x)
                local_val = src[OFFSET(r + m, c, N)];
            __syncthreads();

            float local_max = warpAllReduce_max(local_val);
            for (int c = threadIdx.x; c < N; c += blockDim.x)
                local_val = exp(local_val - local_max);
            __syncthreads();

            float local_sum = warpAllReduce_sum(local_val);
            for (int c = threadIdx.x; c < N; c += blockDim.x)
                local_val /= local_sum;
            __syncthreads();

            for (int c = threadIdx.x; c < N; c += blockDim.x)
                out[OFFSET(r + m, c, N)] = local_val;
        }
    }
}


int main() {
    int M = 4, N = 32;
    size_t size_src = sizeof(float) * M * N;
    size_t size_out = sizeof(float) * M * N;

    float *src = (float *) malloc(size_src);
    float *out = (float *) malloc(size_out);

    for (int i = 0; i < M * N; i++) src[i] = 1.0;
    src[0] = 2.0;

    float *dsrc, *dout;
    cudaMalloc(&dsrc, size_src);
    cudaMalloc(&dout, size_out);

    cudaMemcpy(dsrc, src, size_src, cudaMemcpyHostToDevice);
    softmax_warpAllReduce<<<1, dim3(32, 4)>>>(dsrc, dout ,M, N);
    cudaMemcpy(out, dout, size_out, cudaMemcpyDeviceToHost);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) std::cout << out[OFFSET(i, j, N)] << " ";
        std::cout << std::endl;
    }

    return 0;
}
