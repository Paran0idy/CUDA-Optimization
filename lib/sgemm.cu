#include "sgemm.cuh"
#include <iostream>

// MARCO
#define OFFSET(i, j, N) (i) * (N) + (j)
#define FLOAT4(pointer) reinterpret_cast<float4*>(&(pointer))[0]


// Naive
__global__ void sgemm_v1(float *a, float *b, float*c, int M, int N, int K){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    for(int k = 0; k < K; k++)
        c[OFFSET(row, col, N)] += a[OFFSET(row, k, K)] * b[OFFSET(k, col, N)];
}

// Block Tiling
__global__ void sgemm_v2(float *a, float *b, float *c, int M, int N, int K){

    const int BLOCK_M = 16, BLOCK_N = 16, BLOCK_K = 64;

    __shared__ float shared_a[BLOCK_M][BLOCK_K];
    __shared__ float shared_b[BLOCK_K][BLOCK_N];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int smem_a_m = tid / 16;
    int smem_a_k = (tid % 16) << 2;

    int smem_b_k = tid / 4;
    int smem_b_n = (tid % 4) << 2;

    int gmem_a_m = smem_a_m + blockIdx.y * BLOCK_M;
    int gmem_b_n = smem_b_n + blockIdx.x * BLOCK_N;

    float result = 0.0;
    for(int k = 0; k < K / BLOCK_K; k++){
        // GMEM copy to SMEM
        int gmem_a_k = smem_a_k + k * BLOCK_K;
        int gmem_b_k = smem_b_k + k * BLOCK_K;

        FLOAT4(shared_a[smem_a_m][smem_a_k]) = FLOAT4(a[OFFSET(gmem_a_m, gmem_a_k, K)]);
        FLOAT4(shared_b[smem_b_k][smem_b_n]) = FLOAT4(b[OFFSET(gmem_b_k, gmem_b_n, N)]);

        __syncthreads();

        // Compute
        for(int kk = 0; kk < BLOCK_K; kk++)
            result += shared_a[threadIdx.y][kk] * shared_b[kk][threadIdx.x];
        
        __syncthreads();
    } 

    // Write back
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    c[OFFSET(row, col, N)] = result;
}

// Thread Tiling
__global__ void sgemm_v3(float *a, float *b, float*c, int M, int N, int K){

}

// Warp Tiling
__global__ void sgemm_v4(float *a, float *b, float*c, int M, int N, int K){

}

// Bank Free
__global__ void sgemm_v5(float *a, float *b, float*c, int M, int N, int K){

}

// Pipeline
__global__ void sgemm_v6(float *a, float *b, float*c, int M, int N, int K){

}

// WMMA
__global__ void sgemm_v7(float *a, float *b, float*c, int M, int N, int K){

}

// Auto Tuning



int main(){
    
    //
    int M = 128, N = 128, K = 128;
    size_t size = sizeof(float) * M * N;

    float *a = (float *)malloc(size);
    float *b = (float *)malloc(size);
    float *c = (float *)malloc(size);

    for(int i = 0; i < M * N; i++){
        a[i] = 1.0;
        b[i] = 1.0;
        c[i] = 0.0;
    }

    float *da, *db, *dc;
    cudaMalloc(&da, size);
    cudaMalloc(&db, size);
    cudaMalloc(&dc, size);

    cudaMemcpy(da, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, size, cudaMemcpyHostToDevice);

    const int BLOCK_M = 16, BLOCK_N = 16;
    dim3 grid(N / BLOCK_N, M / BLOCK_M);
    dim3 block(BLOCK_N, BLOCK_M);

    sgemm_v2<<<grid, block>>>(da, db, dc, M, N, K);

    cudaMemcpy(c, dc, size, cudaMemcpyDeviceToHost);

    for(int i = 0; i < M; i++){
        for(int j = 0; j < N; j++)
            std::cout << c[OFFSET(i, j, N)] << " ";
        std::cout << std::endl;
    }

    return 0;
}