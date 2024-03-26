#include "sgemm.cuh"
#include <iostream>
#include <cublas_v2.h>
#include <functional>
#include <random>


// MARCO
#define OFFSET(i, j, N) (i) * (N) + (j)
#define FLOAT4(pointer) reinterpret_cast<float4*>(&(pointer))[0]

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            printf("CUDA Error: \n");                                          \
            printf("    File:       %s\n", __FILE__);                          \
            printf("    Line:       %d\n", __LINE__);                          \
            printf("    Error Code: %d\n", err);                               \
            printf("    Error Text: %s\n", cudaGetErrorString(err));           \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t err = call;                                             \
        if (err != CUBLAS_STATUS_SUCCESS) {                                    \
            printf("cuBLAS Error: \n");                                        \
            printf("    File:       %s\n", __FILE__);                          \
            printf("    Line:       %d\n", __LINE__);                          \
            printf("    Error Code: %d\n", err);                               \
            printf("    Error Text: %s\n", cublasGetStatusString(err));        \
            exit(1);                                                           \
        }                                                                      \
    } while (0)


// Naive
__global__ void sgemm_v1_kernel(float *a, float *b, float*c, int M, int N, int K){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    for(int k = 0; k < K; k++)
        c[OFFSET(row, col, N)] += a[OFFSET(row, k, K)] * b[OFFSET(k, col, N)];
}

float sgemm_v1(float *a, float *b, float *c, int M, int N, int K){
    size_t size_a = sizeof(float) * M * K;
    size_t size_b = sizeof(float) * K * N;
    size_t size_c = sizeof(float) * M * N;

    float *da, *db, *dc;
    cudaMalloc(&da, size_a);
    cudaMalloc(&db, size_b);
    cudaMalloc(&dc, size_c);

    cudaMemcpy(da, a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, size_b, cudaMemcpyHostToDevice);

    const int BLOCK_M = 16, BLOCK_N = 16;
    dim3 grid(N / BLOCK_N, M / BLOCK_M);
    dim3 block(BLOCK_N, BLOCK_M);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    sgemm_v1_kernel<<<grid, block>>>(da, db, dc, M, N, K);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }
    cudaDeviceSynchronize();    
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);

    float msecond = 0.0;
    cudaEventElapsedTime(&msecond, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(c, dc, size_c, cudaMemcpyDeviceToHost);

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    return msecond;
}



// Block Tiling
__global__ void sgemm_v2_kernel(float *a, float *b, float *c, int M, int N, int K){

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

float sgemm_v2(float *a, float *b, float *c, int M, int N, int K){
    size_t size_a = sizeof(float) * M * K;
    size_t size_b = sizeof(float) * K * N;
    size_t size_c = sizeof(float) * M * N;


    float *da, *db, *dc;
    cudaMalloc(&da, size_a);
    cudaMalloc(&db, size_b);
    cudaMalloc(&dc, size_c);

    cudaMemcpy(da, a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, size_b, cudaMemcpyHostToDevice);

    const int BLOCK_M = 16, BLOCK_N = 16;
    dim3 grid(N / BLOCK_N, M / BLOCK_M);
    dim3 block(BLOCK_N, BLOCK_M);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    sgemm_v2_kernel<<<grid, block>>>(da, db, dc, M, N, K);

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);


    float msecond = 0.0;
    cudaEventElapsedTime(&msecond, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(c, dc, size_c, cudaMemcpyDeviceToHost);

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    return msecond;
}

// Thread Tiling
__global__ void sgemm_v3_kernel(float *a, float *b, float*c, int M, int N, int K){
    const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 8;
    const int TILE = 8;

    __shared__ float shared_a[BLOCK_M][BLOCK_K];
    __shared__ float shared_b[BLOCK_K][BLOCK_N];

    float reg_a[TILE];
    float reg_b[TILE];
    float reg_c[TILE][TILE];


    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int smem_a_m = tid / 2;
    int smem_a_k = (tid % 2) << 2;

    int smem_b_k = tid / 32;
    int smem_b_n = (tid % 32) << 2;

    int gmem_a_m = smem_a_m + blockIdx.y * BLOCK_M;
    int gmem_b_n = smem_b_n + blockIdx.x * BLOCK_N;

    for(int k = 0; k < K / BLOCK_K; k++){
        int gmem_a_k = smem_a_k + k * BLOCK_K;
        int gmem_b_k = smem_b_k + k * BLOCK_K;

        // Copy
        FLOAT4(shared_a[smem_a_m][smem_a_k]) = FLOAT4(a[OFFSET(gmem_a_m, gmem_a_k, K)]);
        FLOAT4(shared_b[smem_b_k][smem_b_n]) = FLOAT4(b[OFFSET(gmem_b_k, gmem_b_n, N)]);
        __syncthreads();

        // Compute
        int ty = threadIdx.y * TILE;
        int tx = threadIdx.x * TILE;

        for(int kk = 0; kk < BLOCK_K; kk++){
            for(int i = 0; i < TILE; i++) {
                reg_a[i] = shared_a[ty + i][kk];
                reg_b[i] = shared_b[kk][tx + i];
            }
            for(int i = 0; i < TILE; i++)
                for(int j = 0; j < TILE; j++)
                    reg_c[i][j] += reg_a[i] * reg_b[j];
        }
        __syncthreads();
    }

    // Write Back
    int row = blockIdx.y * BLOCK_M + threadIdx.y * TILE;
    int col = blockIdx.x * BLOCK_N + threadIdx.x * TILE;

    for(int i = 0; i < TILE; i++)
        for(int j = 0; j < TILE; j += 4)
            FLOAT4(c[OFFSET(row + i, col + j, N)]) = FLOAT4(reg_c[i][j]); 
}
float sgemm_v3(float *a, float *b, float *c, int M, int N, int K){
    size_t size_a = sizeof(float) * M * K;
    size_t size_b = sizeof(float) * K * N;
    size_t size_c = sizeof(float) * M * N;


    float *da, *db, *dc;
    cudaMalloc(&da, size_a);
    cudaMalloc(&db, size_b);
    cudaMalloc(&dc, size_c);

    cudaMemcpy(da, a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, size_b, cudaMemcpyHostToDevice);

    const int BLOCK_M = 128, BLOCK_N = 128, TILE = 8;
    dim3 grid(N / BLOCK_N, M / BLOCK_M);
    dim3 block(BLOCK_N / TILE, BLOCK_M / TILE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    sgemm_v3_kernel<<<grid, block>>>(da, db, dc, M, N, K);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);


    float msecond = 0.0;
    cudaEventElapsedTime(&msecond, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(c, dc, size_c, cudaMemcpyDeviceToHost);

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    return msecond;
}

// Warp Tiling
__global__ void sgemm_v4_kernel(float *a, float *b, float*c, int M, int N, int K){
    const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 8;
    const int TILE = 8, NUM = 4;

    __shared__ float shared_a[BLOCK_M][BLOCK_K];
    __shared__ float shared_b[BLOCK_K][BLOCK_N];

    float reg_a[TILE];
    float reg_b[TILE];
    float reg_c[TILE][TILE] = {0};

    int tid = threadIdx.y * blockDim.x + threadIdx.x;


    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    int CTA_Layout[2] = {4, 2};
    int WARP_Layout[2] = {4, 8};

    int WARP_M = BLOCK_M / CTA_Layout[0];
    int WARP_N = BLOCK_N / CTA_Layout[1];

    int warp_y = warp_id / CTA_Layout[1];
    int warp_x = warp_id % CTA_Layout[1];

    int lane_y = lane_id / WARP_Layout[1];
    int lane_x = lane_id % WARP_Layout[1];

    int smem_a_m = tid / 2;
    int smem_a_k = (tid % 2) << 2;

    int smem_b_k = tid / 32;
    int smem_b_n = (tid % 32) << 2;

    int gmem_a_m = smem_a_m + blockIdx.y * BLOCK_M;
    int gmem_b_n = smem_b_n + blockIdx.x * BLOCK_N;

    for(int k = 0; k < K / BLOCK_K; k++){
        int gmem_a_k = smem_a_k + k * BLOCK_K;
        int gmem_b_k = smem_b_k + k * BLOCK_K;

        // Copy
        FLOAT4(shared_a[smem_a_m][smem_a_k]) = FLOAT4(a[OFFSET(gmem_a_m, gmem_a_k, K)]);
        FLOAT4(shared_b[smem_b_k][smem_b_n]) = FLOAT4(b[OFFSET(gmem_b_k, gmem_b_n, N)]);
        __syncthreads();

        // Compute
        int wy = warp_y * WARP_M;
        int wx = warp_x * WARP_N;
        
        for(int kk = 0; kk < BLOCK_K; kk++){
            int ty = wy + lane_y * TILE;
            int tx = wx + lane_x * NUM;

            for(int i = 0; i < TILE; i++) 
                reg_a[i] = shared_a[ty + i][kk];
            
            for(int per = 0; per < 2; per++)
                for(int i = 0; i < NUM; i++)
                    reg_b[i + per * NUM] = shared_b[kk][tx + i + per * NUM];
            
            // Compute NUM = 4
            for(int i = 0; i < TILE; i++)
                for(int j = 0; j < TILE; j++)
                    reg_c[i][j] += reg_a[i] * reg_b[j];
        }
        __syncthreads();
    }

    // Write Back
    int row = blockIdx.y * BLOCK_M + warp_y * WARP_M + lane_y * TILE;
    int col = blockIdx.x * BLOCK_N + warp_x * WARP_N + lane_x * NUM;


    for(int m = 0; m < TILE; m++){
        int r_c_1 = row + m;
        int c_c_1 = col;
        int c_c_2 = c_c_1 + WARP_N / 2;
        FLOAT4(c[OFFSET(r_c_1, c_c_1, N)]) = FLOAT4(reg_c[m][0]);
        FLOAT4(c[OFFSET(r_c_1, c_c_2, N)]) = FLOAT4(reg_c[m][NUM]);
    }
}

float sgemm_v4(float *a, float *b, float *c, int M, int N, int K){
    size_t size_a = sizeof(float) * M * K;
    size_t size_b = sizeof(float) * K * N;
    size_t size_c = sizeof(float) * M * N;


    float *da, *db, *dc;
    cudaMalloc(&da, size_a);
    cudaMalloc(&db, size_b);
    cudaMalloc(&dc, size_c);

    cudaMemcpy(da, a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, size_b, cudaMemcpyHostToDevice);

    const int BLOCK_M = 128, BLOCK_N = 128, TILE = 8;
    dim3 grid(N / BLOCK_N, M / BLOCK_M);
    dim3 block(BLOCK_N / TILE, BLOCK_M / TILE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    sgemm_v4_kernel<<<grid, block>>>(da, db, dc, M, N, K);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);


    float msecond = 0.0;
    cudaEventElapsedTime(&msecond, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(c, dc, size_c, cudaMemcpyDeviceToHost);


    // for(int i = 0; i < M; i++){
    //     for(int j = 0; j < N; j++)
    //         std::cout << c[OFFSET(i, j, N)] << " ";
    //     std::cout << std::endl;
    // }

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    return msecond;
}

// Bank Free
__global__ void sgemm_v5_kernel(float *a, float *b, float*c, int M, int N, int K){
    const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 8;
    const int TILE = 8, NUM = 4;

    __shared__ float shared_a[BLOCK_M][BLOCK_K];
    __shared__ float shared_b[BLOCK_K][BLOCK_N];

    float reg_a[TILE];
    float reg_b[TILE];
    float reg_c[TILE][TILE] = {0};

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int smem_a_m = tid / 2;
    int smem_a_k = (tid % 2) << 2;

    int smem_b_k = tid / 32;
    int smem_b_n = (tid % 32) << 2;

    int gmem_a_m = smem_a_m + blockIdx.y * BLOCK_M;
    int gmem_b_n = smem_b_n + blockIdx.x * BLOCK_N;

    for(int k = 0; k < K / BLOCK_K; k++){
        int gmem_a_k = smem_a_k + k * BLOCK_K;
        int gmem_b_k = smem_b_k + k * BLOCK_K;

        // Copy
        FLOAT4(shared_a[smem_a_m][smem_a_k]) = FLOAT4(a[OFFSET(gmem_a_m, gmem_a_k, K)]);
        FLOAT4(shared_b[smem_b_k][smem_b_n]) = FLOAT4(b[OFFSET(gmem_b_k, gmem_b_n, N)]);
        __syncthreads();

        // Compute
        for(int kk = 0; kk < BLOCK_K; kk++){
            int ty = threadIdx.y * TILE;
            int tx = threadIdx.x * NUM;

            for(int i = 0; i < TILE; i++) 
                reg_a[i] = shared_a[ty + i][kk];
            
            for(int per = 0; per < 2; per++)
                for(int i = 0; i < NUM; i++)
                    reg_b[i + per * NUM] = shared_b[kk][tx + i + per * BLOCK_N / 2];
            
            // Compute NUM = 4
            for(int i = 0; i < TILE; i++)
                for(int j = 0; j < TILE; j++)
                    reg_c[i][j] += reg_a[i] * reg_b[j];
        }
        __syncthreads();
    }

    // Write Back
    int row = blockIdx.y * BLOCK_M + threadIdx.y * TILE;
    int col = blockIdx.x * BLOCK_N + threadIdx.x * NUM;


    for(int m = 0; m < TILE; m++){
        int r_c_1 = row + m;
        int c_c_1 = col;
        int c_c_2 = c_c_1 + BLOCK_N / 2;
        FLOAT4(c[OFFSET(r_c_1, c_c_1, N)]) = FLOAT4(reg_c[m][0]);
        FLOAT4(c[OFFSET(r_c_1, c_c_2, N)]) = FLOAT4(reg_c[m][NUM]);
    }
}

float sgemm_v5(float *a, float *b, float *c, int M, int N, int K){
    size_t size_a = sizeof(float) * M * K;
    size_t size_b = sizeof(float) * K * N;
    size_t size_c = sizeof(float) * M * N;


    float *da, *db, *dc;
    cudaMalloc(&da, size_a);
    cudaMalloc(&db, size_b);
    cudaMalloc(&dc, size_c);

    cudaMemcpy(da, a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, size_b, cudaMemcpyHostToDevice);

    const int BLOCK_M = 128, BLOCK_N = 128, TILE = 8;
    dim3 grid(N / BLOCK_N, M / BLOCK_M);
    dim3 block(BLOCK_N / TILE, BLOCK_M / TILE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    sgemm_v5_kernel<<<grid, block>>>(da, db, dc, M, N, K);
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);


    float msecond = 0.0;
    cudaEventElapsedTime(&msecond, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(c, dc, size_c, cudaMemcpyDeviceToHost);


    // for(int i = 0; i < M; i++){
    //     for(int j = 0; j < N; j++)
    //         std::cout << c[OFFSET(i, j, N)] << " ";
    //     std::cout << std::endl;
    // }

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    return msecond;
}



// Pipeline
__global__ void sgemm_v6(float *a, float *b, float*c, int M, int N, int K){

}

// WMMA
__global__ void sgemm_v7(float *a, float *b, float*c, int M, int N, int K){

}

// float sgemm_cublas(float *a, float *b, float *c, int M, int N, int K){

//     cublasHandle_t handle;
//     CUBLAS_CHECK(cublasCreate(&handle));

//     float cublas_alpha = 1.0;
//     float cublas_beta = 0;

//     size_t size_a = sizeof(float) * M * K;
//     size_t size_b = sizeof(float) * K * N;
//     size_t size_c = sizeof(float) * M * N;

//     float *da, *db, *dc;
//     CUDA_CHECK(cudaMalloc(&da, size_a));
//     CUDA_CHECK(cudaMalloc(&db, size_b));
//     CUDA_CHECK(cudaMalloc(&dc, size_c));

//     cudaMemcpy(da, a, size_a, cudaMemcpyHostToDevice);
//     cudaMemcpy(db, b, size_b, cudaMemcpyHostToDevice);

//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);

//     cudaEventRecord(start);
    
//     cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &cublas_alpha, db, N, da, K, &cublas_beta, dc, N);

//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);


//     float msecond = 0.0;
//     cudaEventElapsedTime(&msecond, start, stop);
    
//     cudaMemcpy(c, dc, size_c, cudaMemcpyDeviceToHost);

//     cudaFree(da);
//     cudaFree(db);
//     cudaFree(dc);
    
//     cudaEventDestroy(start);
//     cudaEventDestroy(stop);

//     cublasDestroy(handle);

//     return msecond;
// }

using Func = std::function<float(float *, float *, float *c, int, int, int)>;

void data_init(float *data, const int num) {
    std::uniform_real_distribution<float> float_gen(-1.0f, 1.0f);
    std::default_random_engine rand_engine(time(nullptr));
    for (int i = 0; i < num; i++) {
        data[i] = float_gen(rand_engine);
    }
}

void testPerformance(Func func, int M, int N, int K, int nums){
    size_t size_a = sizeof(float) * M * K;
    size_t size_b = sizeof(float) * K * N;
    size_t size_c = sizeof(float) * M * N;

    float *a = (float*)malloc(size_a);
    float *b = (float*)malloc(size_b);
    float *c = (float*)malloc(size_c);

    // data_init(a, M * K);
    // data_init(b, K * N);
    for(int i = 0; i < M * K; i++) a[i] = 1.0;
    for(int i = 0; i < K * N; i++) b[i] = 1.0;

    float avg = 0;
    for(int i = 0; i < nums; i++)
        avg += func(a, b, c, M, N, K) / nums;
    
    free(a);
    free(b);
    free(c);

    std::cout << avg << std::endl;
}





int main(){
    int M = 1024, N = 1024, K = 1024;

    testPerformance(sgemm_v1, M, N, K, 10);
    testPerformance(sgemm_v2, M, N, K, 10);
    testPerformance(sgemm_v3, M, N, K, 10);
    testPerformance(sgemm_v4, M, N, K, 10);
    testPerformance(sgemm_v5, M, N, K, 10);


    // testPerformance(sgemm_cublas, M, N, K, 10);


    return 0;
}