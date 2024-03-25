#include "sgemm.cuh"
#include <iostream>
#include <cublas_v2.h>

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


// // Naive
// __global__ void sgemm_v1_kernel(float *a, float *b, float*c, int M, int N, int K){
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;

//     for(int k = 0; k < K; k++)
//         c[OFFSET(row, col, N)] += a[OFFSET(row, k, K)] * b[OFFSET(k, col, N)];
// }

// void sgemm_v1(float *a, float *b, float *c, int M, int N, int K){
//     size_t size_a = sizeof(float) * M * K;
//     size_t size_b = sizeof(float) * K * N;
//     size_t size_c = sizeof(float) * M * N;

//     float *da, *db, *dc;
//     cudaMalloc(&da, size_a);
//     cudaMalloc(&db, size_b);
//     cudaMalloc(&dc, size_c);

//     cudaMemcpy(da, a, size_a, cudaMemcpyHostToDevice);
//     cudaMemcpy(db, b, size_b, cudaMemcpyHostToDevice);

//     const int BLOCK_M = 16, BLOCK_N = 16;
//     dim3 grid(N / BLOCK_N, M / BLOCK_M);
//     dim3 block(BLOCK_N, BLOCK_M);

//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);

//     cudaEventRecord(start, 0);
//     sgemm_v1_kernel<<<grid, block>>>(da, db, dc, M, N, K);
//     cudaError_t error = cudaGetLastError();
//     if (error != cudaSuccess) {
//         std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
//     }
//     cudaDeviceSynchronize();    
//     cudaEventRecord(stop, 0);

//     cudaEventSynchronize(stop);

//     float msecond = 0.0;
//     cudaEventElapsedTime(&msecond, start, stop);

//     cudaEventDestroy(start);
//     cudaEventDestroy(stop);

//     std::cout << msecond << std::endl;

//     cudaMemcpy(c, dc, size_c, cudaMemcpyDeviceToHost);
//     // for(int i = 0; i < M; i++){
//     //     for(int j = 0; j < N; j++)
//     //         std::cout << c[OFFSET(i, j, N)] << " ";
//     //     std::cout << std::endl;
//     // }

//     cudaFree(da);
//     cudaFree(db);
//     cudaFree(dc);
// }



// // Block Tiling
// __global__ void sgemm_v2_kernel(float *a, float *b, float *c, int M, int N, int K){

//     const int BLOCK_M = 16, BLOCK_N = 16, BLOCK_K = 64;

//     __shared__ float shared_a[BLOCK_M][BLOCK_K];
//     __shared__ float shared_b[BLOCK_K][BLOCK_N];

//     int tid = threadIdx.y * blockDim.x + threadIdx.x;

//     int smem_a_m = tid / 16;
//     int smem_a_k = (tid % 16) << 2;

//     int smem_b_k = tid / 4;
//     int smem_b_n = (tid % 4) << 2;

//     int gmem_a_m = smem_a_m + blockIdx.y * BLOCK_M;
//     int gmem_b_n = smem_b_n + blockIdx.x * BLOCK_N;

//     float result = 0.0;
//     for(int k = 0; k < K / BLOCK_K; k++){
//         // GMEM copy to SMEM
//         int gmem_a_k = smem_a_k + k * BLOCK_K;
//         int gmem_b_k = smem_b_k + k * BLOCK_K;

//         FLOAT4(shared_a[smem_a_m][smem_a_k]) = FLOAT4(a[OFFSET(gmem_a_m, gmem_a_k, K)]);
//         FLOAT4(shared_b[smem_b_k][smem_b_n]) = FLOAT4(b[OFFSET(gmem_b_k, gmem_b_n, N)]);

//         __syncthreads();

//         // Compute
//         for(int kk = 0; kk < BLOCK_K; kk++)
//             result += shared_a[threadIdx.y][kk] * shared_b[kk][threadIdx.x];
        
//         __syncthreads();
//     } 

//     // Write back
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;

//     c[OFFSET(row, col, N)] = result;
// }

// void sgemm_v2(float *a, float *b, float *c, int M, int N, int K){
//     size_t size_a = sizeof(float) * M * K;
//     size_t size_b = sizeof(float) * K * N;
//     size_t size_c = sizeof(float) * M * N;


//     float *da, *db, *dc;
//     cudaMalloc(&da, size_a);
//     cudaMalloc(&db, size_b);
//     cudaMalloc(&dc, size_c);

//     cudaMemcpy(da, a, size_a, cudaMemcpyHostToDevice);
//     cudaMemcpy(db, b, size_b, cudaMemcpyHostToDevice);

//     const int BLOCK_M = 16, BLOCK_N = 16;
//     dim3 grid(N / BLOCK_N, M / BLOCK_M);
//     dim3 block(BLOCK_N, BLOCK_M);


//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);

//     cudaEventRecord(start, 0);
//     sgemm_v2_kernel<<<grid, block>>>(da, db, dc, M, N, K);

//     cudaEventRecord(stop, 0);

//     cudaEventSynchronize(stop);


//     float msecond = 0.0;
//     cudaEventElapsedTime(&msecond, start, stop);
    
//     cudaEventDestroy(start);
//     cudaEventDestroy(stop);

//     std::cout << msecond << std::endl;

//     cudaMemcpy(c, dc, size_c, cudaMemcpyDeviceToHost);

//     cudaFree(da);
//     cudaFree(db);
//     cudaFree(dc);
// }

// // Thread Tiling
// __global__ void sgemm_v3_kernel(float *a, float *b, float*c, int M, int N, int K){
//     const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 8;
//     const int TILE = 8;

//     __shared__ float shared_a[BLOCK_M][BLOCK_K];
//     __shared__ float shared_b[BLOCK_K][BLOCK_N];

//     float reg_a[TILE];
//     float reg_b[TILE];
//     float reg_c[TILE][TILE];


//     int tid = threadIdx.y * blockDim.x + threadIdx.x;

//     int smem_a_m = tid / 2;
//     int smem_a_k = (tid % 2) << 2;

//     int smem_b_k = tid / 32;
//     int smem_b_n = (tid % 32) << 2;

//     int gmem_a_m = smem_a_m + blockIdx.y * BLOCK_M;
//     int gmem_b_n = smem_b_n + blockIdx.x * BLOCK_N;

//     for(int k = 0; k < K / BLOCK_K; k++){
//         int gmem_a_k = smem_a_k + k * BLOCK_K;
//         int gmem_b_k = smem_b_k + k * BLOCK_K;

//         // Copy
//         FLOAT4(shared_a[smem_a_m][smem_a_k]) = FLOAT4(a[OFFSET(gmem_a_m, gmem_a_k, K)]);
//         FLOAT4(shared_b[smem_b_k][smem_b_n]) = FLOAT4(b[OFFSET(gmem_b_k, gmem_b_n, N)]);
//         __syncthreads();

//         // Compute
//         int ty = threadIdx.y * TILE;
//         int tx = threadIdx.x * TILE;

//         for(int kk = 0; kk < BLOCK_K; kk++){
//             for(int i = 0; i < TILE; i++) {
//                 reg_a[i] = shared_a[ty + i][kk];
//                 reg_b[i] = shared_b[kk][tx + i];
//             }
//             for(int i = 0; i < TILE; i++)
//                 for(int j = 0; j < TILE; j++)
//                     reg_c[i][j] += reg_a[i] * reg_b[j];
//         }
//         __syncthreads();
//     }

//     // Write Back
//     int row = blockIdx.y * BLOCK_M + threadIdx.y * TILE;
//     int col = blockIdx.x * BLOCK_N + threadIdx.x * TILE;

//     for(int i = 0; i < TILE; i++)
//         for(int j = 0; j < TILE; j += 4)
//             FLOAT4(c[OFFSET(row + i, col + j, N)]) = FLOAT4(reg_c[i][j]); 
// }

// void sgemm_v3(float *a, float *b, float *c, int M, int N, int K){
//     size_t size_a = sizeof(float) * M * K;
//     size_t size_b = sizeof(float) * K * N;
//     size_t size_c = sizeof(float) * M * N;


//     float *da, *db, *dc;
//     cudaMalloc(&da, size_a);
//     cudaMalloc(&db, size_b);
//     cudaMalloc(&dc, size_c);

//     cudaMemcpy(da, a, size_a, cudaMemcpyHostToDevice);
//     cudaMemcpy(db, b, size_b, cudaMemcpyHostToDevice);

//     const int BLOCK_M = 128, BLOCK_N = 128, TILE = 8;
//     dim3 grid(N / BLOCK_N, M / BLOCK_M);
//     dim3 block(BLOCK_N / TILE, BLOCK_M / TILE);

//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);

//     cudaEventRecord(start, 0);
//     sgemm_v3_kernel<<<grid, block>>>(da, db, dc, M, N, K);
//     cudaEventRecord(stop, 0);

//     cudaEventSynchronize(stop);


//     float msecond = 0.0;
//     cudaEventElapsedTime(&msecond, start, stop);

//     cudaEventDestroy(start);
//     cudaEventDestroy(stop);

//     std::cout << msecond << std::endl;

//     cudaMemcpy(c, dc, size_c, cudaMemcpyDeviceToHost);

//     cudaFree(da);
//     cudaFree(db);
//     cudaFree(dc);
// }

// // Warp Tiling
// __global__ void sgemm_v4(float *a, float *b, float*c, int M, int N, int K){

// }

// // Bank Free
// __global__ void sgemm_v5(float *a, float *b, float*c, int M, int N, int K){

// }

// // Pipeline
// __global__ void sgemm_v6(float *a, float *b, float*c, int M, int N, int K){

// }

// // WMMA
// __global__ void sgemm_v7(float *a, float *b, float*c, int M, int N, int K){

// }

void sgemm_cublas(float *a, float *b, float *c, int M, int N, int K){

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    float cublas_alpha = 1.0;
    float cublas_beta = 0;

    size_t size_a = sizeof(float) * M * K;
    size_t size_b = sizeof(float) * K * N;
    size_t size_c = sizeof(float) * M * N;

    float *da, *db, *dc;
    CUDA_CHECK(cudaMalloc(&da, size_a));
    CUDA_CHECK(cudaMalloc(&db, size_b));
    CUDA_CHECK(cudaMalloc(&dc, size_c));

    cudaMemcpy(da, a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, size_b, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &cublas_alpha, da, M, db, K, &cublas_beta, dc, M);
    cudaGetLastError();                             
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);


    float msecond = 0.0;
    cudaEventElapsedTime(&msecond, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "second: " << msecond << std::endl;

    cudaMemcpy(c, dc, size_c, cudaMemcpyDeviceToHost);

    // for(int i = 0; i < M; i++){
    //     for(int j = 0; j < N; j++)
    //         std::cout << c[OFFSET(i, j, N)] << " ";
    //     std::cout << std::endl;
    // }

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    cublasDestroy(handle);
}





int main(){
    int M = 1024, N = 1024, K = 1024;

    size_t size_a = sizeof(float) * M * K;
    size_t size_b = sizeof(float) * K * N;
    size_t size_c = sizeof(float) * M * N;

    float *a = (float*)malloc(size_a);
    float *b = (float*)malloc(size_b);
    float *c = (float*)malloc(size_c);

    for(int i = 0; i < M * K; i++) a[i] = 1.0;
    for(int i = 0; i < K * N; i++) b[i] = 1.0;

    // sgemm_v1(a, b, c, M, N, K);
    // sgemm_v2(a, b, c, M, N, K);
    // sgemm_v3(a, b, c, M, N, K);
    sgemm_cublas(a, b, c, M, N, K);


    return 0;
}