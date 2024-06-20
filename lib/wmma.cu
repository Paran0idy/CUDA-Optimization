#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cfloat>

using namespace nvcuda;

#define OFFSET(i, j, N) ((i) * (N) + (j))
#define FLOAT4(pointer) reinterpret_cast<float4 *>(&pointer)[0]
__global__ void wmma_v1(half *a, half *b, half *c, int M, int N, int K) {
    const int BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 16;

    __shared__ half shared_a[BLOCK_N][BLOCK_K];
    __shared__ half shared_b[BLOCK_K][BLOCK_N];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> frag_a[2][1];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> frag_b[1][4];
    wmma::fragment<wmma::accumulator, 16, 16, 16, half> frag_c[2][4];

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            wmma::fill_fragment(frag_c[i][j], 0.0);

    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    int smem_a_m = tid / 2;
    int smem_a_k = (tid % 2) << 3;

    int smem_b_k = tid / 16;
    int smem_b_n = (tid % 16) << 3;

    int gmem_a_m = BLOCK_M * blockIdx.y + smem_a_m;
    int gmem_b_n = BLOCK_N * blockIdx.x + smem_b_n;

    int wid = tid / 32;
    int frag_c_m = wid % 4;
    int frag_c_n = wid / 4;

    const int WARP_M = 32, WARP_N = 64;

    for (int k = 0; k < K / BLOCK_K; k++) {
        int gmem_a_k = k * BLOCK_K;
        int gmem_b_k = k * BLOCK_K;

        FLOAT4(shared_a[smem_a_m][smem_a_k]) = FLOAT4(a[OFFSET(gmem_a_m, gmem_a_k, K)]);
        FLOAT4(shared_b[smem_b_k][smem_b_n]) = FLOAT4(b[OFFSET(gmem_b_k, gmem_b_n, N)]);
        __syncthreads();

        wmma::load_matrix_sync(frag_a[0][0], &shared_a[frag_c_m * WARP_M][0], BLOCK_K);
        wmma::load_matrix_sync(frag_a[1][0], &shared_a[frag_c_m * WARP_M + 16][0], BLOCK_K);

        wmma::load_matrix_sync(frag_b[0][0], &shared_b[0][frag_c_n * WARP_N], BLOCK_N);
        wmma::load_matrix_sync(frag_b[0][1], &shared_b[0][frag_c_n * WARP_N + 16], BLOCK_N);
        wmma::load_matrix_sync(frag_b[0][2], &shared_b[0][frag_c_n * WARP_N + 32], BLOCK_N);
        wmma::load_matrix_sync(frag_b[0][3], &shared_b[0][frag_c_n * WARP_N + 48], BLOCK_N);

        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 4; j++)
                wmma::mma_sync(frag_c[i][j], frag_a[i][0], frag_b[0][j], frag_c[i][j]);
        __syncthreads();
    }

    int row = blockIdx.y * BLOCK_M + frag_c_m * WARP_M;
    int col = blockIdx.x * BLOCK_N + frag_c_n * WARP_N;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 4; j++)
            wmma::store_matrix_sync(&c[OFFSET(row, col, N) + i * 16 * N + j * 16], frag_c[i][j], N, wmma::mem_row_major);
}


int main() {
    int M = 128, N = 128, K = 128;
    int size_a = sizeof(half) * M * K;
    int size_b = sizeof(half) * K * N;
    int size_c = sizeof(half) * M * N;

    half *a = (half *)malloc(size_a);
    half *b = (half *)malloc(size_b);
    half *c = (half *)malloc(size_c);

    for (int i = 0; i < M * K; i++) a[i] = 1.0;
    for (int i = 0; i < K * N; i++) b[i] = 1.0;

    half *da, *db, *dc;
    cudaMalloc(&da, size_a);
    cudaMalloc(&db, size_b);
    cudaMalloc(&dc, size_c);

    dim3 block(N / 128, M / 128);
    dim3 thread(128 / 8, 128 / 8);

    cudaMemcpy(da, a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(db, b, size_b, cudaMemcpyHostToDevice);
    wmma_v1<<<block, thread>>>(da, db, dc, M, N, K);
    cudaMemcpy(c, dc, size_c, cudaMemcpyDeviceToHost);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++)
            std::cout << float(c[OFFSET(i, j, N)]) << " ";
        std::cout << std::endl;
    }

    return 0;

}