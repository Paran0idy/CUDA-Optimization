import triton
import triton.language as tl

import torch


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_a_m, stride_a_k,
    stride_b_k, stride_b_n,
    stride_c_m, stride_c_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid = N // BLOCK_N
    pid_m = pid // num_pid
    pid_n = pid % num_pid
    
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offset = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offset = tl.arange(0, BLOCK_K)
    
    a_ptrs = a_ptr + m_offset[:, None] * stride_a_m + k_offset[None, :] * stride_a_k
    b_ptrs = b_ptr + k_offset[:, None] * stride_b_k + n_offset[None, :] * stride_b_n
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        
        acc += tl.dot(a, b)
        
        a_ptrs += BLOCK_K * stride_a_k
        b_ptrs += BLOCK_K * stride_b_k
    
    # acc = acc.to(tl.float16)
    c_ptrs = c_ptr + m_offset[:, None] * stride_c_m + n_offset[None, :] * stride_c_n
    tl.store(c_ptrs, acc)
    

def matmul(a, b):
    M, K = a.shape
    K, N = b.shape
    o = torch.empty((M, N), dtype=torch.float16, device="cuda")
    
    grid = lambda meta: (triton.cdiv(M , meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']), )
    matmul_kernel[grid](
        a, b, o,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        o.stride(0), o.stride(1),
        64, 64, 64,
    )
    
    return o


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],  # Argument names to use as an x-axis for the plot
        x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot
        # Possible values for `line_arg`
        line_vals=['cublas', 'triton'],
        # Label name for the lines
        line_names=["cuBLAS", "Triton"],
        # Line styles
        styles=[('green', '-'), ('blue', '-')],
        ylabel="TFLOPS",  # Label name for the y-axis
        plot_name="matmul-performance",  # Name for the plot, used also as a file name for saving the plot.
        args={},
    ))
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device='cuda', dtype=torch.float32)
    b = torch.randn((K, N), device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'cublas':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)


if __name__ == "__main__":
    # M, N, K = 1024, 1024, 1024
    # a = torch.rand((M, K), dtype=torch.float16, device="cuda")
    # b = torch.rand((K, N), dtype=torch.float16, device="cuda")
    
    # torch_out = torch.matmul(a, b)
    # triton_out = matmul(a, b)
    # print(torch_out)
    # print(triton_out)
    
    # print(torch.allclose(torch_out, triton_out))
    
    benchmark.run(show_plots=True, print_data=True)

    
    
    