import numpy as np
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.sparse as sparse
from pycuda.compiler import SourceModule

# CUDA Kernel 代码
kernel_code = """
__global__ void solve_linear_system(float *A_data, int *A_indices, int *A_indptr,
                                    float *x, float *b, int num_rows) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < num_rows) {
        float sum = 0.0f;
        for (int j = A_indptr[i]; j < A_indptr[i + 1]; ++j) {
            sum += A_data[j] * x[A_indices[j]];
        }
        b[i] = sum;
    }
}
"""

# 编译 CUDA Kernel
mod = SourceModule(kernel_code)

# 获取 CUDA Kernel 函数
solve_linear_system_kernel = mod.get_function("solve_linear_system")

def solve_sparse_linear_system(A_data, A_indices, A_indptr, x):
    num_rows = A_indptr.shape[0] - 1
    num_cols = x.shape[0]
    
    # 将数据传输到 GPU
    A_data_gpu = gpuarray.to_gpu(A_data.astype(np.float32))
    A_indices_gpu = gpuarray.to_gpu(A_indices.astype(np.int32))
    A_indptr_gpu = gpuarray.to_gpu(A_indptr.astype(np.int32))
    x_gpu = gpuarray.to_gpu(x.astype(np.float32))
    
    # 分配 GPU 存储空间用于存储结果
    b_gpu = cuda.mem_alloc(num_rows * 4)
    
    # 调用 CUDA Kernel 求解线性方程组
    block_size = 256
    grid_size = (num_rows + block_size - 1) // block_size
    solve_linear_system_kernel(A_data_gpu, A_indices_gpu, A_indptr_gpu, x_gpu, b_gpu,
                               np.int32(num_rows), block=(block_size, 1, 1), grid=(grid_size, 1))

    # 将结果从 GPU 复制回 CPU
    b = np.empty(num_rows, dtype=np.float32)
    cuda.memcpy_dtoh(b, b_gpu)
    
    return b

# 测试
# 构造稀疏矩阵和右端向量
A_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
A_indices = np.array([0, 1, 1, 2], dtype=np.int32)
A_indptr = np.array([0, 2, 3, 4], dtype=np.int32)
x = np.array([1.0, 2.0, 3.0], dtype=np.float32)

# 求解线性方程组
b = solve_sparse_linear_system(A_data, A_indices, A_indptr, x)
print("Solution:", b)
