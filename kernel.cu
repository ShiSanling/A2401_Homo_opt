#include <stdio.h>
#include <cuda_runtime.h>

#define N 4 // 矩阵大小

__device__ float determinant(float *mat);

__global__ void matrixInverse(float *A, float *A_inv) {
    // 使用Cramer法则求逆
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float adjugate[N][N];

    // 计算伴随矩阵
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int sign = ((i + j) % 2 == 0) ? 1 : -1;
            float submat[N-1][N-1];
            int submat_i = 0, submat_j = 0;
            for (int k = 0; k < N; ++k) {
                if (k == i) continue;
                submat_j = 0;
                for (int l = 0; l < N; ++l) {
                    if (l == j) continue;
                    submat[submat_i][submat_j++] = A[k * N + l];
                }
                submat_i++;
            }
            adjugate[i][j] = sign * determinant((float*)submat);
        }
    }

    // 计算矩阵的行列式值
    float det = determinant(A);

    // 计算逆矩阵
    if (det != 0) {
        A_inv[row * N + col] = adjugate[col][row] / det;
    }
}

// 递归计算行列式
__device__ float determinant(float *mat) {
    if (N == 1) {
        return mat[0];
    } else if (N == 2) {
        return mat[0] * mat[3] - mat[1] * mat[2];
    } else {
        float det = 0;
        float submat[(N-1)*(N-1)];
        int sign = 1;
        for (int i = 0; i < N; ++i) {
            int submat_i = 0;
            for (int j = 1; j < N; ++j) {
                int submat_j = 0;
                for (int k = 0; k < N; ++k) {
                    if (k == i) continue;
                    submat[submat_i*(N-1) + submat_j++] = mat[j*N + k];
                }
                submat_i++;
            }
            det += sign * mat[i] * determinant(submat);
            sign = -sign;
        }
        return det;
    }
}

int main() {
    float h_A[N][N], h_A_inv[N][N]; // 主机上的矩阵和逆矩阵
    float *d_A, *d_A_inv; // 设备上的矩阵和逆矩阵

    // 初始化矩阵 h_A
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            h_A[i][j] = i + j;
        }
    }

    // 分配设备内存
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_A_inv, N * N * sizeof(float));

    // 将矩阵数据从主机复制到设备
    cudaMemcpy(d_A, h_A, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // 定义CUDA网格和线程块大小
    dim3 threadsPerBlock(N, N);
    dim3 numBlocks(1, 1);

    // 调用核函数计算逆矩阵
    matrixInverse<<<numBlocks, threadsPerBlock>>>(d_A, d_A_inv);

    // 将结果从设备复制回主机
    cudaMemcpy(h_A_inv, d_A_inv, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果
    printf("Original Matrix:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%.2f\t", h_A[i][j]);
        }
        printf("\n");
    }

    printf("\nInverse Matrix:\n");
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%.2f\t", h_A_inv[i][j]);
        }
        printf("\n");
    }

    // 释放内存
    cudaFree(d_A);
    cudaFree(d_A_inv);

    return 0;
}
