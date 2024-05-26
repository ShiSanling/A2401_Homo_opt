from scipy.sparse import csr_matrix
import numpy as np

# 创建简单的测试数据
sK = np.array([1.0, 2.0, 3.0], dtype=np.float32)
iK = np.array([0, 1, 2], dtype=np.int32)
jK = np.array([0, 1, 2], dtype=np.int32)

# 创建稀疏矩阵
K = csr_matrix((sK, (iK, jK)), shape=(3, 3), dtype=np.float32)

print(K.dtype)  # 检查数据类型
