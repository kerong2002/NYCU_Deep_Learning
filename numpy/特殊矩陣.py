import numpy as np

# 單位矩陣(正方形)
I = np.identity(3)
print(I)

I_rect = np.eye(3, M = 5)
# 單位矩陣(N * M)
print(I_rect)


mat = np.array([1,2,3, 4,5,6, 7,8,9])
mat2 = np.arange(9).reshape(3,3)
# 上三角矩陣
print(np.triu(mat2))
# 下三角矩陣
print(np.tril(mat2))

# 下三角(對角線下移1)
print(np.tril(mat2, k=-1))