import numpy as np

mat = np.array([1,2,3, 4,5,6, 7,8,9])
print(mat.shape); print(mat)

mat = mat.reshape(3,3)
print(mat.shape); print(mat)

mat = mat.reshape(1, 3, 3)
print(mat.shape); print(mat)