import numpy as np

arr1 = np.zeros((2, 2))
arr2 = np.ones((2, 4))
# print(arr1.shape); print(arr1)
# print(arr2.shape); print(arr2)

arr3 = np.full((2, 2), -np.inf)
arr4 = np.full((2,2), [1, 2])

# print(arr3.shape); print(arr3)
# print(arr4.shape); print(arr4)

x = np.array([[1, 2, 3]])
ones_1 = np.ones_like(x)
ones_2 = np.ones(x.shape)
print(x)
print(ones_1)
print(ones_2)