import numpy as np

arr1 = np.linspace(1, 101, 5)
print(arr1.shape); print(arr1)

# 不包括右區間
arr2 = np.linspace(1, 101, 5,
                   endpoint=False)
print(arr2.shape); print(arr2)

arr3 = np.logspace(0, 2, 5, endpoint = True)
print(arr3.shape), print(arr3)