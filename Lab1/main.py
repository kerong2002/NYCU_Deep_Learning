'''
Name: DLP Lab1
Topic: back-propagation
Author: CHEN, KE-RONG
Date: 2025/07/02
'''

import numpy as np
import matplotlib.pyplot as plt

def generate_linear(n=100):
    # import numpy as np
    # 從 [0,1) 區間中均勻地隨機取出 n 筆二維座標資料點。
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        # 計算點在對角線上的投影距離（此變數未實際使用）
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1] :
            labels.append(0)
        else:
            labels.append(1)
    # 將輸入與標籤轉換為 NumPy 陣列，labels reshape 成 (n, 1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    # import numpy as np
    inputs = []
    labels = []

    for i in range(11):
        # 產生對角線 (0,0) 到 (1,1) 上的點，共 11 個，標籤為 0。
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)
        # 若剛好為中心點 (0.5, 0.5)，跳過，不再產生對稱點（避免重複）
        if 0.1*i==0.5 :
            continue

        # 對稱的點（如 (0.1, 0.9), (0.2, 0.8) ...），標記為 1。
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
    # 將輸入與標籤轉換為 NumPy 陣列，labels reshape 成 (21, 1)
    return np.array(inputs), np.array(labels).reshape(21,1)

def plot_data(x, y, title="Data Visualization"):
    """
    顯示資料點的散佈圖，依照 label 區分顏色與標記。
    參數:
        x: shape = (n, 2) 的輸入資料
        y: shape = (n, 1) 的標籤資料（值為 0 或 1）
        title: 圖片標題
    """
    # print(f'x={x}')
    # print(f'y={y}')
    plt.figure(figsize=(6, 6))

    # 類別 0：紅色圓點
    plt.scatter(x[y.flatten() == 0][:, 0],
                x[y.flatten() == 0][:, 1],
                color='red', label='Class 0')

    # 類別 1：藍色叉叉
    plt.scatter(x[y.flatten() == 1][:, 0],
                x[y.flatten() == 1][:, 1],
                color='blue', marker='x', label='Class 1')

    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.show()

# # 產生資料
x, y = generate_linear(n=100)
# # 畫出資料圖
plot_data(x, y, title="Linear Dataset")
#
# # 產生資料
# x, y = generate_XOR_easy()
# plot_data(x, y, title="XOR Dataset")