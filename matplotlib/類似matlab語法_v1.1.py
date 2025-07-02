from data import*
import matplotlib.pyplot as plt

# 使用面向物件 (Object-oriented) 方式繪圖
# fig: 整張畫布 (Figure)
# axes: 2 行 2 列的多個坐標軸 (Axes)，以 NumPy 陣列形式儲存

fig, axes = plt.subplots(2, 1, figsize=(6, 6))  # 建立 2x2 子圖，畫布大小 6x6 吋

# 左上角子圖 (第 0 行第 0 列) 繪製長條圖，x 軸為季度，y 軸為股票1價格
axes[0].bar(seasons, stock1)

# 右上角子圖 (第 0 行第 1 列) 繪製折線圖，藍色三角形虛線，x 軸為季度，y 軸為股票2價格
axes[1].plot(seasons, stock2, "b^--")


# 顯示繪製完成的所有圖形
plt.show()
