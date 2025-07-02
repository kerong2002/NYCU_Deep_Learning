from data import*
import matplotlib.pyplot as plt

# 使用面向物件 (Object-oriented) 方式繪圖
# fig: 整張畫布 (Figure)
# axes: 2 行 1 列的多個坐標軸 (Axes)，以 NumPy 陣列形式儲存

fig, axes = plt.subplots(2, 2, figsize=(6, 6))

# 左上角子圖 (第 0 行第 0 列) 繪製長條圖，x 軸為季度，y 軸為股票1價格
axes[0, 0].bar(seasons, stock1)

# 右上角子圖 (第 0 行第 1 列) 繪製折線圖，藍色三角形虛線，x 軸為季度，y 軸為股票2價格
axes[0, 1].plot(seasons, stock2, "b^--")

ax = axes[1, 0]
# # 取得左下角子圖 (第 1 行第 0 列) 的 Axes 物件，方便後續操作
# 在左下角子圖畫散點圖 (scatterplots)
ax.scatter(
    seasons,             # x 軸：季度
    stock2 - stock1,     # y 軸：股票2與股票1的差價
    s=[10, 20, 50, 100], # 點的大小 (size)，可個別設定
    c=['r', 'b', 'c', 'y']  # 點的顏色，分別為紅、藍、青、黃
)

# 設定左上角子圖的標題為「股票1」
axes[0, 0].set_title("股票1")

# 設定右上角子圖的標題為「股票2」
axes[0, 1].set_title("股票2")

# 設定左下角子圖的 y 軸標籤為「差價(股票1-股票2)」
ax.set_ylabel("差價(股票2 - 股票1)")



# 顯示繪製完成的所有圖形
plt.show()
