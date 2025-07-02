from data import*
import matplotlib.pyplot as plt

# 使用面向物件 (Object-oriented) 方式繪圖
# fig: 整張畫布 (Figure)

# 建立 2x2 子圖，畫布大小 6x6 吋
fig, axes = plt.subplots(2, 2, figsize=(6, 6), sharex=True, sharey=True)
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

axes[1, 1].remove()
axes[0, 0].plot(seasons, stock1, 'r+-')

# 重新添加3D的到右下角
ax = fig.add_subplot(2, 2, 4,
                     projection='3d', facecolor="grey")
ax.stem(seasons, stock1, stock2-stock1)
ax.stem(seasons, stock1, stock2-stock1,
        linefmt='k--', basefmt='k--',
        bottom=10, orientation='y')
ax.plot_surface(np.array([1,1,4,4]).reshape(2,2),
                np.array([2.5,10,2.5,10]).reshape(2,2),
                np.array([0]*4).reshape(2,2),
                alpha=0.2, color='red')
ax.plot_surface(np.array([1,1,4,4]).reshape(2,2),
                np.array([10]*4).reshape(2,2),
                np.array([-2.5,8,-2.5,8]).reshape(2,2),
                alpha=0.2, color='black')
# figure 添加大標題跟註釋
fig.suptitle("股票分析圖")
fig.supylabel("股價")
fig.supxlabel("季度")

# 改變顏色
axes[1, 0].set_facecolor('grey')
axes[1, 0].patch.set_alpha(0.2)
axes[0, 0].set_facecolor('red')
axes[0, 0].patch.set_alpha(0.2)
# 緊湊一點
plt.tight_layout()


# 顯示繪製完成的所有圖形
plt.show()
