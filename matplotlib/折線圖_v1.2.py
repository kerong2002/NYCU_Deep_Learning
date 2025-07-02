import matplotlib.pyplot as plt
import numpy as np
# 設定支援中文字體
plt.rcParams['font.family'] = 'Microsoft JhengHei'  # 微軟正黑體

# 解決負號無法顯示的問題（常見）
plt.rcParams['axes.unicode_minus'] = False
# 資料
date_lst = [1,2,3,4]
stock1 = [4, 8, 2, 6]
stock2 = [10, 12, 5, 3]

# 設定折線圖的格式
plt.plot(date_lst, stock1, "ro--", label="元大:0050")
plt.plot(date_lst, stock2, "b^--", label="台積電:2330")
plt.title("折線圖")
plt.xlabel("時間")
plt.ylabel("股價")
plt.legend() # 圖例

# 設置x/y座標刻度
plt.xticks([1, 2, 3, 4])
plt.yticks(np.arange(2, 13, 1))

# 添加輔助網格
plt.grid()
plt.savefig("images/折線圖_v1.2.png")
plt.show()