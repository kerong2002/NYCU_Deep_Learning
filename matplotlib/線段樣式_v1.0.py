import matplotlib.pyplot as plt
import numpy as np

# 設定支援中文顯示，使用「微軟正黑體」
plt.rcParams['font.family'] = 'Microsoft JhengHei'  # 字體設定

# 建立圖表大小 (寬 9 吋，高 4 吋)
plt.figure(figsize=(9, 4))

# 繪製不同顏色與線條樣式的折線圖
plt.plot(list(range(5)), [5]*5, 'r.-', label="(r.-) 紅色，實心小圓點，實線")
plt.plot(list(range(5)), [4]*5, 'go--', label="(go--) 綠色，大圓點，虛線")
plt.plot(list(range(5)), [3]*5, 'k+-.', label="(k+-.) 黑色，加號標記，點虛線")
plt.plot(list(range(5)), [2]*5, 'c*:', label="(c*:) 青色，星號標記，點狀線")
plt.plot(list(range(5)), [1]*5, 'bs', label="(bs) 藍色，方塊標記，無連線")

# 顯示圖例，位置設定在右下角，字體大小18
plt.legend(loc='lower right', prop={'size': 18})

# 儲存圖片到指定資料夾
plt.savefig("images/線段樣式_v1.0.png")

# 顯示圖表
plt.show()
