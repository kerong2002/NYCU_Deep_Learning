import numpy as np
import matplotlib.pyplot as plt
# 設定支援中文字體
plt.rcParams['font.family'] = 'Microsoft JhengHei'  # 微軟正黑體

# 解決負號無法顯示的問題（常見）
plt.rcParams['axes.unicode_minus'] = False
# 初始化資料
seasons  = [1, 2, 3, 4]           # 季度
stock1 = [4, 8, 2, 6]     # 股票1每個季度對應的股價
stock2 = [10, 12, 5, 3]   # 股票2每個季度對應的股價

# 轉換資料為 Numpy 陣列（可選步驟）
seasons = np.array(seasons)
stock1 = np.array(stock1)
stock2 = np.array(stock2)
