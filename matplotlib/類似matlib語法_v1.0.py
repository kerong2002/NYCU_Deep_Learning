from data import*
import matplotlib.pyplot as plt

# 建立一個寬 9 吋、高 3 吋的繪圖視窗
plt.figure(figsize=(9, 3))

# 建立子圖 1：1 行 2 列，第一個子圖
plt.subplot(121)
# 繪製長條圖，x 軸為 seasons，y 軸為 stock1，並設定圖例標籤
plt.bar(seasons, stock1, label="股票代碼: abc")

# 建立子圖 2：1 行 2 列，第二個子圖
plt.subplot(122)
# 繪製折線圖，x 軸為 seasons，y 軸為 stock2，線型為藍色三角形虛線，設定圖例標籤
plt.plot(seasons, stock2, "b^--", label="股票代碼: def")

# 顯示圖例（預設位置）
plt.legend()

# 顯示繪製完成的所有圖形
plt.show()
