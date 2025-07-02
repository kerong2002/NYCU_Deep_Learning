import matplotlib.pyplot as plt

# 資料
date_lst = [1,2,3,4]
stock1 = [4, 8, 2, 6]
stock2 = [10, 12, 5, 3]

# 繪製兩條折線圖
plt.plot(date_lst, stock1)
plt.plot(date_lst, stock2)
plt.savefig("images/折線圖_v1.0.png")
plt.show()