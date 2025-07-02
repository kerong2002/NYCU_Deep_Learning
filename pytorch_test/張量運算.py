import torch
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print("使用裝置：", device)

# 建立大張量並搬到 GPU
a = torch.randn(10000, 10000).to(device)
b = torch.randn(10000, 10000).to(device)

start = time.time()
c = torch.matmul(a, b)  # 大矩陣乘法
torch.cuda.synchronize()  # 等待 GPU 完成
end = time.time()

print("矩陣乘法完成，耗時：", end - start, "秒")
