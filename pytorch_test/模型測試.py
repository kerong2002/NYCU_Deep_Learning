import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

model = nn.Sequential(
    nn.Linear(1000, 500),
    nn.ReLU(),
    nn.Linear(500, 10)
).to(device)

x = torch.randn(64, 1000).to(device)  # 模擬一個 batch 的輸入
output = model(x)

print("模型輸出：", output.shape)
print("是否在 GPU 上：", output.device)
