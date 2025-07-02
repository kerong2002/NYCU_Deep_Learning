import torch
print(torch.__version__)         # 應該顯示你剛裝的版本
print(torch.cuda.is_available()) # True 表示能用 GPU
print(torch.cuda.get_device_name(0))  # 顯示 GPU 名稱（如 RTX 4060）
