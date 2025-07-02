import torch

print("CUDA 是否可用：", torch.cuda.is_available())
print("可用 GPU 數量：", torch.cuda.device_count())
print("目前 GPU 名稱：", torch.cuda.get_device_name(0))
