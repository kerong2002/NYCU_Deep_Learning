'''
Name: DLP Lab2
Topic: Binary Semantic Segmentation
Author: CHEN, KE-RONG
Date: 2025/07/11
'''
import os
import torch
import shutil
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

from PIL import Image
from tqdm import tqdm
from urllib.request import urlretrieve

class OxfordPetDataset(torch.utils.data.Dataset):
    """
    基礎的 OxfordPetDataset 類別，負責讀取原始圖片和遮罩。
    返回 Numpy Array。
    """
    def __init__(self, root, mode="train"):
        assert mode in {"train", "valid", "test"}
        self.root = root
        self.mode = mode
        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(self.root, "annotations", "trimaps")
        self.filenames = self._read_split()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg")
        mask_path = os.path.join(self.masks_directory, filename + ".png")

        image = np.array(Image.open(image_path).convert("RGB"))
        trimap = np.array(Image.open(mask_path))
        
        mask = self._preprocess_mask(trimap)

        return {'image': image, 'mask': mask}

    @staticmethod
    def _preprocess_mask(mask_np):
        mask_np[mask_np == 2.0] = 0.0  # 背景
        mask_np[(mask_np == 1.0) | (mask_np == 3.0)] = 1.0  # 前景 + 邊界
        return mask_np

    def _read_split(self):
        split_filename = "test.txt" if self.mode == "test" else "trainval.txt"
        split_filepath = os.path.join(self.root, "annotations", split_filename)
        with open(split_filepath) as f:
            split_data = f.read().strip("\n").split("\n")
        filenames = [x.split(" ")[0] for x in split_data]
        if self.mode == "train":
            filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        elif self.mode == "valid":
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
        return filenames

    @staticmethod
    def download(root):

        # load images
        filepath = os.path.join(root, "images.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

        # load annotations
        filepath = os.path.join(root, "annotations.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)
class SimpleOxfordPetDataset(OxfordPetDataset):
    """
    使用 Albumentations 函式庫實現高效資料增強的資料集類別。
    """
    def __init__(self, root, mode="train", image_size=(256, 256)):
        super().__init__(root, mode)

        # 建立訓練階段的資料增強流程（transformation pipeline）
        self.train_transform = A.Compose([

            # 將影像與遮罩調整為指定尺寸（例如 256x256）
            A.Resize(height=image_size[0], width=image_size[1]),

            # 以 50% 的機率進行水平翻轉，模擬左右對稱變化
            A.HorizontalFlip(p=0.5),

            # 以 30% 的機率進行隨機旋轉（範圍 ±15 度），增強對角度變化的魯棒性
            A.Rotate(limit=15, p=0.3),

            # 以 50% 的機率對影像進行顏色擾動（包含亮度、對比、飽和度變化）
            A.ColorJitter(p=0.5),

            # 進行標準化（將像素值從 [0,255] 映射到標準分佈），使用 ImageNet 的平均值與標準差
            A.Normalize(
                mean=[0.485, 0.456, 0.406],  # 三個通道的平均值（R, G, B）
                std=[0.229, 0.224, 0.225],  # 三個通道的標準差
                max_pixel_value=255.0  # 原始像素最大值，用於正規化
            ),

            # 將影像與遮罩從 Numpy 格式轉為 PyTorch Tensor，並轉換通道順序為 (C, H, W)
            ToTensorV2(),
        ])

        # 驗證 / 測試階段的資料轉換流程（不做資料增強，只保留基本處理）
        self.val_transform = A.Compose([
            # 尺寸調整
            A.Resize(height=image_size[0], width=image_size[1]),

            # 標準化（與訓練階段相同）
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0
            ),

            # 轉換成 Tensor 格式
            ToTensorV2(),
        ])

        # 根據 mode（train / valid / test）決定使用哪一組 transform
        self.transform = self.train_transform if mode == 'train' else self.val_transform

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        
        transformed = self.transform(image=sample['image'], mask=sample['mask'])
        
        # Albumentations 的 ToTensorV2 會將 mask 轉為 (H, W, C)，我們需要 (C, H, W)
        # 且分割任務的 mask 通常是 (1, H, W)
        return {'image': transformed['image'], 'mask': transformed['mask'].unsqueeze(0)}

# --- 下載與解壓縮輔助函式 (保持不變) ---
class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None: self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        print(f"檔案 {os.path.basename(filepath)} 已存在。")
        return
    with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=os.path.basename(filepath)) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n

def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    dst_dir = os.path.splitext(filepath)[0]
    if not os.path.exists(dst_dir):
        print(f"正在解壓縮 {os.path.basename(filepath)}...")
        shutil.unpack_archive(filepath, extract_dir)
    else:
        print(f"檔案 {os.path.basename(dst_dir)} 已解壓縮。")

def load_dataset(data_path, mode, image_size=(256, 256)):
    if not os.path.exists(os.path.join(data_path, 'images')) or not os.path.exists(os.path.join(data_path, 'annotations')):
        print("正在下載資料集...")
        OxfordPetDataset.download(data_path)
    else:
        print("資料集已存在。")
    dataset = SimpleOxfordPetDataset(root=data_path, mode=mode, image_size=image_size)
    return dataset