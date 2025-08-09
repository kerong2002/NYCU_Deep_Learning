import os
import json
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class IclevrDataset(Dataset):
    """
    針對 iCLEVR 資料集的自定義 PyTorch Dataset 類別。
    """

    def __init__(self, data_dir: str, image_dir: str, mode: str, objects_json_path: str, transform=None):
        """
        初始化 IclevrDataset。
        """
        assert mode in ["train", "test", "new_test"], f"不支援的模式: '{mode}'"
        self.mode = mode
        self.transform = transform
        self.image_dir = image_dir

        # 載入對應模式的標籤數據 (例如 train.json, test.json)
        json_path = os.path.join(data_dir, f"{mode}.json")
        with open(json_path, 'r') as f:
            self.labels_data = json.load(f)

        # 載入物件名稱到索引的對照表 (objects.json)
        with open(objects_json_path, 'r') as f:
            self.objects_map = json.load(f)
        self.num_classes = len(self.objects_map)

        # 根據模式，設定檔案列表和標籤列表
        if self.mode == 'train':
            self.image_files = list(self.labels_data.keys())
            self.labels = list(self.labels_data.values())
        else:  # 測試模式下，數據直接就是標籤列表
            self.labels = self.labels_data

    def __len__(self) -> int:
        """返回資料集中的樣本總數。"""
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple:
        """
        Args:
            idx (int): 樣本的索引。
        Returns:
            tuple: 根據模式返回 (圖像, 標籤) 或 (虛擬圖像, 標籤)。
        """
        # 將當前樣本的文字標籤轉換為 one-hot 編碼的張量
        current_labels_text = self.labels[idx]
        one_hot_label = torch.zeros(self.num_classes)
        for label_text in current_labels_text:
            if label_text in self.objects_map:
                one_hot_label[self.objects_map[label_text]] = 1

        if self.mode == 'train':
            # 訓練模式下，讀取、轉換並返回真實圖像
            img_name = self.image_files[idx]
            img_path = os.path.join(self.image_dir, img_name)
            try:
                image = Image.open(img_path).convert("RGB")
            except FileNotFoundError:
                print(f"警告：找不到圖像檔案 {img_path}，將使用黑色圖像代替。")
                image = Image.new('RGB', (64, 64), color='black')

            if self.transform:
                image = self.transform(image)
            return image, one_hot_label
        else:
            # 測試模式下，沒有真實圖像，只返回標籤。
            # 為了讓 DataLoader 能以相同的格式工作，我們創建一個虛擬的圖像張量作為佔位符。
            dummy_image = torch.zeros((3, 64, 64))
            return dummy_image, one_hot_label


def get_dataloader(data_dir: str, image_dir: str, mode: str, batch_size: int, shuffle: bool = None,
                   num_workers: int = 4) -> DataLoader:
    # 定義圖像轉換流程
    transform_list = [transforms.Resize((64, 64))]
    if mode == 'train':
        # 只在訓練時進行數據增強（隨機水平翻轉）
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list.extend([
        transforms.ToTensor(),  # 將 PIL 圖像轉換為 PyTorch 張量
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 將 [0, 1] 範圍的張量標準化到 [-1, 1]
    ])
    transform = transforms.Compose(transform_list)

    objects_json_path = os.path.join(data_dir, 'objects.json')

    dataset = IclevrDataset(
        data_dir=data_dir,
        image_dir=image_dir,
        mode=mode,
        objects_json_path=objects_json_path,
        transform=transform
    )

    # 如果未指定 shuffle，則根據模式自動決定 (訓練時打亂，測試時不打亂)
    if shuffle is None:
        shuffle = (mode == 'train')

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True  # 如果記憶體充足，可以加速從 CPU 到 GPU 的數據傳輸
    )
    return dataloader