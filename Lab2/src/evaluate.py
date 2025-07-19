'''
Name: DLP Lab2
Topic: Binary Semantic Segmentation
Author: CHEN, KE-RONG
Date: 2025/07/11
'''
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from torch.utils.data import DataLoader

# 解決 matplotlib 中文顯示問題
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # 指定繁體中文黑體
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
from tqdm import tqdm

# 匯入我們自己的模組
from utils import dice_score
from oxford_pet import load_dataset
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet

# 解決 matplotlib 中文顯示問題
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

def evaluate(model, dataloader, device, output_dir=None):
    """
    在給定的資料集上評估模型，並返回平均 Dice 分數。

    Args:
        model (torch.nn.Module): 已載入權重的模型。
        dataloader (DataLoader): 用於評估的資料載入器。
        device (torch.device): 執行運算的裝置。
        output_dir (str, optional): 儲存視覺化結果的資料夾。若為 None，則不儲存。

    Returns:
        float: 資料集上的平均 Dice 分數。
    """
    model.eval()  # 將模型設定為評估模式
    all_dices = []

    # 如果需要儲存，先刪除舊的資料夾再重建，確保結果乾淨
    if output_dir:
        import shutil
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    with torch.no_grad():
        # 使用 tqdm 顯示進度條
        pbar = tqdm(dataloader, desc="評估中", leave=False)
        for i, batch in enumerate(pbar):
            images = batch['image'].to(device)
            gt_masks = batch['mask'].to(device)

            # 前向傳播
            pred_logits = model(images)
            
            # 計算這個批次的平均 Dice 分數
            batch_dice = dice_score(pred_logits, gt_masks)
            all_dices.append(batch_dice)
            
            pbar.set_postfix(avg_dice=f"{(sum(all_dices) / len(all_dices)):.4f}")

            # 如果需要，儲存視覺化結果
            if output_dir:
                # 將預測的 logits 轉換為二元遮罩
                pred_masks = (torch.sigmoid(pred_logits) > 0.5).float()
                for j in range(images.shape[0]):
                    # 只儲存前幾個批次的前幾張圖，避免產生過多檔案
                    if i < 5 and j < 2:
                        save_figure(
                            images[j].cpu(),
                            gt_masks[j].cpu(),
                            pred_masks[j].cpu(),
                            os.path.join(output_dir, f"batch_{i}_img_{j}.png")
                        )

    avg_dice = sum(all_dices) / len(all_dices) if all_dices else 0
    return avg_dice, all_dices

def save_figure(image, gt_mask, pred_mask, save_path):
    """將單張圖片、真實遮罩與預測遮罩一同繪製並儲存。"""
    # 反正規化圖片以便顯示
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image * std + mean
    image = image.permute(1, 2, 0).numpy()
    image = np.clip(image, 0, 1)

    gt_mask = gt_mask.squeeze().numpy()
    pred_mask = pred_mask.squeeze().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('原始圖片')
    axes[0].axis('off')
    
    axes[1].imshow(gt_mask, cmap='gray')
    axes[1].set_title('真實遮罩')
    axes[1].axis('off')
    
    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title('預測遮罩')
    axes[2].axis('off')
    
    plt.tight_layout()
    # 使用 bbox_inches='tight' 確保標題等元素不會被裁切
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def get_args():
    parser = argparse.ArgumentParser(description='評估已訓練的分割模型')
    parser.add_argument('--model_path', '-m', type=str, required=True, help='已訓練模型的 .pth 檔案路徑')
    parser.add_argument('--data_path', '-d', type=str, required=True, help='資料集路徑')
    parser.add_argument('--model_type', '-t', type=str, required=True, choices=['unet', 'resnet34_unet'], help='模型類型')
    parser.add_argument('--output_dir', '-o', type=str, default=None, help='(可選) 儲存視覺化結果的資料夾路徑')
    parser.add_argument('--batch_size', '-b', type=int, default=16, help='批次大小')
    parser.add_argument('--num_workers', '-w', type=int, default=2, help='資料載入器的工作線程數')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用裝置: {device}")
    
    # 載入驗證資料集 (10%)
    # 注意：我們在 'valid' 模式下載入資料
    test_dataset = load_dataset(args.data_path, mode='valid', image_size=(256, 256))
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    print(f"驗證集大小: {len(test_dataset)}")
    
    # 根據參數選擇模型架構
    if args.model_type == 'unet':
        model = UNet(n_channels=3, n_classes=1)
    elif args.model_type == 'resnet34_unet':
        model = ResNet34_UNet(n_channels=3, n_classes=1)
    
    # 載入已訓練的模型權重
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    print(f"從 {args.model_path} 載入模型")
    
    # 執行評估
    avg_dice_score, all_dices = evaluate(model, test_loader, device, args.output_dir)
    
    print(f"\n評估完成！")
    print(f"驗證集上的平均 Dice 分數: {avg_dice_score:.4f}")

    # 如果有指定輸出目錄，則繪製並儲存 Dice 分數的分佈圖
    if args.output_dir:
        plt.figure(figsize=(10, 6))
        plt.hist(all_dices, bins=25, alpha=0.75, color='cornflowerblue')
        plt.axvline(avg_dice_score, color='red', linestyle='dashed', linewidth=2, label=f'平均值: {avg_dice_score:.4f}')
        plt.title('驗證集 Dice 分數分佈圖', fontsize=16)
        plt.xlabel('Dice 分數', fontsize=12)
        plt.ylabel('樣本數量', fontsize=12)
        plt.legend()
        plt.grid(axis='y', alpha=0.5)
        
        dist_path = os.path.join(args.output_dir, 'dice_distribution.png')
        plt.savefig(dist_path, dpi=150, bbox_inches='tight')
        print(f"Dice 分數分佈圖已儲存至: {dist_path}")