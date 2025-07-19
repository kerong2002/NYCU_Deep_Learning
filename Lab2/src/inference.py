'''
Name: DLP Lab2
Topic: Binary Semantic Segmentation
Author: CHEN, KE-RONG
Date: 2025/07/11
'''
import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

# 匯入我們自己的模組
from utils import dice_score
from oxford_pet import load_dataset
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet

# 解決 matplotlib 中文顯示問題
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

def save_visualization(image, gt_mask, pred_mask, save_path):
    """將單張圖片、真實遮罩與預測遮罩一同繪製並儲存。"""
    # 反正規化圖片以便顯示
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image.cpu() * std + mean
    image = image.permute(1, 2, 0).numpy()
    image = np.clip(image, 0, 1)

    gt_mask = gt_mask.cpu().squeeze().numpy()
    pred_mask = pred_mask.cpu().squeeze().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
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
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def save_dice_distribution(all_dices, output_dir):
    """繪製並儲存 Dice 分數的分佈直方圖。"""
    avg_dice = sum(all_dices) / len(all_dices) if all_dices else 0
    
    plt.figure(figsize=(10, 6))
    plt.hist(all_dices, bins=25, alpha=0.75, color='cornflowerblue')
    plt.axvline(avg_dice, color='red', linestyle='dashed', linewidth=2, label=f'平均值: {avg_dice:.4f}')
    plt.title('Dice 分數分佈圖', fontsize=16)
    plt.xlabel('Dice 分數', fontsize=12)
    plt.ylabel('樣本數量', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    
    dist_path = os.path.join(output_dir, 'dice_distribution.png')
    plt.savefig(dist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Dice 分數分佈圖已儲存至: {dist_path}")

def inference(args):
    """
    在測試集上對已訓練的模型進行測試，回報平均 Dice 分數，並可選擇性地儲存視覺化結果。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用裝置: {device}")

    if args.output_dir:
        import shutil
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)
        print(f"視覺化結果將儲存於: {args.output_dir}")

    if args.model_type == 'unet':
        model = UNet(n_channels=3, n_classes=1)
    elif args.model_type == 'resnet34_unet':
        model = ResNet34_UNet(n_channels=3, n_classes=1)
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"從 {args.model_path} 載入模型")

    test_dataset = load_dataset(args.data_path, mode='test', image_size=(args.image_size, args.image_size))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"測試集大小: {len(test_dataset)}")

    all_dices = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="測試中")):
            images = batch['image'].to(device)
            gt_masks = batch['mask'].to(device)

            pred_logits = model(images)
            
            dice = dice_score(pred_logits, gt_masks)
            all_dices.append(dice)

            if args.output_dir and i < 5:
                pred_masks = (torch.sigmoid(pred_logits) > 0.5).float()
                for j in range(images.shape[0]):
                    save_path = os.path.join(args.output_dir, f"batch_{i}_img_{j}.png")
                    save_visualization(images[j], gt_masks[j], pred_masks[j], save_path)

    avg_dice_score = sum(all_dices) / len(all_dices) if all_dices else 0
    
    print(f"\n測試完成！")
    print(f"測試集上的平均 Dice 分數: {avg_dice_score:.4f}")

    if args.output_dir:
        save_dice_distribution(all_dices, args.output_dir)

def get_args():
    parser = argparse.ArgumentParser(description='對分割模型進行測試')
    parser.add_argument('--model_path', '-m', type=str, required=True, help='已訓練模型的 .pth 檔案路徑')
    parser.add_argument('--data_path', '-d', type=str, required=True, help='資料集路徑')
    parser.add_argument('--model_type', '-t', type=str, required=True, choices=['unet', 'resnet34_unet'], help='模型類型')
    parser.add_argument('--output_dir', '-o', type=str, default=None, help='(可選) 儲存視覺化結果的資料夾路徑')
    parser.add_argument('--image_size', type=int, default=256, help='輸入模型前的圖片大小')
    parser.add_argument('--batch_size', '-b', type=int, default=16, help='批次大小')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    inference(args)