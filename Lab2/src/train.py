'''
Name: DLP Lab2
Topic: Binary Semantic Segmentation
Author: CHEN, KE-RONG
Date: 2025/07/11
'''
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

# 匯入我們自己的模組
from oxford_pet import load_dataset
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from utils import dice_score

# 嘗試匯入 wandb，如果失敗則禁用
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def main(args):
    # --- 1. 設定 ---
    # 確保儲存模型的根目錄存在
    save_dir = args.output_dir
    os.makedirs(save_dir, exist_ok=True)
    print(f"模型將儲存於: {save_dir}")

    # 設定運算裝置 (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用裝置: {device}")

    # 初始化 wandb (如果可用且被啟用)
    if args.use_wandb:
        if WANDB_AVAILABLE:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                wandb.init(project=args.project, config=vars(args), name=f'{args.model}_{timestamp}')
                print("Weights & Biases (wandb) 已成功啟用。")
            except Exception as e:
                print(f"初始化 wandb 時發生錯誤: {e}")
                print("將禁用 wandb 功能。請確認您已安裝 wandb (pip install wandb) 並已登入 (wandb login)。")
                args.use_wandb = False
        else:
            print("警告: 您指定了 --use_wandb，但 wandb 套件未安裝。將禁用 wandb 功能。")
            args.use_wandb = False

    # --- 2. 資料載入 (僅訓練集) ---
    # 注意：所有的 transform 都已經在 oxford_pet.py 中使用 Albumentations 實現了
    train_dataset = load_dataset(args.data_path, mode='train', image_size=(args.image_size, args.image_size))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    print(f"資料集大小 - 訓練集: {len(train_dataset)}")

    # --- 3. 模型、優化器、損失函式、學習率排程器 ---
    # 根據參數選擇要載入的模型
    if args.model == 'unet':
        model = UNet(n_channels=3, n_classes=1)
    elif args.model == 'resnet34_unet':
        model = ResNet34_UNet(n_channels=3, n_classes=1)
    else:
        raise ValueError(f"未知的模型: {args.model}")
    model.to(device)

    # 設定優化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # 設定損失函式，BCEWithLogitsLoss 內部已包含 Sigmoid，更數值穩定
    criterion = nn.BCEWithLogitsLoss()
    
    # 設定學習率排程器
    scheduler = None
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'onecycle':
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate, epochs=args.epochs, steps_per_epoch=len(train_loader))
    elif args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.scheduler == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma)

    # --- 4. 訓練迴圈 ---
    print("\n開始訓練...")
    for epoch in range(args.epochs):
        model.train() # 將模型設定為訓練模式
        total_loss = 0.0
        total_dice = 0.0
        
        # 使用 tqdm 顯示進度條
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True)
        for batch in pbar:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            optimizer.zero_grad() # 清除舊的梯度
            outputs = model(images) # forward
            
            # 確保 mask 的資料型別為 float，以匹配模型的輸出
            masks = masks.float()
            loss = criterion(outputs, masks)
            
            # 反向傳播
            loss.backward()
            # 更新權重
            optimizer.step()
            
            # 如果使用 OneCycleLR，則在每個 step 後更新學習率
            if args.scheduler == 'onecycle':
                scheduler.step()
            
            # 累加 loss 和 dice score
            total_loss += loss.item()
            total_dice += dice_score(outputs, masks)
            
            # 更新進度條的後綴訊息，顯示即時的 loss 和平均 dice
            pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{total_dice / (pbar.n + 1):.4f}", lr=f"{optimizer.param_groups[0]['lr']:.1E}")
        
        # 在 epoch 結束後，更新 epoch-level 的排程器
        if args.scheduler in ['cosine', 'step', 'multistep']:
            scheduler.step()
        
        # 計算整個 epoch 的平均 loss 和 dice
        avg_train_loss = total_loss / len(train_loader)
        avg_train_dice = total_dice / len(train_loader)
        
        print(f"Epoch {epoch+1} 完成 | 平均訓練損失: {avg_train_loss:.4f}, 平均訓練 Dice: {avg_train_dice:.4f}")
        
        # 使用 wandb 紀錄
        if args.use_wandb:
            wandb.log({
                'epoch': epoch + 1, 
                'train_loss': avg_train_loss, 
                'train_dice': avg_train_dice, 
                'learning_rate': optimizer.param_groups[0]['lr']
            })

    # --- 5. 訓練結束，儲存最終模型 ---
    final_model_path = os.path.join(save_dir, f'{args.model}.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"\n訓練完成！最終模型已儲存至: {final_model_path}")

    if args.use_wandb:
        wandb.finish()

def get_args():
    parser = argparse.ArgumentParser(description='在 Oxford-IIIT Pet 資料集上訓練分割模型')
    parser.add_argument('--data_path', type=str, required=True, help='資料集路徑')
    parser.add_argument('--epochs', '-e', type=int, default=20, help='訓練週期數量')
    parser.add_argument('--batch_size', '-b', type=int, default=8, help='批次大小')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-4, help='學習率')
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4, help='權重衰減 (L2正規化)')
    parser.add_argument('--num_workers', '-nw', type=int, default=2, help='資料載入器的工作線程數')
    parser.add_argument('--image_size', type=int, default=256, help='圖片大小 (寬和高相同)')
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'resnet34_unet'], help='要訓練的模型')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['none', 'cosine', 'onecycle', 'step', 'multistep'], help='學習率排程器')
    parser.add_argument('--lr_step_size', type=int, default=10, help='StepLR 的 step size')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='StepLR 和 MultiStepLR 的 gamma 因子')
    parser.add_argument('--lr_milestones', type=int, nargs='+', default=[10, 15], help='MultiStepLR 的 milestones (epoch 節點)')
    parser.add_argument('--output_dir', type=str, default='saved_models', help='儲存模型的資料夾')
    parser.add_argument('--use_wandb', action='store_true', help='使用 wandb 進行實驗追蹤')
    parser.add_argument('--project', type=str, default='DLP_Lab2_SemanticSegmentation', help='wandb 專案名稱')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args)