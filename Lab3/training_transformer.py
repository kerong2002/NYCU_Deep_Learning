'''
Name: DLP Lab3
Topic: MaskGIT
Author: CHEN, KE-RONG
Date: 2025/07/15
'''
import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader
import wandb
import shutil

# 專門用於訓練 Transformer 的類別
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        # 初始化模型並移至指定設備 (如 GPU)
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        # 設定優化器和學習率排程器
        self.optim, self.scheduler = self.configure_optimizers(args)
        # 準備訓練所需的資料夾
        self.prepare_training(args)
        # 保存設備資訊
        self.device = args.device
        
    @staticmethod
    def prepare_training(args):
        # 建立儲存模型檢查點的資料夾
        os.makedirs(args.checkpoint_path, exist_ok=True)

    def train_one_epoch(self, train_dataloader, epoch, args):
        # 單一訓練週期 (epoch) 的邏輯
        losses = []  # 用於記錄每個 batch 的損失
        # 使用 tqdm 顯示進度條
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        self.model.train()  # 將模型設定為訓練模式
        for batch_idx, (images) in enumerate(pbar):
            # 將圖片資料移至指定設備
            x = images.to(args.device)
            # 模型前向傳播，得到預測的 logits 和真實的 token 索引
            logits, z_indices = self.model.forward(x)
            # 計算交叉熵損失，忽略 MASK token 的位置
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), z_indices.view(-1), ignore_index=self.model.mask_token_id)
            # 反向傳播計算梯度
            loss.backward()
            losses.append(loss.item())
            
            # 梯度累積：每 accum_grad 個 batch 才更新一次權重
            if (batch_idx + 1) % args.accum_grad == 0:
                self.optim.step()  # 更新模型權重
                self.optim.zero_grad()  # 清除梯度
                
            # 在進度條上顯示當前的損失和學習率
            pbar.set_postfix(loss=loss.item(), lr=self.optim.param_groups[0]['lr'])
            
            # 如果啟用 wandb，則記錄當前 batch 的資訊
            if args.use_wandb and batch_idx % args.wandb_log_interval == 0:
                wandb.log({
                    "batch": batch_idx + epoch * len(train_dataloader),
                    "train_batch_loss": loss.item(),
                    "learning_rate": self.optim.param_groups[0]['lr']
                })
                
        avg_loss = np.mean(losses)  # 計算整個 epoch 的平均損失
        print(f"Epoch {epoch}/{args.epochs}, Average Training Loss: {avg_loss:.4f}")
        
        # 如果有設定學習率排程器，則更新學習率
        if self.scheduler is not None:
            self.scheduler.step()
            
        return avg_loss

    def eval_one_epoch(self, val_dataloader, epoch, args):
        # 單一驗證週期 (epoch) 的邏輯
        self.model.eval()  # 將模型設定為評估模式
        losses = []
        with torch.no_grad():  # 在評估時不計算梯度
            for batch_idx, (images) in enumerate(val_dataloader):
                x = images.to(self.device)
                logits, z_indices = self.model.forward(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), z_indices.view(-1), ignore_index=self.model.mask_token_id)
                losses.append(loss.item())
        avg_loss = np.mean(losses)
        print(f"Validation Loss: {avg_loss:.4f}")
        return avg_loss
    
    def configure_optimizers(self, args):
        # 設定優化器和學習率排程器
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.learning_rate)
        if args.scheduler == 'cosine':
            # 使用餘弦學習率排程器
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
        elif args.scheduler == 'none':
            scheduler = None
        return optimizer, scheduler


if __name__ == '__main__':
    # --- 命令列參數設定 ---
    parser = argparse.ArgumentParser(description="MaskGIT Transformer 訓練腳本")
    
    # --- 路徑相關 ---
    parser.add_argument('--train_d_path', type=str, default="./lab3_dataset/train/", help='訓練資料集路徑')
    parser.add_argument('--val_d_path', type=str, default="./lab3_dataset/val/", help='驗證資料集路徑')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints_transformer', help='模型檢查點儲存路徑')
    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='MaskGit 模型設定檔路徑')

    # --- 硬體與資料讀取相關 ---
    parser.add_argument('--device', type=str, default="cuda:0", help='指定訓練設備 (例如 "cuda:0" 或 "cpu")')
    parser.add_argument('--num_workers', type=int, default=4, help='資料讀取器使用的人工數量')
    parser.add_argument('--batch-size', type=int, default=40, help='訓練時的批次大小')
    parser.add_argument('--partial', type=float, default=1.0, help='使用資料集的比例 (1.0 代表全部使用)')    
    parser.add_argument('--accum-grad', type=int, default=5, help='梯度累積的步數')

    # --- 訓練超參數 ---
    parser.add_argument('--epochs', type=int, default=10, help='總訓練週期數')
    parser.add_argument('--save-per-epoch', type=int, default=5, help='每 N 個週期儲存一次檢查點')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='從指定的週期開始訓練 (用於恢復訓練)')
    parser.add_argument('--learning-rate', type=float, default=5e-5, help='學習率')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'none'], help='學習率排程器類型')
    parser.add_argument('--gamma-type', type=str, default='cosine', choices=['cosine', 'linear', 'square'], help='遮罩排程 (Mask Scheduling) 使用的 gamma 函數類型')

    # --- Wandb (Weights & Biases) 相關 ---
    parser.add_argument('--use_wandb', action='store_true', help='啟用 Wandb 進行實驗追蹤與紀錄')
    parser.add_argument('--wandb_project', type=str, default='MaskGit-Transformer-Training', help='Wandb 專案名稱')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Wandb 的實體名稱 (通常是您的帳號名)')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='為這次執行在 Wandb 上命名')
    parser.add_argument('--wandb_log_interval', type=int, default=10, help='每 N 個 batch 紀錄一次資訊到 Wandb')

    args = parser.parse_args()

    # 載入模型設定檔
    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    # 使用命令列參數覆寫 gamma 函數類型
    MaskGit_CONFIGS["model_param"]["gamma_type"] = args.gamma_type

    # --- Wandb 初始化 ---
    if args.use_wandb:
        # 準備要記錄到 wandb 的設定
        wandb_config = {
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "scheduler": args.scheduler,
            "accum_grad": args.accum_grad,
            "device": args.device,
        }
        
        # 初始化 wandb run
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config=wandb_config
        )
        
        # 將模型設定也加入到 wandb config 中
        wandb.config.update(MaskGit_CONFIGS["model_param"])
    
    # 建立模型檢查點資料夾
    os.makedirs(args.checkpoint_path, exist_ok=True)
    
    # 實例化訓練器
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    # --- 準備資料集 ---
    train_dataset = LoadTrainData(root=args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root=args.val_d_path, partial=args.partial)
    val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
    # --- 訓練迴圈 ---
    best_train = float('inf')
    best_val = float('inf')
    for epoch in range(args.start_from_epoch + 1, args.epochs + 1):
        train_loss = train_transformer.train_one_epoch(train_loader, epoch, args)
        val_loss = train_transformer.eval_one_epoch(val_loader, epoch, args)
        
        # 如果啟用 wandb，記錄整個 epoch 的指標
        if args.use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            })
        
        # --- 儲存模型檢查點 ---
        if epoch % args.save_per_epoch == 0:
            checkpoint = {
                'model': train_transformer.model.state_dict(),
                'optimizer': train_transformer.optim.state_dict(),
                'epoch': epoch,
                'best_train': best_train,
                'best_val': best_val,
                'args': args
            }
            if args.scheduler is not None:
                checkpoint['scheduler'] = train_transformer.scheduler.state_dict()
            
            torch.save(checkpoint, f"{args.checkpoint_path}/checkpoint_epoch_{epoch}.pth")
            
            # 如果啟用 wandb，將檢查點檔案也上傳
            if args.use_wandb:
                # 手動複製檔案以避免 symlink 權限問題
                src = f"{args.checkpoint_path}/checkpoint_epoch_{epoch}.pth"
                dst = os.path.join(wandb.run.dir, src)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy(src, dst)

        # 如果當前訓練損失是史上最好，儲存模型
        if train_loss < best_train:
            best_train = train_loss
            torch.save(train_transformer.model.transformer.state_dict(), f"{args.checkpoint_path}/best_train.pth")
            if args.use_wandb:
                wandb.run.summary["best_train_loss"] = best_train
                # 手動複製檔案以避免 symlink 權限問題
                src = f"{args.checkpoint_path}/best_train.pth"
                dst = os.path.join(wandb.run.dir, src)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy(src, dst)
                
        # 如果當前驗證損失是史上最好，儲存模型
        if val_loss < best_val:
            best_val = val_loss
            torch.save(train_transformer.model.transformer.state_dict(), f"{args.checkpoint_path}/best_val.pth")
            if args.use_wandb:
                wandb.run.summary["best_val_loss"] = best_val
                # 手動複製檔案以避免 symlink 權限問題
                src = f"{args.checkpoint_path}/best_val.pth"
                dst = os.path.join(wandb.run.dir, src)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy(src, dst)
    
    # --- 結束 Wandb ---
    if args.use_wandb:
        wandb.finish()