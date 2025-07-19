import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import argparse
from utils import LoadTestData, LoadMaskData
from torch.utils.data import Dataset,DataLoader
from torchvision import utils as vutils
import os
from models import MaskGit as VQGANTransformer
import yaml
import torch.nn.functional as F

# 負責執行 MaskGIT 推論的類別
class MaskGIT:
    def __init__(self, args, MaskGit_CONFIGS):
        # 初始化模型並移至指定設備
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        # 載入訓練好的 Transformer 權重
        self.model.load_transformer_checkpoint(args.load_transformer_ckpt_path)
        self.model.eval()  # 設定為評估模式
        
        # 從參數讀取推論設定
        self.total_iter = args.total_iter
        self.mask_func = args.mask_func
        self.sweet_spot = args.sweet_spot
        self.device = args.device
        
        # 準備儲存結果的資料夾
        self.prepare_folders()

    @staticmethod
    def prepare_folders():
        # 建立儲存最終結果、遮罩過程和中間圖片的資料夾
        os.makedirs("test_results", exist_ok=True)
        os.makedirs("mask_scheduling", exist_ok=True)
        os.makedirs("imga", exist_ok=True)
        os.makedirs("masked_inputs", exist_ok=True)

    def inpainting(self, image, mask_b, i):
        # 執行圖像修復 (Inpainting) 的主函式
        
        # 用於儲存每一步迭代的遮罩和解碼圖片，方便視覺化
        maska = torch.zeros(self.total_iter, 3, 16, 16)
        imga = torch.zeros(self.total_iter + 1, 3, 64, 64)
        
        # 準備反標準化所需的均值和標準差
        mean = torch.tensor([0.4868, 0.4341, 0.3844], device=self.device).view(3, 1, 1)  
        std = torch.tensor([0.2620, 0.2527, 0.2543], device=self.device).view(3, 1, 1)
        
        # 將第一張圖設為被遮罩的原始輸入影像
        ori = (image[0] * std) + mean
        imga[0] = ori
        # 儲存被遮罩的輸入影像，方便後續製作對比圖
        vutils.save_image(ori, os.path.join("masked_inputs", f"image_{i:03d}.png"), nrow=1)

        self.model.eval()
        with torch.no_grad():
            # 將輸入影像編碼為離散的 token 序列
            _, z_indices = self.model.encode_to_z(image[0].unsqueeze(0))
            mask_num = mask_b.sum()  # 計算總共有多少個 token 被遮罩
            
            z_indices_predict = z_indices
            mask_bc = mask_b.to(device=self.device)
            
            # 迭代式解碼迴圈
            for step in range(self.total_iter):
                # "sweet_spot" 是一個可以提前終止迭代的技巧，如果設為 -1 則跑完全部迭代
                if step == self.sweet_spot:
                    break

                # 計算當前迭代的比例
                ratio = (step + 1) / self.total_iter
    
                # 呼叫模型進行單步 inpainting
                z_indices_predict, mask_bc = self.model.inpainting(z_indices_predict, mask_bc, mask_num, ratio, self.mask_func)

                # --- 以下為視覺化過程，儲存中間結果 ---
                mask_i = mask_bc.view(1, 16, 16)
                mask_image = torch.ones(3, 16, 16)
                indices = torch.nonzero(mask_i, as_tuple=False)
                mask_image[:, indices[:, 1], indices[:, 2]] = 0
                maska[step] = mask_image
                
                # 將預測的 token 解碼回影像
                shape = (1, 16, 16, 256)
                z_q = self.model.vqgan.codebook.embedding(z_indices_predict).view(shape)
                z_q = z_q.permute(0, 3, 1, 2)
                decoded_img = self.model.vqgan.decode(z_q)
                dec_img_ori = (decoded_img[0] * std) + mean
                imga[step + 1] = dec_img_ori

            # 儲存最終的修復結果，這個資料夾將用於 FID 分數計算
            vutils.save_image(dec_img_ori, os.path.join("test_results", f"image_{i:03d}.png"), nrow=1) 
            
            # 儲存過程視覺化圖片
            vutils.save_image(maska, os.path.join("mask_scheduling", f"test_{i}.png"), nrow=10) 
            vutils.save_image(imga, os.path.join("imga", f"test_{i}.png"), nrow=7)


# 負責載入被遮罩圖片和對應遮罩的類別
class MaskedImage:
    def __init__(self, args):
        # 載入被遮罩的測試圖片
        mi_ori = LoadTestData(root=args.test_maskedimage_path, partial=args.partial)
        self.mi_ori = DataLoader(mi_ori, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True, pin_memory=True, shuffle=False)
        
        # 載入對應的遮罩
        mask_ori = LoadMaskData(root=args.test_mask_path, partial=args.partial)
        self.mask_ori = DataLoader(mask_ori, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True, pin_memory=True, shuffle=False)
        
        self.device = args.device

    def get_mask_latent(self, mask):
        # 將 64x64 的遮罩降採樣到 16x16，以匹配 latent space 的大小
        downsampled1 = torch.nn.functional.avg_pool2d(mask, kernel_size=2, stride=2)
        resized_mask = torch.nn.functional.avg_pool2d(downsampled1, kernel_size=2, stride=2)
        resized_mask[resized_mask != 1] = 0
        mask_tokens = (resized_mask[0][0] // 1).flatten()
        mask_tokens = mask_tokens.unsqueeze(0)
        # 產生布林遮罩，True 代表該位置需要被修復
        mask_b = torch.zeros(mask_tokens.shape, dtype=torch.bool, device=self.device)
        mask_b |= (mask_tokens == 0)
        return mask_b


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT Inpainting 推論腳本")
    
    # --- 硬體與基本設定 ---
    parser.add_argument('--device', type=str, default="cuda", help='指定推論設備 (例如 "cuda" 或 "cpu")')
    parser.add_argument('--batch-size', type=int, default=1, help='測試時的批次大小 (通常設為 1)')
    parser.add_argument('--partial', type=float, default=1.0, help='使用測試資料集的比例')    
    parser.add_argument('--num_workers', type=int, default=4, help='資料讀取器的人工數量')
    
    # --- 模型與路徑設定 ---
    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='MaskGit 模型設定檔路徑')
    parser.add_argument('--load-transformer-ckpt-path', type=str, default='./checkpoints_transformer/best_val.pth', help='載入訓練好的 Transformer 權重路徑')
    parser.add_argument('--test-maskedimage-path', type=str, default='./lab3_dataset/masked_image', help='被遮罩的測試圖片資料夾路徑')
    parser.add_argument('--test-mask-path', type=str, default='./lab3_dataset/mask64', help='測試遮罩資料夾路徑')
    
    # --- MVTM (Mask VQGAN Transformer Masking) 推論參數 ---
    parser.add_argument('--sweet-spot', type=int, default=-1, help='最佳迭代步數 (sweet spot)，設為 -1 代表跑完全部迭代')
    parser.add_argument('--total-iter', type=int, default=7, help='總迭代步數')
    parser.add_argument('--mask-func', type=str, default='cosine', choices=['cosine', 'linear', 'square'], help='遮罩排程函數')

    args = parser.parse_args()

    # 實例化資料載入器
    t = MaskedImage(args)
    # 載入模型設定
    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    # 實例化 MaskGIT 推論器
    maskgit = MaskGIT(args, MaskGit_CONFIGS)

    i = 0
    # 遍歷所有測試圖片和遮罩
    for image, mask in zip(t.mi_ori, t.mask_ori):
        image = image.to(device=args.device)
        mask = mask.to(device=args.device)
        # 取得 latent space 的遮罩
        mask_b = t.get_mask_latent(mask)       
        # 執行 inpainting
        maskgit.inpainting(image, mask_b, i)
        i += 1
