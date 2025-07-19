'''
Name: DLP Lab2
Topic: Binary Semantic Segmentation
Author: CHEN, KE-RONG
Date: 2025/07/11
'''
import torch

def dice_score(pred_mask, gt_mask, threshold=0.5, eps=1e-8):
    """
    計算一個 batch 的 Dice Score，並回傳 batch 中所有圖片的平均分數。
    """
    with torch.no_grad():
        if pred_mask.dim() == 4 and pred_mask.size(1) == 1:
            pred_mask = pred_mask.squeeze(1)
        if gt_mask.dim() == 4 and gt_mask.size(1) == 1:
            gt_mask = gt_mask.squeeze(1)

        pred_bin = (torch.sigmoid(pred_mask) > threshold).float()
        gt_bin = (gt_mask > 0.5).float()

        intersection = (pred_bin * gt_bin).sum(dim=[1, 2])
        union = pred_bin.sum(dim=[1, 2]) + gt_bin.sum(dim=[1, 2])

        dice = (2 * intersection + eps) / (union + eps)
        return dice.mean().item()
