'''
Name: DLP Lab3
Topic: MaskGIT
Author: CHEN, KE-RONG
Date: 2025/07/15
'''
import torch
import torch.nn as nn
import yaml
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer


#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        # 載入預訓練的 VQGAN 模型
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])

        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        """ 載入 Transformer 的權重 """
        # print(f"從 {load_ckpt_path} 載入 Transformer 權重")
        self.transformer.load_state_dict(torch.load(load_ckpt_path))

    @staticmethod
    def load_vqgan(configs):
        """ 載入 VQGAN 模型 """
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True)
        model = model.eval()
        return model

##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        """ 將輸入影像編碼為離散的 token indices """
        '''
        codebook_mapping: [B, C, H, W] => 對應量化後的特徵（z_q_mapping）
        codebook_indices: [B, H, W] => 對應 codebook 向量的索引（z_indices）
        q_loss: quantization loss（訓練時才會用）
        '''

        z_q_mapping, z_indices, _ = self.vqgan.encode(x)
        # 將 z_indices reshape 成平坦的 token 序列
        z_indices = z_indices.reshape(z_q_mapping.shape[0], -1)  # [B, H*W]
        # 轉換成 [B, H*W] 的 token 序列格式，方便之後送進 Transformer
        return z_q_mapping, z_indices

##TODO2 step1-2:
    # 這是用來決定「每次 inpainting 迭代要遮多少 token」的函數：
    def gamma_func(self, mode="cosine"):
        """
        Mask scheduling 函數，根據給定的比例生成一個遮罩率。
        在訓練期間，輸入比例是均勻採樣的；在推論期間，輸入比例是 t/T。
        """
        if mode == "linear":
            return lambda r: 1.0 - r
        elif mode == "cosine":
            return lambda r: math.cos(r * math.pi / 2)
        elif mode == "square":
            return lambda r: 1.0 - r ** 2
        else:
            raise NotImplementedError(f"不支援的 gamma 模式: {mode}")

##TODO2 step1-3:
    def forward(self, image_batch):
        # 訓練時的前向傳播
        # image_batch: (B, C, H, W) -> (批次大小, 通道數, 高度, 寬度)
        
        # 1. 將輸入影像編碼成離散的 token 序列
        _, ground_truth_indices = self.encode_to_z(image_batch)
        
        # 2. 隨機生成遮罩
        # 建立一個與 token 序列形狀相同、值全為 MASK token ID 的張量
        mask_token_tensor = torch.full_like(ground_truth_indices, self.mask_token_id)
        # 依據 0.5 的機率，獨立為每個 token 位置決定是否要遮罩
        random_mask = torch.bernoulli(0.5 * torch.ones_like(ground_truth_indices, dtype=torch.float)).bool()
        
        # 3. 產生 Transformer 的輸入
        # 使用遮罩，將部分真實 token 替換為 MASK token
        masked_input_indices = torch.where(random_mask, mask_token_tensor, ground_truth_indices)
        
        # 4. 透過 Transformer 進行預測
        # Transformer 的目標是預測出被遮罩位置的原始 token
        predicted_logits = self.transformer(masked_input_indices)


        # 5. 回傳預測結果和真實標籤以計算損失
        return predicted_logits, ground_truth_indices

##TODO3 step1-1: define one iteration decoding
    @torch.no_grad()
    def inpainting(self, current_indices, current_mask, total_masked_count, iteration_ratio, mask_schedule_func):
        # 執行單步迭代式解碼 (inpainting)
        
        # 1. 準備 Transformer 輸入：將被遮罩的位置替換為 MASK token
        masked_input = torch.where(current_mask, self.mask_token_id, current_indices)
        
        # 2. 透過 Transformer 預測所有 token 的機率分佈
        logits = self.transformer(masked_input)
        token_probabilities = torch.softmax(logits, dim=-1)
        
        # 3. 從機率分佈中取樣，得到預測的 token
        predicted_indices = torch.distributions.Categorical(logits=logits).sample()
        
        # 4. 確保預測出的 token 不會是 MASK token
        while torch.any(predicted_indices == self.mask_token_id):
            predicted_indices = torch.distributions.Categorical(logits=logits).sample()
            
        # 5. 將預測出的 token 填入當前被遮罩的位置，形成一個完整的 token 序列
        updated_indices = torch.where(current_mask, predicted_indices, current_indices)

        # 6. 取得每個預測 token 的機率 (信心度)
        predicted_probs = token_probabilities.gather(-1, predicted_indices.unsqueeze(-1)).squeeze(-1)
        # 對於本來就未遮罩的位置，我們不考慮其信心度，設為無限大
        predicted_probs = torch.where(current_mask, predicted_probs, torch.inf)

        # 7. 根據 Mask Scheduling 函數計算下一步要保留的遮罩比例
        next_mask_ratio = self.gamma_func(mask_schedule_func)(iteration_ratio)
        
        # 8. 計算下一步要保留的遮罩數量
        num_masks_to_keep = torch.floor(total_masked_count * next_mask_ratio).long()

        # 9. 計算每個 token 的排序分數 (Gumbel-Max Trick)
        # 透過加入 Gumbel 噪音和溫度係數，增加取樣的隨機性，避免模型過於單調
        gumbel_noise = torch.distributions.Gumbel(0, 1).sample(predicted_probs.shape).to(predicted_probs.device)
        temperature = self.choice_temperature * (1 - iteration_ratio)
        ranking_scores = predicted_probs + temperature * gumbel_noise
        
        # 10. 決定下一步的遮罩
        # 找出排序分數最低的 token，這些是模型最不確定的部分，將在下一步繼續被遮罩
        sorted_scores, _ = torch.sort(ranking_scores, dim=-1)
        # 取得分數的截斷閾值
        confidence_threshold = sorted_scores[:, num_masks_to_keep].unsqueeze(-1)
        # 分數低於閾值的 token 位置，即為下一步的遮罩
        next_step_mask = (ranking_scores < confidence_threshold)
        
        return updated_indices, next_step_mask


__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
