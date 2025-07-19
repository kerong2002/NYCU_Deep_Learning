'''
Name: DLP Lab3
Topic: MaskGIT
Author: CHEN, KE-RONG
Date: 2025/07/15
'''
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, dim=768, num_heads=16, attn_drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5 # sqrt(d_k)

        self.query_proj = nn.Linear(dim, dim, bias=False)
        self.key_proj = nn.Linear(dim, dim, bias=False)
        self.value_proj = nn.Linear(dim, dim, bias=False)
        self.output_proj = nn.Linear(dim, dim, bias=False)

        self.attn_dropout = nn.Dropout(attn_drop)

    def forward(self, x):
        batch_size, num_tokens, _ = x.shape

        # 步驟 1: 將輸入分別投影到 Q, K, V
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)

        # 步驟 2: 將總維度 D 拆分為 H 個頭 (num_heads)
        # 維度從 (B, N, D) -> (B, N, H, d_h)
        q_reshaped = q.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        k_reshaped = k.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        v_reshaped = v.view(batch_size, num_tokens, self.num_heads, self.head_dim)

        # 步驟 3: 交換 N 和 H 維度，以便進行批次化矩陣運算
        # 維度從 (B, N, H, d_h) -> (B, H, N, d_h)
        q_permuted = q_reshaped.permute(0, 2, 1, 3)
        k_permuted = k_reshaped.permute(0, 2, 1, 3)
        v_permuted = v_reshaped.permute(0, 2, 1, 3)

        # 步驟 4: 計算Scaled Dot-Product Attention
        # (B, H, N, d) @ (B, H, d, N) -> (B, H, N, N)
        attention_scores = torch.matmul(q_permuted, k_permuted.transpose(-2, -1)) * self.scale
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.attn_dropout(attention_probs)

        # 步驟 5: 將注意力權重應用於 V
        # (B, H, N, N) @ (B, H, N, d) -> (B, H, N, d)
        context_layer = torch.matmul(attention_probs, v_permuted)

        # 步驟 6: 重新組合多頭的結果，首先反轉 permute 操作
        # (B, H, N, d) -> (B, N, H, d)
        context_layer_permuted = context_layer.permute(0, 2, 1, 3)

        # 步驟 7: 將 H 和 d_h 維度合併回 D
        # (B, N, H, d) -> (B, N, D)
        # .contiguous() 確保張量在記憶體中是連續的，這是 view/reshape 前的好習慣
        context_layer_reshaped = context_layer_permuted.contiguous().view(batch_size, num_tokens, self.num_heads * self.head_dim)

        # 步驟 8: 通過最終的輸出投影層
        final_output = self.output_proj(context_layer_reshaped)
        return final_output


class MLP(nn.Sequential):
    def __init__(self, dim=768, hidden_dim=3072, drop_rate=0.1):
        super(MLP, self).__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=0.1)
        )

    def forward(self, input):
        return super().forward(input)


class TokenPredictor(nn.Sequential):
    def __init__(self, dim=768):
        super(TokenPredictor, self).__init__(
            nn.Linear(in_features=dim, out_features=dim),
            nn.GELU(),
            nn.LayerNorm(dim, eps=1e-12)
        )

    def forward(self, input):
        return super().forward(input)


class Encoder(nn.Module):
    def __init__(self, dim=768, hidden_dim=1536):
        super(Encoder, self).__init__()
        self.Attention = MultiHeadAttention(dim)
        self.LayerNorm1 = nn.LayerNorm(dim, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(dim, eps=1e-12)
        self.MLP = MLP(dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attn = self.Attention(x)
        attn = self.dropout(attn)

        x = x + attn
        x = self.LayerNorm1(x)

        mlp = self.MLP(x)
        x = x + mlp
        return self.LayerNorm2(x)

if __name__ == "__main__":
    MHA = MultiHeadAttention()
    x = torch.rand(2, 256, 768)
    out = MHA(x)
    print("output shape:", out.shape)
