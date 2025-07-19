'''
Name: DLP Lab2
Topic: Binary Semantic Segmentation
Author: CHEN, KE-RONG
Date: 2025/07/10
'''

""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F



class DoubleConv(nn.Module):
    """
    一個包含兩個卷積層的區塊: (卷積 -> Batch Normalization -> ReLU) * 2
    Conv->BatchNorm->ReLU->Conv->BatchNorm->ReLU
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels # 預設用out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    '''
    kernel size = 3：3×3 卷積（如 U-Net 論文）
    padding = 1：保留輸入尺寸（不縮小 feature map）
    bias=False：因為 BatchNorm 已經有偏差項
    接 BatchNorm：加速訓練與穩定收斂
    接 ReLU(inplace=True)：非線性轉換
    '''

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    下採樣區塊: MaxPool -> DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),    # maxpool 2x2
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

        # x = Maxpool2d(x)
        # x = DoubleConv(x)
        # return x


class Up(nn.Module):
    """
    上採樣區塊: 上採樣 -> DoubleConv
    """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # Up 涉及多個不同來源的輸入與手動操作 → 不能用 nn.Sequential


        # # 透過雙線性內插法進行上採樣
        # if bilinear:
        #     self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #     self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        # else:

        # 通道數會從1024->512->256->128->64
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1 是來自解碼器前一層的特徵圖, x2 是來自編碼器對應層的特徵圖 (skip connection)
        x1 = self.up(x1)

        # 處理維度不匹配的問題 (如果輸入圖片的長寬不是 16 的倍數，可能會發生)
        diffY = x2.size()[2] - x1.size()[2] # 高度差異
        diffX = x2.size()[3] - x1.size()[3] # 寬度差異

        # F.pad(input, [左, 右, 上, 下])
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,  # 左右padding
                        diffY // 2, diffY - diffY // 2]) # 上下padding

        # 將 skip connection 的特徵圖和上採樣後的特徵圖串接
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

class OutConv(nn.Module):
    """
    輸出卷積層
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        # 388* 388* 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    完整的 UNet 架構
    """
    def __init__(self, n_channels, n_classes):
        """
        n_channels: 輸入圖片的通道數 ( RGB 為 3)
        n_classes: 輸出的類別數量 (二元分割為 1, 狗、貓)
        """

        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # 編碼器 (Contracting Path)
        self.input = DoubleConv(n_channels, out_channels=64)
        self.down1 = Down(in_channels=64, out_channels=128)
        self.down2 = Down(in_channels=128, out_channels=256)
        self.down3 = Down(in_channels=256, out_channels=512)
        self.down4 = Down(in_channels=512, out_channels=1024)

        # 解碼器 (Expansive Path)
        self.up1 = Up(in_channels=1024, out_channels=512)
        self.up2 = Up(in_channels=512, out_channels=256)
        self.up3 = Up(in_channels=256, out_channels=128)
        self.up4 = Up(in_channels=128, out_channels=64)
        
        # 輸出層
        self.output = OutConv(in_channels=64, out_channels=n_classes)

    def forward(self, x):
        # 編碼器
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # 解碼器 + Skip Connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # 輸出
        return self.output(x)

    # def forward_debug(self, x):
    #     print(f"[Input] {x.shape}")
    #
    #     # Encoder
    #     x1 = self.input(x)
    #     print(f"[x1] after input: {x1.shape}")
    #
    #     x2 = self.down1(x1)
    #     print(f"[x2] after down1: {x2.shape}")
    #
    #     x3 = self.down2(x2)
    #     print(f"[x3] after down2: {x3.shape}")
    #
    #     x4 = self.down3(x3)
    #     print(f"[x4] after down3: {x4.shape}")
    #
    #     x5 = self.down4(x4)
    #     print(f"[x5] after down4: {x5.shape}")
    #
    #     # Decoder + Skip Connections
    #     x = self.up1(x5, x4)
    #     print(f"[x] after up1: {x.shape}")
    #
    #     x = self.up2(x, x3)
    #     print(f"[x] after up2: {x.shape}")
    #
    #     x = self.up3(x, x2)
    #     print(f"[x] after up3: {x.shape}")
    #
    #     x = self.up4(x, x1)
    #     print(f"[x] after up4: {x.shape}")
    #
    #     out = self.output(x)
    #     print(f"[Output] {out.shape}")
    #     return out


if __name__ == "__main__":

    x = torch.randn((1, 3, 572, 572))
    model = UNet(n_channels=3, n_classes=1)
    print(f"Model parameter count: {sum(p.numel() for p in model.parameters())}")


    # U-net run
    output = model(x)
    # output = model.forward_debug(x)


    # Print output shape
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")