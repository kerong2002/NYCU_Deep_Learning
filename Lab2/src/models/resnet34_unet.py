'''
Name: DLP Lab2
Topic: Binary Semantic Segmentation
Author: CHEN, KE-RONG
Date: 2025/07/11
'''
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    """ResNet 的基本殘差塊"""
    expansion = 1  # 每個 block 不改變通道數

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x # 傳入的x，用來做skip connection
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None: # 若輸入/輸出維度不同
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNetEncoder(nn.Module):
    """從零開始實作的 ResNet34 編碼器"""
    def __init__(self, block, layers, n_channels=3):
        super(ResNetEncoder, self).__init__()
        self.inplanes = 64

        # 初始卷積層
        self.conv1 = nn.Conv2d(n_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 堆疊殘差塊
        self.layer1 = self._make_layer(block, planes=64, blocks=layers[0])
        self.layer2 = self._make_layer(block, planes=128, blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(block, planes=256, blocks=layers[2], stride=2)
        self.layer4 = self._make_layer(block, planes=512, blocks=layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # 尺寸要縮小，輸入與輸出為度相同才能out += identity
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x0 = self.relu(x)
        x = self.maxpool(x0)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x4, x3, x2, x1, x0

class DecoderBlock(nn.Module):
    """UNet 解碼器中的一個區塊"""
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            conv3x3(out_channels * 2, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)

        # 處理因下採樣導致的尺寸不匹配問題
        # 如果 skip connection 的特徵圖比升採樣後的特徵圖大，則進行填充
        if skip.size(2) != x.size(2) or skip.size(3) != x.size(3):
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

class ResNet34_UNet(nn.Module):
    """結合 ResNet34 編碼器和 UNet 解碼器的模型"""
    def __init__(self, n_channels=3, n_classes=1):
        super(ResNet34_UNet, self).__init__()

        # 編碼器
        self.encoder = ResNetEncoder(BasicBlock, [3, 4, 6, 3], n_channels=n_channels)

        # 解碼器
        self.decoder4 = DecoderBlock(in_channels=512, out_channels=256)
        self.decoder3 = DecoderBlock(in_channels=256, out_channels=128)
        self.decoder2 = DecoderBlock(in_channels=128, out_channels=64)
        self.decoder1 = DecoderBlock(in_channels=64, out_channels=64)

        # 最終的升採樣和輸出層
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            conv3x3(32, 32),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            conv1x1(32, n_classes)
        )

    def forward(self, x):
        # 編碼器路徑
        e4, e3, e2, e1, e0 = self.encoder(x)

        # 解碼器路徑
        d4 = self.decoder4(e4, e3)
        d3 = self.decoder3(d4, e2)
        d2 = self.decoder2(d3, e1)
        d1 = self.decoder1(d2, e0)

        # 最終輸出
        out = self.final_up(d1)
        out = self.final_conv(out)

        return out

    # def forward_debug(self, x):
    #     """
    #     一個包含詳細 print 輸出的 forward 版本，用於除錯。
    #     """
    #     print(f"[Input] shape: {x.shape}")
    #
    #     # --- 編碼器路徑 ---
    #     e4, e3, e2, e1, e0 = self.encoder(x)
    #     print(f"--- Encoder Outputs ---")
    #     print(f"[e0] after initial conv: {e0.shape}")
    #     print(f"[e1] after layer1:       {e1.shape}")
    #     print(f"[e2] after layer2:       {e2.shape}")
    #     print(f"[e3] after layer3:       {e3.shape}")
    #     print(f"[e4] after layer4:       {e4.shape}")
    #     print(f"-----------------------")
    #
    #     # --- 解碼器路徑 (帶有跳躍連接) ---
    #     print(f"\n--- Decoder Path ---")
    #     d4 = self.decoder4(e4, e3)
    #     print(f"[d4] after decoder4(e4, e3): {d4.shape}")
    #
    #     d3 = self.decoder3(d4, e2)
    #     print(f"[d3] after decoder3(d4, e2): {d3.shape}")
    #
    #     d2 = self.decoder2(d3, e1)
    #     print(f"[d2] after decoder2(d3, e1): {d2.shape}")
    #
    #     d1 = self.decoder1(d2, e0)
    #     print(f"[d1] after decoder1(d2, e0): {d1.shape}")
    #     print(f"----------------------")
    #
    #     # --- 最終輸出 ---
    #     print(f"\n--- Final Output ---")
    #     out = self.final_up(d1)
    #     print(f"[out] after final_up:   {out.shape}")
    #
    #     out = self.final_conv(out)
    #     print(f"[out] after final_conv: {out.shape}")
    #     print(f"--------------------")
    #
    #     return out

if __name__ == "__main__":

    x = torch.randn((1, 3, 572, 572))
    model = ResNet34_UNet(n_channels=3, n_classes=1)
    print(f"Model parameter count: {sum(p.numel() for p in model.parameters())}")


    # U-net run
    output = model(x)
    # output = model.forward_debug(x)


    # Print output shape
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")