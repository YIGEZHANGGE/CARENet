# -*- coding: utf-8 -*-
# Python 3.7.12 / PyTorch 1.11
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class CBAM(nn.Module):
    """ Convolutional Block Attention Module """
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(in_channels)  # 传入 in_channels
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        batch_size = x.size(0)
        avg_out = self.fc(self.avg_pool(x).view(batch_size, -1))
        max_out = self.fc(self.max_pool(x).view(batch_size, -1))
        out = avg_out + max_out
        scale = self.sigmoid(out).view(batch_size, -1, 1, 1)
        return x * scale

class SpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 修改为输出 in_channels 通道
        self.conv = nn.Conv2d(2, in_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(x_cat)
        scale = self.sigmoid(attention)
        return x * scale  # 现在可以正确广播

class ResBlock(nn.Module):
    """ Residual block with CBAM attention """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.attention = CBAM(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.attention(out)
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out

class CNNWithResAttention(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(CNNWithResAttention, self).__init__()
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 5 convolutional layers with ResBlocks
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 1, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layer for classification
        self.fc = nn.Linear(256, num_classes)
        
    def _make_layer(self, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResBlock(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class AdaptiveFeatureReCalibration(nn.Module):
    """修正后的自适应特征重校准模块（1D版本）"""
    def __init__(self, channels, reduction_ratio=16):
        super(AdaptiveFeatureReCalibration, self).__init__()
        self.channels = channels
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 输入x的形状应为(batch_size, channels)
        y = self.fc(x)
        return x * y

class DynamicSparseActivation(nn.Module):
    """动态稀疏激活函数"""
    def __init__(self, alpha=0.01):
        super(DynamicSparseActivation, self).__init__()
        self.alpha = alpha
        self.threshold = Parameter(torch.tensor(0.0))
    
    def forward(self, x):
        # 动态阈值计算
        threshold = torch.mean(x) + self.alpha * torch.std(x)
        self.threshold.data = threshold.detach()
        return torch.where(x > threshold, x, torch.zeros_like(x))

class ProgressiveFeatureFusion(nn.Module):
    """完全修正的渐进式特征融合模块"""
    def __init__(self, in_features, out_features):
        super(ProgressiveFeatureFusion, self).__init__()
        # 确保输出维度正确
        self.transform = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            DynamicSparseActivation()
        )
        self.afr = AdaptiveFeatureReCalibration(out_features)
    
    def forward(self, x_prev, x_current):
        # 首先将x_current投影到与transform相同的维度
        if x_current.size(1) != x_prev.size(1):
            x_current = nn.Linear(x_current.size(1), x_prev.size(1)).to(x_current.device)(x_current)
        
        transformed = self.transform(x_prev + x_current)  # 先融合再变换
        return self.afr(transformed)

class MultiScaleFeatureExtractor(nn.Module):
    """修正后的多尺度特征提取模块"""
    def __init__(self, in_features, out_features):
        super(MultiScaleFeatureExtractor, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Linear(in_features, out_features // 2),
            nn.BatchNorm1d(out_features // 2),
            nn.GELU()
        )
        self.branch2 = nn.Sequential(
            nn.Linear(in_features, out_features // 4),
            nn.BatchNorm1d(out_features // 4),
            nn.GELU(),
            nn.Linear(out_features // 4, out_features // 4),
            nn.BatchNorm1d(out_features // 4),
            nn.GELU()
        )
        self.merge = nn.Linear((out_features // 2) + (out_features // 4), out_features)
        self.afr = AdaptiveFeatureReCalibration(out_features)
    
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        merged = torch.cat([b1, b2], dim=1)
        out = self.merge(merged)
        return self.afr(out)
        
# ----------------------------
# CBAM: Channel + Spatial Attention
# ----------------------------
class ChannelAttention(nn.Module):
    """
    通道注意力：AvgPool/MaxPool -> 共享 MLP(1x1 conv) -> Sigmoid 通道权重
    """
    def __init__(self, in_channels: int, reduction: int = 16):
        super(ChannelAttention, self).__init__()
        hidden = max(in_channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, kernel_size=1, bias=False),
        )
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.mlp(self.avg_pool(x))
        mx  = self.mlp(self.max_pool(x))
        w = self.act(avg + mx)
        return x * w


class SpatialAttention(nn.Module):
    """
    空间注意力：沿通道做 Avg/Max -> 拼接 [B,2,H,W] -> 7x7 conv -> Sigmoid 空间权重
    """
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        s = torch.cat([avg, mx], dim=1)
        w = self.act(self.conv(s))
        return x * w


class CBAM(nn.Module):
    """
    标准 CBAM：ChannelAttention -> SpatialAttention
    """
    def __init__(self, in_channels: int, reduction: int = 16, sa_kernel: int = 7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_channels, reduction)
        self.sa = SpatialAttention(sa_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        x = self.sa(x)
        return x


# ----------------------------
# Residual Block（Basic，2个3x3）
# ----------------------------
class ResBlock(nn.Module):
    """
    结构：Conv3x3-BN-ReLU -> Conv3x3-BN -> (shortcut) -> ReLU
    支持 stride 下采样与 1x1 投影以匹配维度
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.proj = None
        if stride != 1 or in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.proj is not None:
            identity = self.proj(identity)
        out = self.act(out + identity)
        return out


# ----------------------------
# 5-Stage CNN Backbone: ResBlock + CBAM
# ----------------------------
class CBAMResNet5(nn.Module):
    """
    5 阶段特征提取骨干：
      stem -> [stage1..stage5] (ResBlock + CBAM) -> GAP -> feature vector

    参数：
      in_channels: 输入通道(灰度=1, 彩色=3)
      base_channels: 第一阶段通道基数
      channels: 每阶段输出通道（长度=5），默认 [64,128,256,256,512]
      reduction: CBAM 通道注意力 reduction
      sa_kernel: CBAM 空间注意力卷积核
      out_dim: 若不为 None，则使用线性+BN 投影到该维度；否则输出 C5 维特征
      return_pyramid: 若 True，则 forward 返回 (特征向量, [stage1..5特征图])
    """
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        channels: Optional[List[int]] = None,
        reduction: int = 16,
        sa_kernel: int = 7,
        out_dim: Optional[int] = None,
        return_pyramid: bool = False,
    ):
        super(CBAMResNet5, self).__init__()
        if channels is None:
            channels = [base_channels, base_channels*2, base_channels*4, base_channels*4, base_channels*8]
        assert len(channels) == 5, "channels 长度必须为 5"

        # stem：不做大步长，尽量保留细节（医学图像友好）
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        # 5 个阶段：stage2~5 下采样（stride=2）
        self.stage1 = nn.Sequential(
            ResBlock(channels[0], channels[0], stride=1),
            CBAM(channels[0], reduction, sa_kernel),
        )
        self.stage2 = nn.Sequential(
            ResBlock(channels[0], channels[1], stride=2),
            CBAM(channels[1], reduction, sa_kernel),
        )
        self.stage3 = nn.Sequential(
            ResBlock(channels[1], channels[2], stride=2),
            CBAM(channels[2], reduction, sa_kernel),
        )
        self.stage4 = nn.Sequential(
            ResBlock(channels[2], channels[3], stride=2),
            CBAM(channels[3], reduction, sa_kernel),
        )
        self.stage5 = nn.Sequential(
            ResBlock(channels[3], channels[4], stride=2),
            CBAM(channels[4], reduction, sa_kernel),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.out_dim = out_dim
        feat_dim = channels[-1]
        self.proj = None
        if out_dim is not None:
            self.proj = nn.Sequential(
                nn.Flatten(),
                nn.Linear(feat_dim, out_dim, bias=False),
                nn.BatchNorm1d(out_dim),
            )
            self.feature_dim = out_dim
        else:
            self.feature_dim = feat_dim

        self.return_pyramid = return_pyramid
        self._init_weights()

    def _init_weights(self):
        # Kaiming for convs / ones for BN gamma / zeros for BN beta
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @torch.no_grad()
    def get_feature_dim(self) -> int:
        return self.feature_dim

    def forward(self, x: torch.Tensor, return_intermediate: bool = False):
        feats = []
        x = self.stem(x)              # [B, C1, H, W]
        x = self.stage1(x); feats.append(x)
        x = self.stage2(x); feats.append(x)
        x = self.stage3(x); feats.append(x)
        x = self.stage4(x); feats.append(x)
        x = self.stage5(x); feats.append(x)

        g = self.gap(x)               # [B, C5, 1, 1]
        g = torch.flatten(g, 1)       # [B, C5]
        if self.proj is not None:
            g = self.proj(g)          # [B, out_dim]

        if return_intermediate or self.return_pyramid:
            return g, feats
        return g


# ----------------------------
# 可选：端到端分类头
# ----------------------------
class CBAMResNet5Classifier(nn.Module):
    """
    以特征骨干为主体，叠加一个线性分类头
    """
    def __init__(self, num_classes: int, **backbone_kwargs):
        super(CBAMResNet5Classifier, self).__init__()
        self.backbone = CBAMResNet5(**backbone_kwargs)
        self.classifier = nn.Linear(self.backbone.get_feature_dim(), num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        feats = self.backbone(x)  # [B, D]
        logits = self.classifier(feats)
        if return_features:
            return logits, feats
        return logits


# ----------------------------
# 运行示例（形状自检）
# ----------------------------
if __name__ == "__main__":
    # 例：3 通道 224x224（灰度医学图像可设 in_channels=1）
    x = torch.randn(4, 3, 224, 224)

    # 仅提取图像特征（投影到 256 维，适合检索/对比学习/多模态融合）
    backbone = CBAMResNet5(in_channels=3, base_channels=64, out_dim=256)
    feats = backbone(x)  # [4, 256]
    print("Feature shape:", feats.shape)

    # 返回金字塔特征（供 FPN/Transformer 融合）
    feats_vec, multi_feats = backbone(x, return_intermediate=True)
    print("Pyramid stages:", [f.shape for f in multi_feats])

    # 端到端二分类示例
    model = CBAMResNet5Classifier(
        num_classes=2,
        in_channels=3,
        base_channels=64,
        out_dim=None  # 不做投影，直接用 C5
    )
    logits, z = model(x, return_features=True)
    print("Logits:", logits.shape, "  feat_dim:", z.shape)
