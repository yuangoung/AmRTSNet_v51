#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.decoder_optimized import build_decoder
from modeling.backbone import build_backbone

# ----- CBAM 注意力模块 -----
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_cat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.channel_att(x)
        out = out * self.spatial_att(out)
        return out


class DeepLab(nn.Module):
    def __init__(self, backbone='mobilenet', output_stride=8, num_classes=2,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        BatchNorm = SynchronizedBatchNorm2d if sync_bn else nn.BatchNorm2d

        # Backbone 提取高低层特征
        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        # CBAM 注意力，用于低层特征（24 通道）和高层特征（296 通道）都可复用
        # 这里示例对低层特征加 CBAM；要对高层也加，请在 forward 中额外调用
        self.cbam_low  = CBAM(24)
        self.cbam_high = CBAM(296)

        # Decoder
        self.decoder_yy = build_decoder(num_classes, backbone, BatchNorm)
        self.freeze_bn = freeze_bn

    def forward(self, input):
        # input: [B, C=3 or 4, H, W]
        highF, lowF = self.backbone(input)
        # 在低层和高层特征上分别加 CBAM 注意力
        lowF  = self.cbam_low(lowF)
        highF = self.cbam_high(highF)
        x = self.decoder_yy(highF, lowF)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, (SynchronizedBatchNorm2d, nn.BatchNorm2d)):
                m.eval()

    def get_1x_lr_params(self):
        for m in self.backbone.modules():
            if isinstance(m, (nn.Conv2d, SynchronizedBatchNorm2d, nn.BatchNorm2d)):
                for p in m.parameters():
                    if p.requires_grad:
                        yield p

    def get_10x_lr_params(self):
        for m in self.decoder_yy.modules():
            if isinstance(m, (nn.Conv2d, SynchronizedBatchNorm2d, nn.BatchNorm2d)):
                for p in m.parameters():
                    if p.requires_grad:
                        yield p

# =======================================
# Standalone test & CBAM 复杂度测算
# =======================================
if __name__ == "__main__":
    # 1. 测试整个 DeepLab forward
    dummy_input = torch.randn(1, 4, 512, 512)  # 若 4 波段，改为 torch.randn(1,4,512,512)
    model = DeepLab(backbone='mobilenet', output_stride=8, num_classes=2,
                    sync_bn=False, freeze_bn=False)
    model.eval()
    out = model(dummy_input)
    print(f"DeepLab output size: {out.size()}")  # e.g. [1,2,512,512]

    # 2. 计算 CBAM 模块的参数量和 FLOPs
    try:
        from thop import profile
    except ImportError:
        raise ImportError("请先安装 thop：pip install thop")

    # CBAM 分别对低层 (C=24) 和高层 (C=296) 统计，以低层为例
    cbam = CBAM(296)
    cbam.eval()
    dummy_cbam_in = torch.randn(1, 296, 128, 128)  # 低层特征分辨率示例
    flops, params = profile(cbam, inputs=(dummy_cbam_in,), verbose=False)

    print(f"CBAM (C=24) 参数量: {int(params)} Params")
    print(f"CBAM (C=24) FLOPs:    {flops/1e6:.2f} MFLOPs")
