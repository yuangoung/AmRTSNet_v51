#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.attention_mechanism.sge import SpatialGroupEnhance
from modeling.decoder_optimized import build_decoder
from modeling.backbone import build_backbone

class AmRTSNet(nn.Module):
    def __init__(self, backbone='mobilenet', output_stride=8, num_classes=2,
                 sync_bn=True, freeze_bn=False):
        super(AmRTSNet, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        BatchNorm = SynchronizedBatchNorm2d if sync_bn else nn.BatchNorm2d

        # Backbone 提取高低层特征
        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        # Spatial Group Enhance 注意力
        self.SGE = SpatialGroupEnhance(groups=8)
        # Decoder
        self.decoder_yy = build_decoder(num_classes, backbone, BatchNorm)
        self.freeze_bn = freeze_bn

    def forward(self, input):
        # input: [B, C=3 or 4, H, W]
        highF, lowF = self.backbone(input)
        # 若要在前向中使用 SGE，可以解开以下注释：
        # lowF = self.SGE(lowF)
        # highF = self.SGE(highF)
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

# Standalone test & complexity measurement
if __name__ == "__main__":
    # 简单前向测试
    dummy_input = torch.randn(1, 4, 512, 512)  # 若四波段替换为 4
    model = AmRTSNet(backbone='mobilenet', output_stride=8, num_classes=2,
                    sync_bn=False, freeze_bn=False)
    model.eval()
    out = model(dummy_input)
    print(f"DeepLab output size: {out.size()}")  # e.g. [1, 2, 512, 512]

    # 计算 SGE 模块的参数量和计算量
    try:
        from thop import profile
    except ImportError:
        raise ImportError("请先安装 thop: pip install thop")

    # 实例化 SpatialGroupEnhance
    sge = SpatialGroupEnhance(groups=8)
    sge.eval()
    # dummy 输入需为 low-level 特征通道数：MobileNet 对应 lowF 通道数 24, 分辨率可选
    dummy_sge_in = torch.randn(1, 24, 128, 128)
    flops, params = profile(sge, inputs=(dummy_sge_in,), verbose=False)

    # 直接输出参数总数和 MFLOPs
    print(f"SpatialGroupEnhance 参数量: {int(params)} Params")
    print(f"SpatialGroupEnhance FLOPs:  {flops/1e6:.2f} MFLOPs")
