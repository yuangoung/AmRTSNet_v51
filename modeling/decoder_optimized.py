import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.attention_mechanism.ECAAttention import ECAAttention


class OptimizedDecoder(nn.Module):
    """
    最优化 Decoder，实现高低层特征融合并按照指定解码流程:
    1) 上采样高层特征至低层分辨率并与低层特征拼接
    2) 1x1 Conv 降维至 96 通道
    3) ConvTranspose2d 上采样至 256x256, 通道 96->64
    4) Dilated Conv 降维 64->32
    5) ConvTranspose2d 上采样至 512x512, 通道 32->16
    6) 1x1 Conv 分类 16->num_classes
    """

    def __init__(self, num_classes, backbone, BatchNorm):
        super().__init__()
        # MobileNet Backbone
        if backbone != 'mobilenet':
            raise NotImplementedError("This decoder only supports 'mobilenet' backbone.")

        # 低层与高层特征通道数
        low_ch = 24  # 128x128x24
        high_ch = 296  # 16x16x296

        # Stage0: 放大低层特征通道维数 -> 128x128, conv1x1 升为 64 通道
        self.conv1 = nn.Conv2d(low_ch, 64, 1, bias=False)
        self.bn1 = BatchNorm(64)
        self.relu1 = nn.ReLU()

        # Stage1: 融合高低层特征 -> 128x128, conv1x1 降为 96 通道
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(64 + high_ch, 96, kernel_size=1, bias=False),
            BatchNorm(96),
            nn.ReLU(inplace=True)
        )
        # 注意力模块
        self.ECA = ECAAttention()

        # Stage2: 上采样至 256x256, 通道 96->64
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(96, 64, kernel_size=4, stride=2, padding=1, bias=False),
            BatchNorm(64),
            nn.ReLU(inplace=True)
        )

        # Stage3: 空洞卷积降维 64->32, 保持 256x256
        self.dilate = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=2, dilation=2, bias=False),
            BatchNorm(32),
            nn.ReLU(inplace=True)
        )

        # Stage4: 上采样至 512x512, 通道 32->16
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False),
            BatchNorm(16),
            nn.ReLU(inplace=True)
        )

        # Stage5: 最终分类 16->num_classes
        self.classifier = nn.Conv2d(16, num_classes, kernel_size=1, bias=True)

        self._init_weight()

    def forward(self, x, low_level_feat):
        # x: 高层特征, [B, 296, 16, 16]
        # low_level_feat: 低层特征, [B, 64, 128, 128]
        # 放大底层特征通道维数
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu1(low_level_feat)
        # 上采样高层特征至低层分辨率
        high_up = F.interpolate(x,
                                size=low_level_feat.shape[2:],
                                mode='bilinear',
                                align_corners=True)
        # 融合拼接
        fusion = torch.cat([high_up, low_level_feat], dim=1)  # [B, 360, 128, 128]
        # 注意力模块

        fusion = self.ECA(fusion)

        y = self.fuse_conv(fusion)  # [B, 96, 128, 128]

        # 上采样至 256x256
        y = self.up1(y)  # [B, 64, 256, 256]
        # 空洞卷积
        y = self.dilate(y)  # [B, 32, 256, 256]
        # 上采样至 512x512
        y = self.up2(y)  # [B, 16, 512, 512]
        # 分类
        y = self.classifier(y)  # [B, num_classes, 512, 512]
        return y

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, SynchronizedBatchNorm2d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_decoder(num_classes, backbone, BatchNorm):
    return OptimizedDecoder(num_classes, backbone, BatchNorm)
