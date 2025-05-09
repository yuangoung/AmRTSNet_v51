import torch
import torch.nn.functional as F
import torch.nn as nn
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.attention_mechanism.TripletAttention import TripletAttention


def conv_bn(inp, oup, stride, BatchNorm):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=3, stride=stride, padding=1, bias=False),
        BatchNorm(oup),
        nn.ReLU6(inplace=True)
    )


def fixed_padding(inputs, kernel_size, dilation):
    """
    对输入进行等效的扩张卷积 padding。
    """
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    # F.pad 格式: (pad_left, pad_right, pad_top, pad_bottom)
    return F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, dilation, expand_ratio, BatchNorm):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = (self.stride == 1 and inp == oup)

        # 构建通道扩张/深度/逐点的 conv 序列
        if expand_ratio == 1:
            # 只有 depthwise + pointwise-linear
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 0,
                          dilation, groups=hidden_dim, bias=False),
                BatchNorm(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm(oup),
            )
        else:
            # pointwise, depthwise, pointwise-linear
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                BatchNorm(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 0,
                          dilation, groups=hidden_dim, bias=False),
                BatchNorm(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                BatchNorm(oup),
            )

    def forward(self, x):
        # 根据 conv[0] 是否有 dilation 属性，动态获取 kernel_size 与 dilation
        first_conv = self.conv[0]
        dilation = first_conv.dilation[0] if hasattr(first_conv, 'dilation') else 1
        x_pad = fixed_padding(x, kernel_size=3, dilation=dilation)
        out = self.conv(x_pad)

        if self.use_res_connect:
            # 若形状不一致，则居中裁剪 out，使其与 x 尺寸对齐
            if out.size(2) != x.size(2) or out.size(3) != x.size(3):
                diffH = out.size(2) - x.size(2)
                diffW = out.size(3) - x.size(3)
                # 对 H/W 居中裁剪
                out = out[:, :,
                          diffH // 2 : diffH // 2 + x.size(2),
                          diffW // 2 : diffW // 2 + x.size(3)]
            return x + out
        else:
            return out


class MobileNetV2(nn.Module):
    def __init__(self, output_stride=16,
                 BatchNorm=SynchronizedBatchNorm2d,
                 width_mult=1.0,
                 pretrained=False):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        # 初始通道数
        input_channel = int(32 * width_mult)
        current_stride = 1
        rate = 1

        # 配置列表：t=expand_ratio, c=output_channels, n=repeats, s=stride
        interverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 296, 1, 1],
        ]

        # --- 第一层：4 通道输入 ---
        features = [conv_bn(4, input_channel, 2, BatchNorm)]
        current_stride *= 2

        # 构建倒残差块序列
        for t, c, n, s in interverted_residual_setting:
            if current_stride == output_stride:
                stride = 1
                dilation = rate
                rate *= s
            else:
                stride = s
                dilation = 1
                current_stride *= s

            output_channel = int(c * width_mult)
            for i in range(n):
                features.append(
                    block(input_channel, output_channel,
                          stride if i == 0 else 1,
                          dilation, t, BatchNorm))
                input_channel = output_channel

        # 将特征层打包
        self.features = nn.Sequential(*features)

        # 注意力模块
        self.attention = TripletAttention()
        # 切分
        self.low_level_features = self.features[:4]
        self.high_level_features = self.features[4:]

        # 权重初始化
        self._initialize_weights()

    def forward(self, x):
        low_feat = self.low_level_features(x)
        x = self.high_level_features(low_feat)
        # 注意力
        low_feat = self.attention(low_feat)
        x = self.attention(x)
        return x, low_feat

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d) or isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == "__main__":
    # 测试四通道输入
    input_tensor = torch.rand(1, 4, 513, 513)
    model = MobileNetV2(output_stride=32,
                        BatchNorm=nn.BatchNorm2d,
                        width_mult=1.0,
                        pretrained=False)
    out, low = model(input_tensor)
    print(f"Output shape: {out.shape}")
    print(f"Low-level shape: {low.shape}")
