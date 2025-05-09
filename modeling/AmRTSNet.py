import torch.nn as nn
import torch
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

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)

        # self.ACmix = ACmix()
        # self.CCNET =CrissCrossAttention(320)
        # self.CBAM24 = CBAM(24)
        # self.CBAM298= CBAM(296)
        self.SGE = SpatialGroupEnhance(groups=8)

        self.decoder_yy = build_decoder(num_classes, backbone, BatchNorm)
        self.freeze_bn = freeze_bn

 # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def forward(self, input):
        highF, lowF = self.backbone(input)

        lowF_SGE = self.SGE(lowF)
        highF_SGE = self.SGE(highF)

        # lowF_SGE = self.CBAM24(lowF)
        # highF_SGE = self.CBAM298(highF)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        x = self.decoder_yy(highF_SGE, lowF_SGE)

        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.decoder_yy]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p


class Tiff:
    def __init__(self, filename):
        self.filename = filename
        self.im_band = None
        self.im_width = None
        self.im_height = None
        self.im_band = None
        self.im_proj = None
        self.im_geotrans = None
        self.im_data = None
        self.overall_img = None

    def read_img(self):
        dataset = gdal.Open(self.filename)

        self.im_width = dataset.RasterXSize
        self.im_height = dataset.RasterYSize

        self.im_geotrans = dataset.GetGeoTransform()
        self.im_proj = dataset.GetProjection()
        self.im_data = dataset.ReadAsArray(0, 0, self.im_width, self.im_height)
        self.im_data = self.im_data[[0, 1, 2], :, :]
        self.im_band = self.im_data.shape[0]

        return self.im_data
        del dataset

if __name__ == "__main__":
    input = torch.rand(1, 4, 512, 512)
    model = AmRTSNet(backbone='mobilenet', output_stride=8)
    model.eval()
    output = model(input)
    print(output.size())


