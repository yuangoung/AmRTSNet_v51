from __future__ import print_function, division
import os
import numpy as np
import rasterio
from torch.utils.data import Dataset
from Dataloaders.DATApath import Path
from torchvision import transforms
from Dataloaders import custom_transforms as tr
from PIL import Image

class RTS_Segmentation(Dataset):
    """GaoFen-7四波段滑坡分割数据集."""

    NUM_CLASSES = 2

    def __init__(self, args, base_dir=Path.db_root_dir('rrhtdata'), split='train'):
        super().__init__()
        self._base_dir   = base_dir
        self._image_dir  = os.path.join(self._base_dir, 'IMG_4937_TIF')
        self._cat_dir    = os.path.join(self._base_dir, 'GF7_GT119_2')
        self.split       = [split] if isinstance(split, str) else sorted(split)
        self.args        = args

        # 读取 split 列表
        splits_dir = os.path.join(self._base_dir, 'ImageSets', 'Segmentation')
        self.im_ids, self.images, self.categories = [], [], []

        for splt in self.split:
            list_path = os.path.join(splits_dir, splt + '.txt')
            with open(list_path, 'r') as f:
                lines = f.read().splitlines()
            for line in lines:
                img_path = os.path.join(self._image_dir, line + '.tif')
                cat_path = os.path.join(self._cat_dir,   line + '.png')
                assert os.path.isfile(img_path), f"图像文件不存在: {img_path}"
                assert os.path.isfile(cat_path), f"标签文件不存在: {cat_path}"
                self.im_ids.append(line)
                self.images.append(img_path)
                self.categories.append(cat_path)

        assert len(self.images) == len(self.categories)
        print(f'Number of images in {split}: {len(self.images)}')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # 读取原始影像和标签
        image, label = self._make_img_gt_point_pair(index)
        sample = {'image': image, 'label': label}

        # 根据 split 调用不同的 transform
        if 'train' in self.split:
            return self.transform_tr(sample)
        else:
            return self.transform_val(sample)

    def _make_img_gt_point_pair(self, index):
        # 读取四波段 TIFF (C, H, W) -> 转为 (H, W, C)
        with rasterio.open(self.images[index]) as src:
            img_array = src.read().astype(np.float32)
        img_array = np.transpose(img_array, (1, 2, 0))

        # 读取单通道标签，留作 PIL.Image 供 custom_transforms 处理
        label = Image.open(self.categories[index])
        return img_array, label

    def transform_tr(self, sample):
        """训练时的数据增强：随机翻转、缩放裁剪、高斯模糊 + 归一化 + ToTensor"""
        composed = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size,
                               crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(1506, 2009, 2092, 3103),
                         std=(361,  447,  450,  587)),
            tr.ToTensor()
        ])
        return composed(sample)

    def transform_val(self, sample):
        """验证/测试时的预处理：中心裁剪 + 归一化 + ToTensor"""
        composed = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(1506, 2009, 2092, 3103),
                         std=(361,  447,  450,  587)),
            tr.ToTensor()
        ])
        return composed(sample)
