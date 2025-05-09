import torch
import random
import numpy as np
from scipy.ndimage import rotate as nd_rotate, gaussian_filter
from skimage.transform import resize
from PIL import Image

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation on original DN values.
    Args:
        mean (tuple): means for each of the 4 bands.
        std (tuple): standard deviations for each of the 4 bands.
    """
    def __init__(self, mean=(1506, 2009, 2092, 3103),
                         std=(361,  447,  450,  587)):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, sample):
        img = sample['image']         # H×W×4, dtype float32 or float64
        mask = sample['label']        # 已可为 np.ndarray 或 PIL.Image
        # 若 mask 还是 PIL.Image，转为数组
        if isinstance(mask, Image.Image):
            mask = np.array(mask)
        # 直接用原始 DN 值做标准化
        img = (img - self.mean) / self.std
        return {'image': img.astype(np.float32),
                'label': mask.astype(np.float32)}

class ToTensor(object):
    """Convert ndarrays in sample to torch Tensors."""
    def __call__(self, sample):
        img = sample['image']    # H×W×4
        mask = sample['label']   # H×W
        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        mask = torch.from_numpy(mask).long()
        return {'image': img, 'label': mask}

class RandomHorizontalFlip(object):
    """随机左右翻转。"""
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # mask np.ndarray
        if random.random() < 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
        return {'image': img, 'label': mask}

class RandomRotate(object):
    """在[-degree, +degree]范围内随机旋转，双线性/最近邻插值。"""
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # 转为 np.ndarray（若尚未转）
        if isinstance(mask, Image.Image):
            mask = np.array(mask)
        angle = random.uniform(-self.degree, self.degree)
        # reshape=False 保持原尺寸
        img = nd_rotate(img, angle, axes=(0,1), reshape=False, order=1, mode='reflect')
        mask = nd_rotate(mask, angle, reshape=False, order=0, mode='nearest')
        return {'image': img, 'label': mask}

class RandomGaussianBlur(object):
    """以 50% 概率对每个波段做随机 sigma 的高斯模糊。"""
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # mask 不变
        if random.random() < 0.5:
            sigma = random.random()
            blurred = np.zeros_like(img)
            for b in range(img.shape[2]):
                blurred[..., b] = gaussian_filter(img[..., b], sigma=sigma)
            img = blurred
        return {'image': img, 'label': mask}

class RandomScaleCrop(object):
    """随机缩放后裁剪到 crop_size，保持原逻辑的填充和裁剪。"""
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # 转为 np.ndarray（若尚未转）
        if isinstance(mask, Image.Image):
            mask = np.array(mask)

        short_size = random.randint(int(self.base_size * 0.5),
                                    int(self.base_size * 2.0))
        h, w = img.shape[:2]
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)

        img = resize(img, (oh, ow, img.shape[2]),
                     order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
        mask = resize(mask, (oh, ow),
                      order=0, preserve_range=True, anti_aliasing=False).astype(mask.dtype)

        # 填充
        if short_size < self.crop_size:
            pad_h = max(self.crop_size - oh, 0)
            pad_w = max(self.crop_size - ow, 0)
            img = np.pad(img,
                         ((0, pad_h), (0, pad_w), (0,0)),
                         mode='constant', constant_values=self.fill)
            mask = np.pad(mask,
                          ((0, pad_h), (0, pad_w)),
                          mode='constant', constant_values=self.fill)

        # 随机裁剪
        h2, w2 = img.shape[:2]
        x1 = random.randint(0, w2 - self.crop_size)
        y1 = random.randint(0, h2 - self.crop_size)
        img = img[y1:y1 + self.crop_size, x1:x1 + self.crop_size, :]
        mask = mask[y1:y1 + self.crop_size, x1:x1 + self.crop_size]

        return {'image': img, 'label': mask}

class FixScaleCrop(object):
    """中心固定裁剪到 crop_size。"""
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if isinstance(mask, Image.Image):
            mask = np.array(mask)

        h, w = img.shape[:2]
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)

        img = resize(img, (oh, ow, img.shape[2]),
                     order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
        mask = resize(mask, (oh, ow),
                      order=0, preserve_range=True, anti_aliasing=False).astype(mask.dtype)

        x1 = int(round((ow - self.crop_size) / 2.))
        y1 = int(round((oh - self.crop_size) / 2.))
        img = img[y1:y1 + self.crop_size, x1:x1 + self.crop_size, :]
        mask = mask[y1:y1 + self.crop_size, x1:x1 + self.crop_size]

        return {'image': img, 'label': mask}

class FixedResize(object):
    """固定缩放到 size×size。"""
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if isinstance(mask, Image.Image):
            mask = np.array(mask)

        img = resize(img, (self.size, self.size, img.shape[2]),
                     order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
        mask = resize(mask, (self.size, self.size),
                      order=0, preserve_range=True, anti_aliasing=False).astype(mask.dtype)

        return {'image': img, 'label': mask}
