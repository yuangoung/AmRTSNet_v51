import os
import numpy as np
from PIL import Image

# Binary_GroundTruth GT
input_folder = '.../Ground Truth/SegmentationClass/PNG256Images'
output_folder = '.../Ground Truth/PNG128Images/'

# output
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# threshold
threshold = 128

# 遍历输入文件夹中的所有PNG图像
for filename in os.listdir(input_folder):
    if filename.endswith('.png'):
        # 打开PNG图像
        img = Image.open(os.path.join(input_folder, filename))

        # 将图像转换为灰度图像
        img_gray = img.convert('L')

        # 转换为NumPy数组
        img_array = np.array(img_gray)

        # 使用阈值将图像二值化
        img_binary = (img_array > threshold).astype(np.uint8)*128

        # 创建新的PIL图像对象
        img_binary = Image.fromarray(img_binary)

        # 保存二值化后的图像到输出文件夹
        img_binary.save(os.path.join(output_folder, filename))