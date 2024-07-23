
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2
import math

class MyDataset(Dataset):
    def __init__(self, root_dir, scale_factor, crop_size, transform=None):
        self.root_dir = root_dir
        self.scale_factor = scale_factor
        self.transform = transform
        self.image_files = self._load_image_files()
        self.crop_size = (crop_size, crop_size)  # 统一尺寸 (height, width)

    def _load_image_files(self):
        # 读取所有高分辨率图像文件
        return [f for f in os.listdir(self.root_dir) if f.endswith('.png') or f.endswith('.jpg')]

    def _generate_lr_image(self, hr_image):
        # 获取高分辨率图像尺寸
        hr_width, hr_height = hr_image.size
        lr_width, lr_height = hr_width // self.scale_factor, hr_height // self.scale_factor
        
        # 使用双三次插值生成低分辨率图像
        hr_image_np = np.array(hr_image)
        lr_image_np = cv2.resize(hr_image_np, (lr_width, lr_height), interpolation=cv2.INTER_LANCZOS4)
        lr_image = Image.fromarray(lr_image_np)
        return lr_image

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        hr_image = Image.open(img_name).convert('RGB')
        hr_image = transforms.RandomCrop(self.crop_size)(hr_image)
        hr_image = transforms.RandomHorizontalFlip(p=0.5)(hr_image)
        hr_image = transforms.RandomVerticalFlip(p=0.5)(hr_image)
        lr_image = self._generate_lr_image(hr_image)

        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)

        return lr_image, hr_image

# 数据转换和标准化
transform = transforms.Compose([
    transforms.ToTensor(),
])
