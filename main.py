import os
import random
import PIL

import torch
from torch.utils import data
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch import optim

import torchvision
import torchvision.transforms as transforms

# Segmentation을 위한 dataset 클래스 정의
class SegmentationDataset(data.Dataset):
    def __init__(self, inputs:list, targets:list, transform_input=None, transform_target=None):
        self.inputs = inputs
        self.targets = targets
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index:int):
        X = PIL.Image.open('data/train/X/' + self.inputs[index])
        y = PIL.Image.open('data/train/y/' + self.targets[index])

        if self.transform_input is not None:
            torch.manual_seed(42)
            X = self.transform_input(X)

        if self.transform_target is not None:
            torch.manual_seed(42)
            y = self.transform_target(y)

        return X, y

train_X_list = os.listdir('data/train/X')
train_y_list = os.listdir('data/train/y')

train_X_list.sort()
train_y_list.sort()

# training data 증강에 사용할 transform 정의
trans_input = transforms.Compose([
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(180),
    transforms.RandomResizedCrop(256, scale=(0.75, 1.25)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.GaussianBlur(kernel_size=5),
    transforms.ToTensor()  # 이미지를 텐서로 나타냄
])
trans_target = transforms.Compose([
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(180),
    transforms.RandomResizedCrop(256, scale=(0.75, 1.25)),
    transforms.ToTensor()  # 이미지를 텐서로 나타냄
])

train_dataset = SegmentationDataset(
    inputs=train_X_list,
    targets=train_y_list,
    transform_input=trans_input,
    transform_target=trans_target,
)

_batch_size = 4
learning_rate = 0.001
n_epoch = 30

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=_batch_size,
    shuffle=True
)