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

from torchsummary import summary

import segmentation_dataset
import aug
from models import unet
import loader

index = 3

train_Xi_trans = loader.train_dataset.__getitem__(index)[0]
train_yi_trans = loader.train_dataset.__getitem__(index)[1]
trans_toimage = transforms.ToPILImage()
train_Xi_trans = trans_toimage(train_Xi_trans)
train_yi_trans = trans_toimage(train_yi_trans)

train_Xi_trans.show()
train_yi_trans.show()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = unet.Model().to(device)

print(summary(model, (3, 572, 572)))