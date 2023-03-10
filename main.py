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

import matplotlib.pyplot as plt

import segmentation_dataset
import aug
import loader

index = 3

train_Xi_trans = loader.train_dataset.__getitem__(index)[0]
train_yi_trans = loader.train_dataset.__getitem__(index)[1]
trans_toimage = transforms.ToPILImage()
train_Xi_trans = trans_toimage(train_Xi_trans)
train_yi_trans = trans_toimage(train_yi_trans)

train_Xi_trans.show()
train_yi_trans.show()