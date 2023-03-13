import torch
from torch import nn

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init()

    def forward(self, x, y, smooth=1):
        x = x.view(-1)
        y = y.view(-1)

        intersection = x * y