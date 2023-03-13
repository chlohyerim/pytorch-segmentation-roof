import torch
from torch import nn
from torch.nn import functional as F

class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, X, y, smooth=1):
        X = F.sigmoid(X)

        # flatten
        X = X.view(-1)
        y = y.view(-1)

        intersection = (X * y).sum()
        loss = 1. - (2. * intersection + smooth) / (X.sum() + y.sum() + smooth)
        bce = F.binary_cross_entropy(X, y, reduction='mean')
        
        return loss + bce