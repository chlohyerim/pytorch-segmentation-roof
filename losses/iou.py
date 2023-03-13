import torch
from torch import nn
from torch.nn import functional as F

class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, X, y, smooth=1):
        X = F.sigmoid(X)

        # flatten
        X = X.view(-1)
        y = y.view(-1)

        intersection = (X * y).sum()
        total = (X + y).sum()
        union = total - intersection
        iou = (intersection + smooth) / (union + smooth)
        
        return 1. - iou