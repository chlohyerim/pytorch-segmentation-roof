import torch
from torch import nn
from torch.nn import functional as F

import torchvision

# convolutional layer 2번
class DoubleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.sequence = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3),
            nn.ReLU()
        )

    def forward(self, x):
        return self.sequence(x)
        
# maxpooling -> convolutional layer 2번 적용으로 expansive path에서 concatenate할 context를 만듦
class ContractingPath(nn.Module):
    def __init__(self, in_channels, out_channels, max_pooling=True):
        super().__init__()

        self.max_pooling = max_pooling
        self.double_conv2d = DoubleConv2d(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x):
        if self.max_pooling: x = F.max_pool2d(x, kernel_size=2)

        return self.double_conv2d(x)

# upconvolution -> context 가운데 부분 crop해서 concatenate -> convolutional layer 2번 적용
# padding을 하지 않아서 crop 사용(u-net 논문과 최대한 동일하게 구현)    
class ExpansivePath(nn.Module):
    def __init__(self, in_channels, out_channels, context):
        super().__init__()

        self.context = context
        self.upconv2d = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
        self.double_conv2d = DoubleConv2d(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x, context):
        x = self.upconv2d(x)

        context_crop_top = (context.size()[2] - x.size()[2]) // 2
        context_crop_left = (context.size()[3] - x.size()[3]) // 2
        x_height = x.size()[2]
        x_width = x.size()[3]
        context = torchvision.transforms.functional.crop(context, top=context_crop_top, left=context_crop_left, height=x_height, width=x_width)

        x = torch.cat((context, x), dim=1)
        x = self.double_conv2d(x)

        return x

# 마지막으로 1채널로 만드는 1x1 커널 convolutional layer + dropout 적용
class FullyConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.sequence = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Dropout2d(0.5)
        )

    def forward(self, x):
        return self.sequence(x)

# 모델 구성
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.contract1 = ContractingPath(in_channels=3, out_channels=8, max_pooling=False)
        self.contract2 = ContractingPath(in_channels=8, out_channels=16)
        self.contract3 = ContractingPath(in_channels=16, out_channels=32)
        self.contract4 = ContractingPath(in_channels=32, out_channels=64)
        self.contract5 = ContractingPath(in_channels=64, out_channels=128)
        
        self.expand4 = ExpansivePath(in_channels=128, out_channels=64, context=None)     
        self.expand3 = ExpansivePath(in_channels=64, out_channels=32, context=None)
        self.expand2 = ExpansivePath(in_channels=32, out_channels=16, context=None)
        self.expand1 = ExpansivePath(in_channels=16, out_channels=8, context=None)

        self.fconv2d = FullyConv2d(in_channels=8, out_channels=1)

    def forward(self, x):
        context1 = self.contract1(x)
        context2 = self.contract2(context1)
        context3 = self.contract3(context2)
        context4 = self.contract4(context3)
        x = self.contract5(context4)

        x = self.expand4(x, context=context4)
        x = self.expand3(x, context=context3)
        x = self.expand2(x, context=context2)
        x = self.expand1(x, context=context1)

        x = self.fconv2d(x)

        return x