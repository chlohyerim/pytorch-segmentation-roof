import torch
from torch import nn
from torch.nn import functional as F

# 모델 구성
class UNet(nn.module):
    def __init__(self):
        super(UNet, self).__init__()
        # conv2d + relu
        def conv2d_relu(in_channels, out_channels, kernel_size=3):
            layers = []
            layers += [nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
            )]
            layers += [nn.ReLU()]

            return nn.Sequential(*layers)

        self.contract1_first = conv2d_relu(in_channels=3, out_channels=8)
        self.contract1_second = conv2d_relu(in_channels=8, out_channels=8)
        self.contract2_first = conv2d_relu(in_channels=8, out_channels=16)
        self.contract2_second = conv2d_relu(in_channels=16, out_channels=16)
        self.contract3_first = conv2d_relu(in_channels=16, out_channels=32)
        self.contract3_second = conv2d_relu(in_channels=32, out_channels=32)
        self.contract4_first = conv2d_relu(in_channels=32, out_channels=64)
        self.contract4_second = conv2d_relu(in_channels=64, out_channels=64)
        self.contract5 = conv2d_relu(in_channels=64, out_channels=128)
        self.expand5 = conv2d_relu(in_channels=128, out_channels=128)
        self.upconv2d4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.expand4 = conv2d_relu(in_channels=128, out_channels=64)
        self.upconv2d3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.expand3 = conv2d_relu(in_channels=64, out_channels=32)
        self.upconv2d2 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2)
        self.expand2 = conv2d_relu(in_channels=32, out_channels=16)
        self.upconv2d1 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=2, stride=2)
        self.expand1 = conv2d_relu(in_channels=16, out_channels=8)
        self.fconv = conv2d_relu(in_channels=8, out_channels=1)

    def forward(self, x):
        # contracting, downsampling
        x = self.contract1_first(x)
        context1 = self.contract1_second(x)
        x = F.max_pool2d(context1)

        x = self.contract2_first(x)
        context2 = self.contract2_second(x)
        x = F.max_pool2d(context2)

        x = self.contract3_first(x)
        context3 = self.contract3_second(x)
        x = F.max_pool2d(context3)

        x = self.contract4_first(x)
        context4 = self.contract4_second(x)
        x = F.max_pool2d(context4)

        x = self.contract5(x)
        
        # expanding, upsampling
        x = self.expand5(x)

        x = self.upconv2d4(x)
        x = torch.cat((context4, x), dim=1)
        x = self.expand4(x)
        x = self.contract4_second(x)

        x = self.upconv2d3(x)
        x = torch.cat((context3, x), dim=1)
        x = self.expand3(x)
        x = self.contract3_second(x)

        x = self.upconv2d2(x)
        x = torch.cat((context2, x), dim=1)
        x = self.expand2(x)
        x = self.contract2_second(x)

        x = self.upconv2d1(x)
        x = torch.cat((context1, x), dim=1)
        x = self.expand1(x)
        x = self.contract1_second(x)

        x = self.fconv(x)

        return x