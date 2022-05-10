# -*- coding: utf-8 -*-
# @Author : youngx
# @Time : 9:39  2022-05-07
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
from torchvision.models import mobilenetv3
from torchvision.models._utils import IntermediateLayerGetter


class RotNetRegression(nn.Module):
    def __init__(self, imagesize=None):
        super(RotNetRegression, self).__init__()
        backbone = mobilenetv3.MobileNetV3(pretrained=True)
        kernelSize = imagesize // 32
        self.back = IntermediateLayerGetter(backbone, {"layer4": 0})

        self.conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Dropout(0.2),
            nn.Conv2d(256, 1, kernel_size=kernelSize, stride=kernelSize, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.back(x)
        out = self.conv(x[0])

        BS = out.size(0)
        out = out.view(BS, 1)
        return out


class RotNet(nn.Module):
    def __init__(self, imagesize=None, angleNum=360):
        super(RotNet, self).__init__()
        backbone = mobilenetv3.mobilenet_v3_small()
        kernelSize = imagesize // 32
        self.angleNum = angleNum
        self.back = IntermediateLayerGetter(backbone, {"features": 0})
        self.conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(576, self.angleNum, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.Softmax(),
        )

    def forward(self, x):
        x = self.back(x)
        out = self.conv(x[0])

        BS = out.size(0)
        out = out.view(BS, self.angleNum)
        return out


if __name__ == '__main__':
    net = RotNetRegression(imagesize=224)
    inputa = torch.randn((1, 3, 224, 224))
    out = net(inputa)
    print(out.shape)
