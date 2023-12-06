'''
Author: CT
Date: 2023-12-04 23:56
LastEditors: CT
LastEditTime: 2023-12-06 09:59
'''
import torch.nn.functional as F
from torchvision import models
import torch.nn as nn

from Config import config

class ICNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ICNet, self).__init__()
        resnet = models.resnet50(weights=None)

        self.initial_conv = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.conv_sub1 = nn.Sequential(*list(resnet.children())[1:6])
        self.conv_sub2 = list(resnet.children())[6]
        self.conv_sub4 = list(resnet.children())[7]

        self.fusion_layer = nn.Conv2d(2048, num_classes, kernel_size=1)

    def forward(self, x):
        original_size = x.shape[2:]

        x = self.initial_conv(x)

        sub1 = self.conv_sub1(x)
        sub2 = self.conv_sub2(sub1)
        sub4 = self.conv_sub4(sub2)

        fusion = self.fusion_layer(sub4)

        upsampled = F.interpolate(fusion, size=original_size, mode='bilinear', align_corners=True)

        return upsampled


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.icnet = ICNet()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.icnet(x)
        x = self.softmax(x)
        return x

if __name__ == "__main__":
    from torchinfo import summary
    
    model = Backbone()
    summary(model, input_size=(config.batch_size, config.input_dim, config.image_size, config.image_size))
    # print(model)