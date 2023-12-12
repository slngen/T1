'''
Author: CT
Date: 2023-12-04 23:56
LastEditors: CT
LastEditTime: 2023-12-07 19:48
'''
import torch.nn.functional as F
import torch.nn as nn

from Config import config

class CGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(CGBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out

class Backbone(nn.Module):
    def __init__(self, num_classes=2):
        super(Backbone, self).__init__()
        self.initial_layer = CGBlock(6, 32, stride=2)
        self.context_module = CGBlock(32, 64, stride=2)
        self.multi_scale_module = CGBlock(64, 128, stride=2)
        self.final_layer = nn.Conv2d(128, num_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.initial_layer(x)
        x = self.context_module(x)
        x = self.multi_scale_module(x)
        x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=False)
        x = self.final_layer(x)
        x = self.softmax(x)
        return x

if __name__ == "__main__":
    from torchinfo import summary
    
    model = Backbone()
    summary(model, input_size=(config.batch_size, config.input_dim, config.image_size, config.image_size))
    # print(model)