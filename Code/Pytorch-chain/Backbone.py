'''
Author: CT
Date: 2023-07-17 21:25
LastEditors: CT
LastEditTime: 2023-09-13 10:53
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from Config import config

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=5):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    
class ChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(num_channels, num_channels // reduction_ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(num_channels // reduction_ratio, num_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class AttentionUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(AttentionUnit, self).__init__()
        self.unit = Unit(in_channels, out_channels, kernel_size, stride, padding)
        self.sa = SpatialAttention()  # 初始化空间注意力模块
        self.ca = ChannelAttention(out_channels)  # 初始化通道注意力模块

    def forward(self, x):
        x = self.unit(x)
        x = self.sa(x) * x  # 添加空间注意力
        x = self.ca(x) * x  # 添加通道注意力
        return x

def Unit(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels),
        )

def makeEncoderLayer(layer_index, in_channels, out_channels):
    layer = nn.ModuleList()
    for _ in range(3**layer_index):
        Unit_3 = AttentionUnit(in_channels, out_channels, 3, 1, 1)
        Unit_5 = AttentionUnit(in_channels, out_channels, 5, 1, 2)
        Unit_7 = AttentionUnit(in_channels, out_channels, 7, 1, 3)
        layer.append(nn.ModuleList(
            [Unit_3, Unit_5, Unit_7]
        ))
    return layer

def makeDecoderLayer(layer_index, groups_num, in_channels, out_channels):
    kernel_size = 1
    padding = 0
    if (layer_index==0):
        kernel_size = 3
        padding = 1
    layer = nn.ModuleList([
        AttentionUnit(in_channels, out_channels, kernel_size, 1, padding) for _ in range(groups_num)
    ])
    return layer

class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dims = config.backbone_dims
        self.layer_nums = config.layer_nums

        self.layers = nn.ModuleList()

        self.encoder_0 = Unit(config.input_dim, self.dims[0], 3, 1, 1)
        for layer_index in range(self.layer_nums):
            layer = makeEncoderLayer(layer_index, self.dims[layer_index], self.dims[layer_index+1])
            self.layers.append(layer)

    def forward(self, x):
        f_0 = self.encoder_0(x)

        features = []
        features.append(f_0)

        layer_input = [f_0]
        for layer_index in range(self.layer_nums):
            feature_temp = 0
            layer_out = []
            layer = self.layers[layer_index]
            for group_index in range(len(layer)):
                group_input = layer_input[group_index]
                for unit in layer[group_index]:
                    unit_out = unit(group_input)
                    layer_out.append(unit_out)
                    feature_temp += unit_out
            layer_input = layer_out
            if (layer_index<self.layer_nums-1):
                features.insert(0, feature_temp)
        
        return layer_input, features

class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dims = config.backbone_dims
        self.layer_nums = config.layer_nums
        self.group_nums = 3**(config.layer_nums-1)
        self.layers = nn.ModuleList()

        for layer_index in range(self.layer_nums):
            input_dim = self.dims[self.layer_nums-layer_index]
            output_dim = self.dims[self.layer_nums-layer_index-1]
            self.layers.append(makeDecoderLayer(layer_index, self.group_nums, input_dim, output_dim))

        self.layers.append(makeDecoderLayer(0, self.group_nums, self.dims[0], config.class_nums))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_output, features):
        layer_input = []
        for layer_index in range(len(self.layers)):
            layer = self.layers[layer_index]
            if layer_index == 0:
                for group_index in range(int(len(encoder_output)/3)):
                    group_out = layer[group_index](encoder_output[group_index]+encoder_output[group_index+1]+encoder_output[group_index+2])
                    layer_input.append(group_out)
            else:
                layer_out = []
                for unit_index in range(len(layer_input)):
                    if (layer_index <= len(features)):
                        unit_output = layer[unit_index](layer_input[unit_index]+features[layer_index-1])
                    else:
                        unit_output = layer[unit_index](layer_input[unit_index])
                    layer_out.append(unit_output)
                layer_input = layer_out
        # Vote
        score = 0
        for out in layer_input:
            score += self.softmax(out)
        return score

class Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoder_output, features = self.encoder(x)
        score = self.decoder(encoder_output, features)
        return [score]
    
if __name__ == '__main__':
    model = Backbone()
    print(model)
    summary(model, input_size=(32, config.input_dim, config.image_size, config.image_size))