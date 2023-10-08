'''
Author: CT
Date: 2023-07-17 21:25
LastEditors: CT
LastEditTime: 2023-09-13 10:53
'''
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from Config import config

def Unit(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=out_channels),
        )

def makeLayer(layer_index, in_channels, out_channels):
    layer = nn.ModuleList()
    for _ in range(3**layer_index):
        Unit_3 = Unit(in_channels, out_channels, 3, 1, 1)
        Unit_5 = Unit(in_channels, out_channels, 5, 1, 2)
        Unit_7 = Unit(in_channels, out_channels, 7, 1, 3)
        layer.append(nn.ModuleList(
            [Unit_3, Unit_5, Unit_7]
        ))
    return layer

class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dims = config.backbone_dims
        self.layer_nums = config.layer_nums

        self.layers = nn.ModuleList()

        self.encoder_0 = Unit(config.input_dim, self.dims[0], 3, 1, 1)
        for layer_index in range(self.layer_nums):
            layer = makeLayer(layer_index, self.dims[layer_index], self.dims[layer_index+1])
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
            features.append(feature_temp)
        
        return layer_input, features







# class Backbone(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.dim_0 = config.backbone_dims[0]
#         self.dim_1 = config.backbone_dims[1]
#         self.dim_2 = config.backbone_dims[2]
#         ### layer 0
#         self.encoder_0 = nn.Sequential(
#             nn.Conv2d(in_channels=config.input_dim, out_channels=self.dim_0, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(num_features=self.dim_0),
#             nn.Conv2d(in_channels=self.dim_0, out_channels=self.dim_0, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(),
#             nn.BatchNorm2d(num_features=self.dim_0),
#         )

#         ### layer 1
#         self.encoder_1_3 = nn.Sequential(
#             nn.Conv2d(in_channels=self.dim_0, out_channels=self.dim_1, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(num_features=self.dim_1),
#             nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_1, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(),
#             nn.BatchNorm2d(num_features=self.dim_1),
#         )
#         self.encoder_1_5 = nn.Sequential(
#             nn.Conv2d(in_channels=self.dim_0, out_channels=self.dim_1, kernel_size=5, stride=1, padding=2),
#             nn.ReLU(),
#             nn.BatchNorm2d(num_features=self.dim_1),
#             nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_1, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(),
#             nn.BatchNorm2d(num_features=self.dim_1),
#         )
#         self.encoder_1_7 = nn.Sequential(
#             nn.Conv2d(in_channels=self.dim_0, out_channels=self.dim_1, kernel_size=7, stride=1, padding=3),
#             nn.ReLU(),
#             nn.BatchNorm2d(num_features=self.dim_1),
#             nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_1, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(),
#             nn.BatchNorm2d(num_features=self.dim_1),
#         )
#         ### layer-2
#         self.encoder_2_0_3 = nn.Sequential(
#             nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_2, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(num_features=self.dim_2),
#             nn.Conv2d(in_channels=self.dim_2, out_channels=self.dim_2, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(),
#             nn.BatchNorm2d(num_features=self.dim_2),
#         )
#         self.encoder_2_0_5 = nn.Sequential(
#             nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_2, kernel_size=5, stride=1, padding=2),
#             nn.ReLU(),
#             nn.BatchNorm2d(num_features=self.dim_2),
#             nn.Conv2d(in_channels=self.dim_2, out_channels=self.dim_2, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(),
#             nn.BatchNorm2d(num_features=self.dim_2),
#         )
#         self.encoder_2_0_7 = nn.Sequential(
#             nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_2, kernel_size=7, stride=1, padding=3),
#             nn.ReLU(),
#             nn.BatchNorm2d(num_features=self.dim_2),
#             nn.Conv2d(in_channels=self.dim_2, out_channels=self.dim_2, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(),
#             nn.BatchNorm2d(num_features=self.dim_2),
#         )
#         self.encoder_2_1_3 = nn.Sequential(
#             nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_2, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(num_features=self.dim_2),
#             nn.Conv2d(in_channels=self.dim_2, out_channels=self.dim_2, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(),
#             nn.BatchNorm2d(num_features=self.dim_2),
#         )
#         self.encoder_2_1_5 = nn.Sequential(
#             nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_2, kernel_size=5, stride=1, padding=2),
#             nn.ReLU(),
#             nn.BatchNorm2d(num_features=self.dim_2),
#             nn.Conv2d(in_channels=self.dim_2, out_channels=self.dim_2, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(),
#             nn.BatchNorm2d(num_features=self.dim_2),
#         )
#         self.encoder_2_1_7 = nn.Sequential(
#             nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_2, kernel_size=7, stride=1, padding=3),
#             nn.ReLU(),
#             nn.BatchNorm2d(num_features=self.dim_2),
#             nn.Conv2d(in_channels=self.dim_2, out_channels=self.dim_2, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(),
#             nn.BatchNorm2d(num_features=self.dim_2),
#         )
#         self.encoder_2_2_3 = nn.Sequential(
#             nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_2, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(num_features=self.dim_2),
#             nn.Conv2d(in_channels=self.dim_2, out_channels=self.dim_2, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(),
#             nn.BatchNorm2d(num_features=self.dim_2),
#         )
#         self.encoder_2_2_5 = nn.Sequential(
#             nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_2, kernel_size=5, stride=1, padding=2),
#             nn.ReLU(),
#             nn.BatchNorm2d(num_features=self.dim_2),
#             nn.Conv2d(in_channels=self.dim_2, out_channels=self.dim_2, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(),
#             nn.BatchNorm2d(num_features=self.dim_2),
#         )
#         self.encoder_2_2_7 = nn.Sequential(
#             nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_2, kernel_size=7, stride=1, padding=3),
#             nn.ReLU(),
#             nn.BatchNorm2d(num_features=self.dim_2),
#             nn.Conv2d(in_channels=self.dim_2, out_channels=self.dim_2, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(),
#             nn.BatchNorm2d(num_features=self.dim_2),
#         )

#         '''
#         Decoder
#         '''
#         # layer-~2
#         self.decoder_2_0_3 = nn.Sequential(
#             nn.Conv2d(in_channels=self.dim_2, out_channels=self.dim_1, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(num_features=self.dim_1),
#             nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_1, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(),
#             nn.BatchNorm2d(num_features=self.dim_1),
#         )
#         self.decoder_2_1_3 = nn.Sequential(
#             nn.Conv2d(in_channels=self.dim_2, out_channels=self.dim_1, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(num_features=self.dim_1),
#             nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_1, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(),
#             nn.BatchNorm2d(num_features=self.dim_1),
#         )
#         self.decoder_2_2_3 = nn.Sequential(
#             nn.Conv2d(in_channels=self.dim_2, out_channels=self.dim_1, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.BatchNorm2d(num_features=self.dim_1),
#             nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_1, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(),
#             nn.BatchNorm2d(num_features=self.dim_1),
#         )

#         # layer-~1
#         self.decoder_1_0_1 = nn.Sequential(
#             nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_0, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(),
#             nn.BatchNorm2d(num_features=self.dim_0),
#             nn.Conv2d(in_channels=self.dim_0, out_channels=self.dim_0, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(),
#             nn.BatchNorm2d(num_features=self.dim_0),
#         )
#         self.decoder_1_1_1 = nn.Sequential(
#             nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_0, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(),
#             nn.BatchNorm2d(num_features=self.dim_0),
#             nn.Conv2d(in_channels=self.dim_0, out_channels=self.dim_0, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(),
#             nn.BatchNorm2d(num_features=self.dim_0),
#         )
#         self.decoder_1_2_1 = nn.Sequential(
#             nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_0, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(),
#             nn.BatchNorm2d(num_features=self.dim_0),
#             nn.Conv2d(in_channels=self.dim_0, out_channels=self.dim_0, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(),
#             nn.BatchNorm2d(num_features=self.dim_0),
#         )

#         # layer-~0
#         self.decoder_0_0_1 = nn.Sequential(
#             nn.Conv2d(in_channels=self.dim_0, out_channels=2, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(),
#             nn.BatchNorm2d(num_features=2),
#             nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(),
#             nn.BatchNorm2d(num_features=2),
#         )
#         self.decoder_0_1_1 = nn.Sequential(
#             nn.Conv2d(in_channels=self.dim_0, out_channels=2, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(),
#             nn.BatchNorm2d(num_features=2),
#             nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(),
#             nn.BatchNorm2d(num_features=2),
#         )
#         self.decoder_0_2_1 = nn.Sequential(
#             nn.Conv2d(in_channels=self.dim_0, out_channels=2, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(),
#             nn.BatchNorm2d(num_features=2),
#             nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(),
#             nn.BatchNorm2d(num_features=2),
#         )

#         self.softmax = nn.Softmax(dim=1)
    
#     def forward(self, x):
#         '''
#         Encoder
#         '''
#         f_0 = self.encoder_0(x)

#         f_1_3 = self.encoder_1_3(f_0)
#         f_1_5 = self.encoder_1_5(f_0)
#         f_1_7 = self.encoder_1_7(f_0)

#         f_2_0_3 = self.encoder_2_0_3(f_1_3)
#         f_2_0_5 = self.encoder_2_0_5(f_1_5)
#         f_2_0_7 = self.encoder_2_0_7(f_1_7)
#         f_2_1_3 = self.encoder_2_1_3(f_1_3)
#         f_2_1_5 = self.encoder_2_1_5(f_1_5)
#         f_2_1_7 = self.encoder_2_1_7(f_1_7)
#         f_2_2_3 = self.encoder_2_2_3(f_1_3)
#         f_2_2_5 = self.encoder_2_2_5(f_1_5)
#         f_2_2_7 = self.encoder_2_2_7(f_1_7)

#         '''
#         Decoder
#         '''
#         ## layer-~2
#         d_2_0_3 = self.decoder_2_0_3(f_2_0_3+f_2_0_5+f_2_0_7)
#         d_2_1_3 = self.decoder_2_1_3(f_2_1_3+f_2_1_5+f_2_1_7)
#         d_2_2_3 = self.decoder_2_2_3(f_2_2_3+f_2_2_5+f_2_2_7)

#         ## layer-~1
#         d_1_0_1 = self.decoder_1_0_1(d_2_0_3+f_1_3+f_1_5+f_1_7)
#         d_1_1_1 = self.decoder_1_1_1(d_2_1_3+f_1_3+f_1_5+f_1_7)
#         d_1_2_1 = self.decoder_1_2_1(d_2_2_3+f_1_3+f_1_5+f_1_5)

#         ## layer-~0
#         out_0_0_1 = self.decoder_0_0_1(d_1_0_1+f_0)
#         out_0_1_1 = self.decoder_0_1_1(d_1_1_1+f_0)
#         out_0_2_1 = self.decoder_0_2_1(d_1_2_1+f_0)

#         out_0_1 = self.softmax(out_0_0_1)
#         out_1_1 = self.softmax(out_0_1_1)
#         out_2_1 = self.softmax(out_0_2_1)

#         score = out_0_1+out_1_1+out_2_1
#         score = self.softmax(score)
#         return [score]
    
if __name__ == '__main__':
    # model = Backbone()
    model = Encoder()
    print(model)
    summary(model, input_size=(8, config.input_dim, config.image_size, config.image_size))