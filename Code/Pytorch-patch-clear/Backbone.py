'''
Author: CT
Date: 2023-07-17 21:25
LastEditors: CT
LastEditTime: 2023-08-16 09:13
'''
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from Config import config

class Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dim_0 = config.backbone_dims[0]
        self.dim_1 = config.backbone_dims[1]
        self.dim_2 = config.backbone_dims[2]

        self.encoder_0 = nn.Sequential(
            nn.Conv2d(in_channels=config.input_dim, out_channels=self.dim_0, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_0),
            nn.Conv2d(in_channels=self.dim_0, out_channels=self.dim_0, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_0),
        )
        self.encoder_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_0, out_channels=self.dim_1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_1),
            nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_1)
        )
        self.encoder_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_2),
            nn.Conv2d(in_channels=self.dim_2, out_channels=self.dim_2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_2)
        )

        self.decoder_0 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_0, out_channels=2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=2),
            nn.ReLU()
        )
        self.decoder_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_0, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=self.dim_0),
            nn.ReLU()
        )
        self.decoder_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_2, out_channels=self.dim_1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=self.dim_1),
            nn.ReLU()
        )
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def scalex2(self, x):
        return F.interpolate(x, scale_factor=2, mode='bilinear')
    
    def forward(self, x):
        f_0 = self.encoder_0(x)
        f_0 = self.pool(f_0)
        f_1 = self.encoder_1(f_0)
        f_1 = self.pool(f_1)
        f_2 = self.encoder_2(f_1)
        f_2 = self.pool(f_2)

        out_2 = self.decoder_2(f_2)
        out_2 = self.scalex2(out_2)
        out_1 = self.decoder_1(f_1+out_2)
        out_1 = self.scalex2(out_1)
        out_0 = self.decoder_0(f_0+out_1)
        out_0 = self.scalex2(out_0)
        out_0 = F.interpolate(out_0, size=(config.raw_size, config.raw_size), mode='bilinear', align_corners=False)
        score = self.softmax(out_0)

        return [score]
    
if __name__ == '__main__':
    model = Backbone()
    print(model)
    summary(model, input_size=(config.batch_size, config.input_dim, config.image_size, config.image_size))