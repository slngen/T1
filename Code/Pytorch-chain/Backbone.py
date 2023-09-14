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

class Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dim_0 = config.backbone_dims[0]
        self.dim_1 = config.backbone_dims[1]
        self.dim_2 = config.backbone_dims[2]
        ### layer 0
        self.encoder_0 = nn.Sequential(
            nn.Conv2d(in_channels=config.input_dim, out_channels=self.dim_0, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_0),
            nn.Conv2d(in_channels=self.dim_0, out_channels=self.dim_0, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_0),
        )

        ### layer 1
        self.encoder_1_3 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_0, out_channels=self.dim_1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_1),
            nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_1),
        )
        self.encoder_1_5 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_0, out_channels=self.dim_1, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_1),
            nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_1),
        )
        self.encoder_1_7 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_0, out_channels=self.dim_1, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_1),
            nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_1),
        )
        ### layer-2
        self.encoder_2_0_3 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_2),
            nn.Conv2d(in_channels=self.dim_2, out_channels=self.dim_2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_2),
        )
        self.encoder_2_0_5 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_2, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_2),
            nn.Conv2d(in_channels=self.dim_2, out_channels=self.dim_2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_2),
        )
        self.encoder_2_0_7 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_2, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_2),
            nn.Conv2d(in_channels=self.dim_2, out_channels=self.dim_2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_2),
        )

        self.encoder_2_1_3 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_2),
            nn.Conv2d(in_channels=self.dim_2, out_channels=self.dim_2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_2),
        )
        self.encoder_2_1_5 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_2, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_2),
            nn.Conv2d(in_channels=self.dim_2, out_channels=self.dim_2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_2),
        )
        self.encoder_2_1_7 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_2, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_2),
            nn.Conv2d(in_channels=self.dim_2, out_channels=self.dim_2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_2),
        )

        self.encoder_2_2_3 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_2),
            nn.Conv2d(in_channels=self.dim_2, out_channels=self.dim_2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_2),
        )
        self.encoder_2_2_5 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_2, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_2),
            nn.Conv2d(in_channels=self.dim_2, out_channels=self.dim_2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_2),
        )
        self.encoder_2_2_7 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_2, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_2),
            nn.Conv2d(in_channels=self.dim_2, out_channels=self.dim_2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_2),
        )

        '''
        Decoder
        '''
        # layer-~2
        self.decoder_2_0_3 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_2, out_channels=self.dim_1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_1),
            nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_1),
        )
        self.decoder_2_0_5 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_2, out_channels=self.dim_1, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_1),
            nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_1),
        )
        self.decoder_2_0_7 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_2, out_channels=self.dim_1, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_1),
            nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_1),
        )

        self.decoder_2_1_3 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_2, out_channels=self.dim_1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_1),
            nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_1),
        )
        self.decoder_2_1_5 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_2, out_channels=self.dim_1, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_1),
            nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_1),
        )
        self.decoder_2_1_7 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_2, out_channels=self.dim_1, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_1),
            nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_1),
        )

        self.decoder_2_2_3 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_2, out_channels=self.dim_1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_1),
            nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_1),
        )
        self.decoder_2_2_5 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_2, out_channels=self.dim_1, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_1),
            nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_1),
        )
        self.decoder_2_2_7 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_2, out_channels=self.dim_1, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_1),
            nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_1),
        )

        # layer-~1
        self.decoder_1_0_3 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_0, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_0),
            nn.Conv2d(in_channels=self.dim_0, out_channels=self.dim_0, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_0),
        )
        self.decoder_1_0_5 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_0, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_0),
            nn.Conv2d(in_channels=self.dim_0, out_channels=self.dim_0, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_0),
        )
        self.decoder_1_0_7 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_0, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_0),
            nn.Conv2d(in_channels=self.dim_0, out_channels=self.dim_0, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_0),
        )

        self.decoder_1_1_3 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_0, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_0),
            nn.Conv2d(in_channels=self.dim_0, out_channels=self.dim_0, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_0),
        )
        self.decoder_1_1_5 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_0, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_0),
            nn.Conv2d(in_channels=self.dim_0, out_channels=self.dim_0, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_0),
        )
        self.decoder_1_1_7 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_0, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_0),
            nn.Conv2d(in_channels=self.dim_0, out_channels=self.dim_0, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_0),
        )

        self.decoder_1_2_3 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_0, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_0),
            nn.Conv2d(in_channels=self.dim_0, out_channels=self.dim_0, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_0),
        )
        self.decoder_1_2_5 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_0, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_0),
            nn.Conv2d(in_channels=self.dim_0, out_channels=self.dim_0, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_0),
        )
        self.decoder_1_2_7 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_1, out_channels=self.dim_0, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_0),
            nn.Conv2d(in_channels=self.dim_0, out_channels=self.dim_0, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=self.dim_0),
        )

        # layer-~0
        self.decoder_0_0_3 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_0, out_channels=2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=2),
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=2),
        )
        self.decoder_0_0_5 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_0, out_channels=2, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=2),
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=2),
        )
        self.decoder_0_0_7 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_0, out_channels=2, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=2),
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=2),
        )

        self.decoder_0_1_3 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_0, out_channels=2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=2),
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=2),
        )
        self.decoder_0_1_5 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_0, out_channels=2, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=2),
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=2),
        )
        self.decoder_0_1_7 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_0, out_channels=2, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=2),
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=2),
        )

        self.decoder_0_2_3 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_0, out_channels=2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=2),
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=2),
        )
        self.decoder_0_2_5 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_0, out_channels=2, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=2),
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=2),
        )
        self.decoder_0_2_7 = nn.Sequential(
            nn.Conv2d(in_channels=self.dim_0, out_channels=2, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=2),
            nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=2),
        )

        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        '''
        Encoder
        '''
        f_0 = self.encoder_0(x)

        f_1_3 = self.encoder_1_3(f_0)
        f_1_5 = self.encoder_1_5(f_0)
        f_1_7 = self.encoder_1_7(f_0)

        f_2_0_3 = self.encoder_2_0_3(f_1_3)
        f_2_0_5 = self.encoder_2_0_5(f_1_5)
        f_2_0_7 = self.encoder_2_0_7(f_1_7)
        f_2_1_3 = self.encoder_2_1_3(f_1_3)
        f_2_1_5 = self.encoder_2_1_5(f_1_5)
        f_2_1_7 = self.encoder_2_1_7(f_1_7)
        f_2_2_3 = self.encoder_2_2_3(f_1_3)
        f_2_2_5 = self.encoder_2_2_5(f_1_5)
        f_2_2_7 = self.encoder_2_2_7(f_1_7)

        '''
        Decoder
        '''
        ## layer-~2
        d_2_0_3 = self.decoder_2_0_3(f_2_0_3)
        d_2_0_5 = self.decoder_2_0_5(f_2_0_5)
        d_2_0_7 = self.decoder_2_0_7(f_2_0_7)
        d_2_1_3 = self.decoder_2_0_3(f_2_1_3)
        d_2_1_5 = self.decoder_2_0_5(f_2_1_5)
        d_2_1_7 = self.decoder_2_0_7(f_2_1_7)
        d_2_2_3 = self.decoder_2_0_3(f_2_2_3)
        d_2_2_5 = self.decoder_2_0_5(f_2_2_5)
        d_2_2_7 = self.decoder_2_0_7(f_2_2_7)

        ## layer-~1
        d_1_0_3 = self.decoder_1_0_3(d_2_0_3+f_1_3)
        d_1_0_5 = self.decoder_1_0_5(d_2_0_5+f_1_5)
        d_1_0_7 = self.decoder_1_0_7(d_2_0_7+f_1_7)
        d_1_1_3 = self.decoder_1_0_3(d_2_1_3+f_1_3)
        d_1_1_5 = self.decoder_1_0_5(d_2_1_5+f_1_5)
        d_1_1_7 = self.decoder_1_0_7(d_2_1_7+f_1_7)
        d_1_2_3 = self.decoder_1_0_3(d_2_2_3+f_1_3)
        d_1_2_5 = self.decoder_1_0_5(d_2_2_5+f_1_5)
        d_1_2_7 = self.decoder_1_0_7(d_2_2_7+f_1_7)

        ## layer-~0
        out_0_3 = self.decoder_0_0_3(d_1_0_3+f_0)
        out_0_5 = self.decoder_0_0_5(d_1_0_5+f_0)
        out_0_7 = self.decoder_0_0_7(d_1_0_7+f_0)
        out_1_3 = self.decoder_0_0_3(d_1_1_3+f_0)
        out_1_5 = self.decoder_0_0_5(d_1_1_5+f_0)
        out_1_7 = self.decoder_0_0_7(d_1_1_7+f_0)
        out_2_3 = self.decoder_0_0_3(d_1_2_3+f_0)
        out_2_5 = self.decoder_0_0_5(d_1_2_5+f_0)
        out_2_7 = self.decoder_0_0_7(d_1_2_7+f_0)

        out_0_3 = self.softmax(out_0_3)
        out_0_5 = self.softmax(out_0_5)
        out_0_7 = self.softmax(out_0_7)
        out_1_3 = self.softmax(out_1_3)
        out_1_5 = self.softmax(out_1_5)
        out_1_7 = self.softmax(out_1_7)
        out_2_3 = self.softmax(out_2_3)
        out_2_5 = self.softmax(out_2_5)
        out_2_7 = self.softmax(out_2_7)

        score = out_0_3+out_0_5+out_0_7+out_1_3+out_1_5+out_1_7+out_2_3+out_2_5+out_2_7
        score = self.softmax(score)
        return [score]
    
if __name__ == '__main__':
    model = Backbone()
    # print(model)
    summary(model, input_size=(8, config.input_dim, config.image_size, config.image_size))