'''
Author: CT
Date: 2023-11-26 09:43
LastEditors: CT
LastEditTime: 2023-11-26 09:59
'''
import torch
import torch.nn as nn
import torchvision.models as models

from Config import config

class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channels, pool_sizes):
        super(PyramidPoolingModule, self).__init__()
        self.pool_sizes = pool_sizes
        self.pooling_layers = nn.ModuleList()
        for size in pool_sizes:
            self.pooling_layers.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(in_channels, in_channels // len(pool_sizes), kernel_size=1),
                nn.BatchNorm2d(in_channels // len(pool_sizes)),
                nn.ReLU(inplace=True)
            ))

    def forward(self, x):
        features = [x]
        for pooling_layer in self.pooling_layers:
            pooled = pooling_layer(x)
            upsampled = nn.functional.interpolate(pooled, size=x.shape[2:], mode='bilinear', align_corners=False)
            features.append(upsampled)
        return torch.cat(features, dim=1)

class Backbone(nn.Module):
    def __init__(self, num_classes=2, in_channels=6):
        super(Backbone, self).__init__()
        backbone = models.resnet50(weights=None)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.features[0] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.pyramid_pooling = PyramidPoolingModule(2048, [1, 2, 3, 6])

        self.final_conv = nn.Conv2d(4096, num_classes, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.features(x)
        x = self.pyramid_pooling(x)
        x = self.final_conv(x)
        x = nn.functional.interpolate(x, scale_factor=32, mode='bilinear', align_corners=False)
        return self.softmax(x)
    
if __name__ == "__main__":
    from torchinfo import summary
    
    model = Backbone()
    summary(model, input_size=(config.batch_size, config.input_dim, config.image_size, config.image_size))
    # print(model)