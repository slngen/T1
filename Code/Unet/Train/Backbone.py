'''
Author: CT
Date: 2023-10-29 16:06
LastEditors: CT
LastEditTime: 2023-12-24 17:09
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from Config import config

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, features):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        for feature in features:
            self.layers.append(DoubleConv(in_channels, feature))
            self.layers.append(nn.MaxPool2d(2))
            in_channels = feature

    def forward(self, x):
        feature_maps = []
        for layer in self.layers:
            if isinstance(layer, nn.MaxPool2d):
                x = layer(x)
            else:
                x = layer(x)
                feature_maps.append(x)
        return feature_maps

class Decoder(nn.Module):
    def __init__(self, features, out_channels):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            self.layers.append(nn.ConvTranspose2d(features[i], features[i - 1], 2, stride=2))
            self.layers.append(DoubleConv(2*features[i - 1], features[i - 1]))
        self.conv = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, feature_maps):
        x = feature_maps[-1]
        for i in range(0, len(self.layers), 2):
            x = self.layers[i](x)
            skip_connection = feature_maps[-(i // 2 + 2)]
            x = torch.cat((x, skip_connection), dim=1)
            x = self.layers[i + 1](x)
        return self.conv(x)

class FuseConv(nn.Module):
    def __init__(self, channels):
        super(FuseConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class FuseModule(nn.Module):
    def __init__(self):
        super(FuseModule, self).__init__()
        self.embeddings = nn.ModuleDict()
        self.device = config.device

        self.fuseDict = nn.ModuleDict()
        directions = ["up", "down", "left", "right"]
        for dir in directions:
            self.fuseDict[dir] = nn.ModuleDict({
                str(channels): FuseConv(channels)
                for channels in config.features
            })
        self.fuseDict.to(self.device)

    def _direction_to_key(self, direction):
        direction_map = {
            tuple([0, -1]): "up",
            tuple([0, 1]): "down",
            tuple([-1, 0]): "left",
            tuple([1, 0]): "right"
        }
        return direction_map[tuple(direction.tolist())]
    
    def _cat_features(self, center_features, guide_features, dir):
        if dir == "up":
            return torch.cat([guide_features, center_features], dim=3)
        elif dir == "down":
            return torch.cat([center_features, guide_features], dim=3)
        elif dir == "left":
            return torch.cat([guide_features, center_features], dim=2)
        elif dir == "right":
            return torch.cat([center_features, guide_features], dim=2)
        
    def _reduce_features(self, feature, dir):
        size = min(feature.shape[2:])
        if dir == "up":
            return feature[:,:,:,size:]
        elif dir == "down":
            return feature[:,:,:,:size]
        elif dir == "left":
            return feature[:,:,size:,:]
        elif dir == "right":
            return feature[:,:,:size,:]

    def forward(self, center_features, guide_features, direction):
        fuse_features = []
        dir = self._direction_to_key(direction)
        for index in range(len(center_features)):
            cat_feature = self._cat_features(center_features[index], guide_features[index], dir)
            cat_fuse_feature = self.fuseDict[dir][str(center_features[index].shape[1])](cat_feature)
            fuse_feature = self._reduce_features(cat_fuse_feature, dir)
            fuse_features.append(fuse_feature)
        return fuse_features

class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.device = config.device
        self.encoder = Encoder(config.input_dim, config.features)
        self.decoder = Decoder(config.features, 2)
        self.fuse = FuseModule()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        center_patch = x[:,0,:,:,:]
        guide_patchs = x[:,1:,:,:,:]

        score = 0
        center_features = self.encoder(center_patch)
        directions = torch.tensor([[0, -1], [0, 1], [-1, 0], [1, 0]])
        for pos_index in range(len(directions)):
            guide_features = self.encoder(guide_patchs[:,pos_index,:,:,:])
            fuse_features = self.fuse(center_features, guide_features, directions[pos_index])
            output = self.decoder(fuse_features)
            output = self.softmax(output)
            score += output
        score = self.softmax(score)
        return score

if __name__ == "__main__":
    from torchinfo import summary
    
    model = Backbone()
    summary(model, input_size=(config.batch_size, 5, config.input_dim, config.image_size, config.image_size))
    # print(model)