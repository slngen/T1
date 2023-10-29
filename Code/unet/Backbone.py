'''
Author: CT
Date: 2023-10-29 16:06
LastEditors: CT
LastEditTime: 2023-10-29 20:25
'''
import torch
import torch.nn as nn
from torchinfo import summary

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
        for i in range(len(features) - 1):
            self.layers.append(nn.ConvTranspose2d(features[i], features[i + 1], 2, stride=2))
            self.layers.append(DoubleConv(features[i] + features[i + 1], features[i + 1]))
        self.conv = nn.Conv2d(features[-1], out_channels, 1)

    def forward(self, feature_maps):
        x = feature_maps[-1]
        for i in range(0, len(self.layers), 2):
            x = self.layers[i](x)
            x = torch.cat((x, feature_maps[-(i // 2 + 2)]), dim=1)
            x = self.layers[i + 1](x)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, features):
        super(UNet, self).__init__()
        self.encoder = Encoder(in_channels, features)
        self.decoder = Decoder(features, out_channels)

    def forward(self, x):
        feature_maps = self.encoder(x)
        return self.decoder(feature_maps)

class PositionalEmbedding(nn.Module):
    def __init__(self, device):
        super(PositionalEmbedding, self).__init__()
        self.embeddings = nn.ParameterDict()
        self.device = device

    def forward(self, pos, channel, height, width):
        batch_size = pos.shape[0]
        embeddings = []
        for i in range(batch_size):
            key = f"pos{'_'.join(map(str, pos[i].int().tolist()))}_ch{channel}_h{height}_w{width}"
            if key not in self.embeddings:
                self.embeddings[key] = nn.Parameter(torch.randn(1, channel, height, width).to(self.device))
            embeddings.append(self.embeddings[key])
        return torch.cat(embeddings, dim=0)

def position_encoding(feature_map, positional_encoding):
    return torch.einsum('bchw,bcwj->bchj', feature_map, positional_encoding)

# 测试函数
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch, channel, height, width = 8, 32, 64, 64
    input_dim = 6
    
    # features = [64, 128, 256, 512]
    # model = UNet(input_dim, 2, features)
    # summary(model, input_size=(batch, input_dim, height, width))

    embedding = PositionalEmbedding(device)  # 实例化模块
    pos = torch.Tensor([[0,0],[0,1],[1,1],[0,0],[0,1],[1,1],[0,0],[0,1]])  # 设置位置
    pos = embedding(pos, channel, height, width).to(device)  # 获取位置编码
    
    feature_map = torch.rand(batch, channel, height, width).to(device)
    result = position_encoding(feature_map, pos)
    print(result.shape)



