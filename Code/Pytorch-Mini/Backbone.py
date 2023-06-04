'''
Author: CT
Date: 2023-06-03 23:51
LastEditors: CT
LastEditTime: 2023-06-04 14:56
'''
import torch.nn as nn
from torchinfo import summary
from torchvision.transforms import Resize

class BasicBlock(nn.Module):
    def __init__(self, conv_nums, in_channel_nums, out_channel_nums):
        super().__init__()
        self.conv_List = nn.ModuleList([
            nn.Conv2d(in_channel_nums, out_channel_nums, 1, 1) if index==0 else nn.Conv2d(out_channel_nums, out_channel_nums, 1, 1) for index in range(conv_nums)
        ])
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        for conv in self.conv_List:
            x = conv(x)
            x = self.relu(x)
        x = self.pool(x)
        return x
    
class Branch(nn.Module):
    def __init__(self, conv_nums_List, dims_List, shape_dim):
        super().__init__()
        self.embedding = nn.Conv2d(dims_List[0], dims_List[1], 3, 1, 1)
        self.Blocks = nn.ModuleList([
            BasicBlock(conv_nums_List[index], dims_List[index+1], dims_List[index+2]) for index in range(len(conv_nums_List))
        ])
        self.resize = Resize((shape_dim, shape_dim))

    def forward(self, x):
        sync_List = []
        x = self.embedding(x)
        for block in self.Blocks:
            x = block(x)
            sync_List.append(x)
        x = self.resize(x)
        return x, sync_List
    
if __name__ == '__main__':
    model = Branch(conv_nums_List=[2,2,2,2], dims_List=[6,64,128,256,512,2], shape_dim=512)
    print(model)
    summary(model, input_size=(4, 6, 512, 512))