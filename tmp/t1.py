# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchsummary
from torch.nn import init
class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet,self).__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=256,kernel_size=1,stride=1,padding=1,bias=False,)
        self.conv2=nn.Conv2d(in_channels=256,out_channels=64,kernel_size=3,stride=1,padding=1,bias=False)
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_uniform_(m.weight.data)
        #         if m.bias is not None:
        #             m.bias.data.zero_()
        #     elif isinstance(m, nn.BatchNorm2d):
        #         init.normal_(m.weight.data, 1.0, 0.02)
        #         init.constant_(m.bias.data, 0.0)
        self.bn1 = nn.BatchNorm2d(num_features=256)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(64)

    def make_residual(self, in_plain=3):
        return nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=in_plain, out_channels=64, padding=0, dilation=1, stride=1),
            nn.BatchNorm2d(num_features=64),
        )
    def forward(self,x):
        y=self.conv1(x)
        y = self.bn1(y)
        y = self.relu(y)
        out_map=self.conv2(y)#注意这里换成了conv1
        out_map = self.bn2(out_map)
        # identity = self.residual_part_layer(x)
        # out_map = out_map + identity

        return out_map


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


model = BaseNet().to("cuda")
torchsummary.summary(model, (3, 512,2048))
# print('parameters_count:',count_parameters(model))