import torch
import torch.nn as nn
import torchsummary
from torch.nn import init
class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet,self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,stride=1,padding=1,bias=False)
        self.conv2=nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,stride=1,padding=1,bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)
    def forward(self,x):
        x=self.conv1(x)
        out_map=self.conv2(x)
        return out_map
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




model = BaseNet().to("cuda")
torchsummary.summary(model, (1, 512, 512))
print('parameters_count:',count_parameters(model))