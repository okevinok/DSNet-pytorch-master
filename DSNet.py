import torch.nn as nn
import torch
from torchvision import models
from utils import save_net, load_net
from torchvision.models.resnet import ResNet, resnet18
from torchsummary import summary


class DSNet(nn.Module):
    def __init__(self, load_weights=False):
        super(DSNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat = [128, 64]
        self.frontend = make_layers(self.frontend_feat)
        self.middlend =
        self.backend = make_layers(self.backend_feat, in_channeals=512, dilation=False)

        # self.DDCB = make_layers()

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if not load_weights:
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            for i in range(len(self.frontend.state_dict().items())):
                self.frontend.state_dict().items()[i][1].data[:] = mod.state_dict().items()[i][1].data[:]

    def forward(self, x):
        x = self.frontend(x)

        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for i, v in cfg:
        if i==len(cfg)-1:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=1, padding=d_rate, dilation=d_rate)
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

# TODO 没有将 创建好的DDCB进行组合构成DSNet
class DDCB(nn.Module):
    def __init__(self,):
        super(DDCB, self).__init__()
        self.base_block1 = DSNetBasicBlock(in_channels=3,  dilation_num=1).to("cuda")

        self.base_block2 = DSNetBasicBlock(in_channels=64, dilation_num=2).to("cuda")
        self.out2_res = self.base_block2.make_residual(in_plain=3)

        self.base_block3 = DSNetBasicBlock(in_channels=64, dilation_num=3).to("cuda")
        self.out3_res = self.base_block2.make_residual(in_plain=3)

        self.conv_3x3_512 = nn.Conv2d(in_channels=64, out_channels=512, kernel_size=3,dilation=1,padding=1)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        out1 = self.base_block1(input)
        out1 = self.relu(out1)

        out2 = self.base_block2(out1)
        # 添加额外的残差
        input_part = self.out2_res(input)
        out2 += input_part
        out2 = self.relu(out2)

        out3 = self.base_block3(out2)
        # 添加额外的残差
        input_part = self.out3_res(input)
        out3 += input_part
        out3 = self.relu(out3)

        final = self.conv_3x3_512(out3)
        final = self.relu(final)

        return final

class DSNetBasicBlock(nn.Module):
    def __init__(self, in_channels=3, dilation_num=1,residual_part=None):
        super(DSNetBasicBlock, self).__init__()
        self.in_channel = in_channels
        self.conv_1x1_D_1 = nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=1, dilation=1,padding=0)
        self.conv_3x3_D_num = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3,dilation=dilation_num, padding=dilation_num,stride=1)
        self.bn1 = nn.BatchNorm2d(num_features=256)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(64)
        self.mid = 0
        self.out = 0
        self.residual_part = residual_part
        self.residual_part_layer= self.make_residual(in_plain=self.in_channel)

    def make_residual(self,in_plain=3):
        return  nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=in_plain, out_channels=64,padding=0,dilation=1,stride=1),
            nn.BatchNorm2d(num_features=64),
            )

    def forward(self, input):
        identity = input
        out = self.conv_1x1_D_1(input)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv_3x3_D_num(out)
        self.out = self.bn2(out)

        # 自带的残差边
        identity = self.residual_part_layer(input)

        self.final = self.out + identity

        return self.final

if __name__ == '__main__':
    # model = DSNetBasicBlock().to("cuda")
    # summary(model,input_size=(3,512,512))


    # model_resnet = resnet18().to("cuda")
    # summary(model_resnet,input_size=(3,1024,1024))


    model = DDCB().to("cuda")
    summary(model,input_size=(3, 256, 512))
    # print(model.eval())








