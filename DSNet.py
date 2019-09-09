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
        self.backend = make_layers(self.backend_feat, in_channels=512,dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self._initialize_weights()

        # 加载权重
        if not load_weights:
            pre_train_dict = models.vgg16(pretrained = True).state_dict()
            model_dict = self.frontend.state_dict()
            model_back_dict = self.backend.state_dict()
            pre_train_dict = {k: v for k, v in pre_train_dict.items() if k in model_dict}
            pre_train_back_dict = {k: v for k, v in pre_train_dict.items() if k in model_back_dict}
            model_dict.update(pre_train_dict)
            model_back_dict.update(pre_train_back_dict)
            self.frontend.load_state_dict(model_dict)
            self.backend.load_state_dict(model_back_dict)

        # 固定某些参数
        for param in self.frontend.parameters():
            param.requires_grad = False
        for param in self.backend.parameters():
            param.requires_grad = False

        """
        创建DDCB残差网络， 训练的时候只是训练DDCB部分 和最后一层
        """
        self.multi_DDCB = multi_DDCB_residual(in_plain=512)

    def forward(self, x):
        x = self.frontend(x)
        x = self.multi_DDCB(x)
        x = self.backend(x)
        y = self.output_layer(x)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def compute_loss(self, nn_outputs, y_true):
        criterion = nn.MSELoss(size_average=False).cuda()
        # loss = K.sqrt(K.sum(K.square(nn_outputs - y_true), axis=-1))
        loss = criterion(nn_outputs, y_true)
        return loss


def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class multi_DDCB_residual(nn.Module):
    def __init__(self, in_plain=512):
        super(multi_DDCB_residual, self).__init__()
        "尝试进行权重共享"
        self.in_plain = in_plain
        self.DDCB1 = DDCB(in_plain=512)
        self.DDCB2 = DDCB(in_plain=512)
        self.DDCB3 = DDCB(in_plain=512)
        self.residual = self.make_residual()
        self.relu = nn.ReLU(inplace=True)


    def make_residual(self):
        return nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=self.in_plain, out_channels=512, padding=0, dilation=1, stride=1),
            nn.BatchNorm2d(num_features=512),
        )

    def forward(self, input):
        out1 = self.DDCB1(input)
        out1 = self.relu(out1)
        res01 = self.residual(input)
        final1 = out1 + res01

        out2 = self.DDCB2(final1)
        out2 = self.relu(out2)
        res12 = self.residual(out1)
        res02 = self.residual(input)
        final2 = out2 + res12 + res02

        out3 = self.DDCB3(final2)
        out3 = self.relu(out3)
        res03 = self.residual(input)
        res13 = self.residual(out1)
        res23 = self.residual(out2)
        final3 = out3 + res03 + res13 +res23

        return final3



    # TODO 没有将 创建好的DDCB进行组合构成DSNet
class DDCB(nn.Module):
    def __init__(self,in_plain=3):
        super(DDCB, self).__init__()
        self.base_block1 = DSNetBasicBlock(in_channels=in_plain,  dilation_num=1).to("cuda")

        self.base_block2 = DSNetBasicBlock(in_channels=64, dilation_num=2).to("cuda")
        self.out2_res = self.make_residual(in_plain=in_plain)

        self.base_block3 = DSNetBasicBlock(in_channels=64, dilation_num=3).to("cuda")
        self.out3_res = self.make_residual(in_plain=in_plain)

        # 最后一层
        self.conv_3x3_512 = nn.Conv2d(in_channels=64, out_channels=512, kernel_size=3,dilation=1,padding=1)
        self.relu = nn.ReLU(inplace=True)

        # 进行参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_residual(self, in_plain):
        return  nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=in_plain, out_channels=64,padding=0,dilation=1,stride=1),
            nn.BatchNorm2d(num_features=64),
            )

    def forward(self, input):
        final1, out1 = self.base_block1(input,forward_out=input)
        out1 = self.relu(out1)

        final2, out2 = self.base_block2(final1, forward_out=out1)
        # 添加额外的残差
        input_part = self.out2_res(input)
        final2 = input_part + final2
        out2 = self.relu(out2)

        final3, out3 = self.base_block3(final2, forward_out=out2)
        # 添加额外的残差
        input_part = self.out3_res(input)
        final3 = input_part + final3
        out3 = self.relu(out3)

        final = self.conv_3x3_512(final3)

        return final


class DSNetBasicBlock(nn.Module):
    def __init__(self, in_channels=3, dilation_num=1):
        super(DSNetBasicBlock, self).__init__()
        self.in_channel = in_channels
        self.conv_1x1_D_1 = nn.Conv2d(in_channels=in_channels, out_channels=256, kernel_size=1, dilation=1,padding=0)
        self.conv_3x3_D_num = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3,dilation=dilation_num, padding=dilation_num,stride=1)
        self.bn1 = nn.BatchNorm2d(num_features=256)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(64)
        self.residual_part_layer= self.make_residual()

    def make_residual(self):
        in_plain = self.in_channel
        return  nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=in_plain, out_channels=64,padding=0,dilation=1,stride=1),
            nn.BatchNorm2d(num_features=64)
            )

    def forward(self, input, forward_out):

        out = self.conv_1x1_D_1(input)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv_3x3_D_num(out)
        self.out = self.bn2(out)

        # 自带的残差边
        identity = self.residual_part_layer(forward_out)

        self.final = self.out + identity

        return self.final, self.out


if __name__ == '__main__':
    # model = DSNetBasicBlock().to("cuda")
    # summary(model,input_size=(3,512,512))

    # model_resnet = resnet18().to("cuda")
    # summary(model_resnet,input_size=(3,1024,1024))


    
    model = DSNet().to("cuda")
    summary(model,input_size=(3, 1024, 512))
    # print(model.eval())

    # watch - n 1 nvidia - smi