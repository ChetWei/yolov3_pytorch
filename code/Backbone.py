import math
from collections import OrderedDict

import torch
import torch.nn as nn


# ---------------------------------------------------------------------#
#   残差结构
#   利用一个1x1卷积下降通道数，然后利用一个3x3卷积提取特征并且上升通道数
#   最后接上一个残差边
# ---------------------------------------------------------------------#
class BasicResidual(nn.Module):
    def __init__(self, res_input_channel, res_out_c1, res_out_c2):
        """
        :param input_channels:
        :param num_channels: [channel_1,channel_2]
        channel_1 是残差网络第一个卷积的输出通道数，channel_2是第二个卷积输出通道数=残差最终输出通道数
        """
        super(BasicResidual, self).__init__()
        self.conv1 = nn.Conv2d(res_input_channel, res_out_c1,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(res_out_c1)
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv2d(res_out_c1, res_out_c2,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(res_out_c2)
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        #残差网络
        out += residual
        return out


class DarkNet(nn.Module):
    def __init__(self, layers_nums):
        """
        :param layers: 列表代表依次每个残差网络的堆叠次数
        """
        super(DarkNet, self).__init__()
        self.res_input_channel = 32  # 第一层卷积后的输出通道数，也是首次进入残差网络的 通道数
        # 416,416,3 -> 416,416,32
        self.conv1 = nn.Conv2d(3, self.res_input_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.res_input_channel)
        self.relu1 = nn.LeakyReLU(0.1)

        # 416,416,32 -> 208,208,64
        self.layer1 = self.make_layer(32, 64, layers_nums[0])
        # 208,208,64 -> 104,104,128
        self.layer2 = self.make_layer(64, 128, layers_nums[1])
        # 104,104,128 -> 52,52,256
        self.layer3 = self.make_layer(128, 256, layers_nums[2])
        # 52,52,256 -> 26,26,512
        self.layer4 = self.make_layer(256, 512, layers_nums[3])
        # 26,26,512 -> 13,13,1024
        self.layer5 = self.make_layer(512, 1024, layers_nums[4])

        self.layers_out_filters = [64, 128, 256, 512, 1024]

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # ---------------------------------------------------------------------#
    #   在每一个layer里面，首先利用一个步长为2的3x3卷积进行下采样
    #   然后进行残差结构的堆叠
    # ---------------------------------------------------------------------#
    def make_layer(self, res_out_c1, res_out_c2, pile_num):
        """
        :param res_out_c1: 残差网络第一个conv输出通道
        :param res_out_c2: 残差网络第二个conv输出通道
        :param pile_num: 残差网络堆叠次数
        :return: 残差网络之前的一个Conv和 堆叠后的同一个残差网络  的网络结构
        """

        layers = []  # 存放一个残差块的网络结构

        # =======进入残差网络之前的下采样，步长为2，卷积核大小为3，缩小一半图片尺寸==============
        layers.append(
            ("ds_conv", nn.Conv2d(self.res_input_channel, res_out_c2, kernel_size=3, stride=2, padding=1, bias=False)))
        layers.append(("ds_bn", nn.BatchNorm2d(res_out_c2)))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))

        # 残差结构
        self.res_input_channel = res_out_c2
        for i in range(0, pile_num):
            layers.append(("BasicResidual_{}".format(i), BasicResidual(self.res_input_channel, res_out_c1, res_out_c2)))

        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        out3 = self.layer3(x)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

        return out3, out4, out5


def darknet53():
    model = DarkNet([1, 2, 8, 8, 4])
    return model


if __name__ == '__main__':
    x = torch.rand((1, 3, 416, 416))

    net = darknet53()
    out3, out4, out5 = net(x)
    print(out3.shape)
    print(out4.shape)
    print(out5.shape)
