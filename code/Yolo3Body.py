# !/usr/bin/env python
# -*-coding:utf-8 -*-

"""
# Time       ：2021/11/3 13:49
# Author     ：thomas
# Description：
"""
from itertools import chain

from collections import OrderedDict
import torch
import torch.nn as nn

from Backbone import darknet53

"""
一个完整的卷积 包括 conv2d bn relu
"""


def CBL(channel_in, channel_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    net = nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(channel_in, channel_out, kernel_size=kernel_size,
                           stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(channel_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

    return net


def make_last_layers(filters_list, first_in_filter, final_out_filter):
    """
    最后7层网络，5个CBL层 + 1个CBL层 + 1个Conv层
    :param filters_list:  在BCL层的通道变换情况列表 长度为2的list
    :param first_in_filter: 首次进入7层网络 输入的通道数
    :param final_out_filter: 最后返回的通道数
    :return: 返回
    """
    m = nn.Sequential(
        # CBL * 5
        CBL(first_in_filter, filters_list[0], kernel_size=1),
        CBL(filters_list[0], filters_list[1], kernel_size=3),
        CBL(filters_list[1], filters_list[0], kernel_size=1),
        CBL(filters_list[0], filters_list[1], kernel_size=3),
        CBL(filters_list[1], filters_list[0], kernel_size=1),

        CBL(filters_list[0], filters_list[1], kernel_size=3),
        # 最后一个卷积 为了 转换输出通道
        nn.Conv2d(filters_list[1], final_out_filter, kernel_size=1, stride=1, padding=0, bias=True)
    )
    return m


class YOLOV3(nn.Module):
    def __init__(self, cls_num, anchors, img_size=416):
        """
        :param cls_num: 类别个数
        """
        super(YOLOV3, self).__init__()
        # 生成darknet53的主干网络
        # 输入尺寸 412x412
        # 获得darknet53的最后三个特征层输出 shape
        # 52,52, 256
        # 26, 26, 512
        # 13, 13, 1024
        self.backbone = darknet53()

        self.cls_num = cls_num
        self.img_size = img_size

        # backbone中每个残差网络的输出通道数
        # out_filters = [64, 128, 256, 512, 1024]
        out_filters = self.backbone.layers_out_filters

        """13x13x255输出层"""
        self.last_layer0 = make_last_layers(
            filters_list=[512, 1024],
            first_in_filter=out_filters[-1],  # res最后一层出来的输出作为输入
            final_out_filter=3 * (self.cls_num + 5)  # 设置最后卷积层输出的通道数
        )

        """26x26x255输出层"""
        # 最后一个resnet进过 CBL*5 特征提取后 加入到下一个输出堆叠
        # 经过 一个CBL + 上采样
        self.last_layer1_CBL_1 = CBL(512, 256, kernel_size=1)
        # 上采样，2倍放大 13x13xchannel -> 26x26xchannel
        self.last_layer1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer1 = make_last_layers(
            filters_list=[256, 512],  #
            first_in_filter=out_filters[-2] + 256,  # res倒数第二个输出通道数+堆叠上一层采样后的通道数
            final_out_filter=3 * (self.cls_num + 5)
        )

        """52x52x255输出层"""
        self.last_layer2_CBL_1 = CBL(256, 128, kernel_size=1)
        self.last_layer2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2 = make_last_layers(
            filters_list=[128, 256],
            first_in_filter=out_filters[-3] + 128,
            final_out_filter=3 * (self.cls_num + 5)
        )

        self.yolo_out_layer = [YoloOutLayer(anchors[0], self.cls_num, self.img_size),
                                YoloOutLayer(anchors[1], self.cls_num, self.img_size),
                                YoloOutLayer(anchors[2], self.cls_num, self.img_size)]

    def forward(self, x):
        #  获得三个有效特征层，他们的shape分别是：
        #   52,52,256; 26,26,512; 13,13,1024
        x2, x1, x0 = self.backbone(x)

        # ---------------------------------------------------#
        #   第一个特征层
        #   out0 = (batch_size,255,13,13)
        # ---------------------------------------------------#
        # 经过5个CBL输出特征  保留给下一层
        out0_branch = self.last_layer0[:5](x0)
        # 输出第一个特征层的最后结果
        head_out0 = self.last_layer0[5:](out0_branch)

        # ---------------------------------------------------#
        #   第二个特征层
        #   out1 = (batch_size,255,26,26)
        # ---------------------------------------------------#
        # 上一层的特征 经过一个CBL 和 上采样
        x1_in = self.last_layer1_CBL_1(out0_branch)
        x1_in = self.last_layer1_upsample(x1_in)
        # 堆叠 channel
        x1_in = torch.cat([x1_in, x1], 1)
        # 经过5个CBL输出特征  并保留给下一层
        out1_branch = self.last_layer1[:5](x1_in)
        # 输出第二个特征层的最后结果
        head_out1 = self.last_layer1[5:](out1_branch)

        # ---------------------------------------------------#
        #   第三个特征层
        #   out1 = (batch_size,255,52,52)
        # ---------------------------------------------------#
        x2_in = self.last_layer2_CBL_1(out1_branch)
        x2_in = self.last_layer2_upsample(x2_in)
        # 堆叠 channel
        x2_in = torch.cat([x2_in, x2], 1)
        head_out2 = self.last_layer2(x2_in)

        out0 = self.yolo_out_layer[0](head_out0)
        out1 = self.yolo_out_layer[1](head_out1)
        out2 = self.yolo_out_layer[2](head_out2)

        return out0, out1, out2


class YoloOutLayer(nn.Module):
    "检测之前的预数据处理"

    def __init__(self, anchors, num_classes, img_size):
        """
        :param anchors: [[w,h],[w,h],[w,h]]
        :param num_classes: 类别个数
        """
        super(YoloOutLayer, self).__init__()
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.num_attr = num_classes + 5
        self.img_size = img_size  # 图片尺寸 默认是正方形
        #
        anchors = torch.tensor(list(chain(*anchors))).float().view(-1, 2)
        self.register_buffer('anchors', anchors)
        self.register_buffer(
            'anchor_grid', anchors.clone().view(1, -1, 1, 1, 2))  # ？
        self.stride = None  # 先定义

    def forward(self, x):
        # 调整一下维度，并设置输出维度的一些基本信息
        stride = self.img_size // x.size(2)  # 单元格的长度
        self.stride = stride
        bs, _, feature_h, feature_w = x.shape
        # [bs, 75, 13, 13] -> [bs,3,25,13,13] -> [bs,3,13,13,25]
        x = x.view(bs, self.num_anchors, self.num_attr, feature_h, feature_w).permute(0, 1, 3, 4, 2).contiguous()

        return x
        # torch.Size([1, 3, 13, 13, 25])
        # torch.Size([1, 3, 26, 26, 25])
        # torch.Size([1, 3, 52, 52, 25])


if __name__ == '__main__':
    x = torch.rand((1, 3, 416, 416))
    #   13x13的特征层对应的anchor是[116,90],[156,198],[373,326]
    #   26x26的特征层对应的anchor是[30,61],[62,45],[59,119]
    #   52x52的特征层对应的anchor是[10,13],[16,30],[33,23]
    anchors_mask1 = [[116, 90], [156, 198], [373, 326]]
    anchors_mask2 = [[30, 61], [62, 45], [59, 119]]
    anchors_mask3 = [[10, 13], [16, 30], [33, 23]]
    anchors_mask_list = [anchors_mask1, anchors_mask2, anchors_mask3]
    num_classes = 20

    model = YOLOV3(20, anchors_mask_list)

    #
    final_out0, final_out1, final_out2 = model(x)
    # (80+5)*3 = 255
    print(final_out0.shape)  # torch.Size([1, 255, 13, 13])
    print(final_out1.shape)  # torch.Size([1, 255, 26, 26])
    print(final_out2.shape)  # torch.Size([1, 255, 52, 52])
