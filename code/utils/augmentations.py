# -*- coding: utf-8 -*-
import imgaug.augmenters as iaa
from torchvision import transforms
from .transforms import ToTensor, PadSquare, RelativeLabels, AbsoluteLabels, ImgAug

class DefaultAug(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.Sharpen((0.0, 0.1)), #增强器 锐化图像并将结果与原始图像叠加
            #创建一个适用的增强器仿射缩放（放大/缩小）到图像。
            # 它们被缩放到它们大小的 80-150%
            iaa.Affine(rotate=(-0, 0), translate_percent=(-0.1, 0.1), scale=(0.8, 1.5)),
            iaa.AddToBrightness((-60, 40)), #增加亮度
            iaa.AddToHue((-10, 10)), #离散的均匀范围中采样随机值[-10..10]，，即添加到颜色空间中的的H通道中HSV
            iaa.Fliplr(0.5), #水平翻转整个图像的 50%
        ])


class StrongAug(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.Dropout([0.0, 0.01]),
            iaa.Sharpen((0.0, 0.1)),
            iaa.Affine(rotate=(-10, 10), translate_percent=(-0.1, 0.1), scale=(0.8, 1.5)),
            iaa.AddToBrightness((-60, 40)),
            iaa.AddToHue((-20, 20)),
            iaa.Fliplr(0.5),
        ])

#训练的时候使用
AUGMENTATION_TRANSFORMS = transforms.Compose([
    #AbsoluteLabels(), #目标检测框坐标 从归一化数据转为绝对数据
    DefaultAug(), #默认的数据增强
    PadSquare(), #padding图像为正方形
    RelativeLabels(),  #根据宽高获得归一化后的坐标
    ToTensor(), #转为tensor数据
])