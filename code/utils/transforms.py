# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import numpy as np

import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

# from util import xywh2xyxy_np
import torchvision.transforms as transforms


class ImgAug(object):
    def __init__(self, augmentations=[]):
        self.augmentations = augmentations

    def __call__(self, data):
        """
        :param data[0]: PIL转为numpy的图片数据
        :param data[1]: 目标框坐标信息 [classid,x1,y1,x2,y2]
        :return:
        """

        img, boxes = data
        # 转换 xywh to x1y1x2y2
        # boxes = np.array(boxes)
        # boxes[:, 1:] = xywh2xyxy_np(boxes[:, 1:])

        # 转换目标框到imgaug ，这样就可以更正增强之后的位置
        bounding_boxes = BoundingBoxesOnImage(
            [BoundingBox(*box[1:], label=box[0]) for box in boxes],  # box默认参数坐标是 x1y1x2y2
            shape=img.shape)

        # 应用数据增强
        img, bounding_boxes = self.augmentations(image=img, bounding_boxes=bounding_boxes)

        # 数据增强之后获取更正后的目标框位置
        bounding_boxes = bounding_boxes.clip_out_of_image()
        # 转换为numpy
        boxes = np.zeros((len(bounding_boxes), 5))
        for box_idx, box in enumerate(bounding_boxes):
            # xyxy
            boxes[box_idx, 0] = box.label
            boxes[box_idx, 1] = box.x1
            boxes[box_idx, 2] = box.y1
            boxes[box_idx, 3] = box.x2
            boxes[box_idx, 4] = box.y2

        # 返回 (x1, y1, x2, y2)
        return img, boxes


# 坐标转换
# bounding_boxes = bounding_boxes.clip_out_of_image() 传入的bbox
class CoordinateTransform(object):
    def __init__(self, xyxy2xywh=True):
        self.xyxy2xywh = xyxy2xywh
        pass

    def __call__(self, data):
        img, bboxes = data
        if self.xyxy2xywh:
            bboxes[:, 3:5] = bboxes[:, 3:5] - bboxes[:, 1:3]  # w,h
            bboxes[:, 1:3] = bboxes[:, 1:3] + bboxes[:, 3:5] / 2  # x,y

        return img, bboxes


# 根据宽高获得归一化的坐标
class RelativeLabels(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes = data
        h, w, _ = img.shape
        boxes[:, [1, 3]] /= w
        boxes[:, [2, 4]] /= h
        return img, boxes


# 将归一化的bbox坐标调整为绝对坐标
class AbsoluteLabels(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes = data
        h, w, _ = img.shape
        boxes[:, [1, 3]] *= w
        boxes[:, [2, 4]] *= h
        return img, boxes


# 将图片pad成正方形
class PadSquare(ImgAug):
    def __init__(self, ):
        self.augmentations = iaa.Sequential([
            iaa.PadToAspectRatio(
                1.0,
                position="center-center").to_deterministic()
        ])


# 将数据转为tensor
class ToTensor(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes = data
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(img)
        # 在bbox上增加一个值，用来表示当前数据属于当前batch的第几个，存batch_id
        bb_targets = torch.zeros((len(boxes), 6))
        bb_targets[:, 1:] = transforms.ToTensor()(boxes)  # 自动归一化为tensor 0-1

        return img, bb_targets


# 标准化
class Normalize(object):
    def __init__(self, mean=[.5, .5, .5], std=[.5, .5, .5], inplace=False):
        """
        :param mean: 各个通道的均值
        :param std: 各个通道的标准差
        :param inplace: 是否原地操作
        """
        self.t = transforms.Normalize(mean,std,inplace)

    def __call__(self, data):
        tensor_img, boxes = data
        tensor_img = self.t(tensor_img)
        return tensor_img, boxes


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        img, boxes = data
        img = F.interpolate(img.unsqueeze(0), size=self.size, mode="nearest").squeeze(0)
        return img, boxes


# 检测的时候使用，没有使用数据增强
DEFAULT_TRANSFORMS = transforms.Compose([
    # AbsoluteLabels(),
    PadSquare(),
    CoordinateTransform(),  # 坐标转换
    RelativeLabels(),
    ToTensor(),
    Normalize()
])

if __name__ == '__main__':
    from PIL import Image, ImageDraw
    import numpy as np

    pil_img = Image.open("/Users/weimingan/work/dataset/VOCdevkit/VOC2007/JPEGImages/000030.jpg")
    np_img = np.array(pil_img)
    # x1y1x2y2
    a = [[14, 36, 205, 180, 289], [10, 51, 160, 150, 292], [10, 295, 138, 450, 290]]
    bbox = np.array(a)

    img, bb_targets = DEFAULT_TRANSFORMS((np_img, bbox))

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    for box in bb_targets.tolist():
        cls_idx, xc, yc, w, h = box
        x1, y1 = xc - w / 2.0, yc - h / 2.0
        x2, y2 = xc + w / 2.0, yc + h / 2.0

        draw.rectangle((x1, y1, x2, y2), outline=(102, 255, 102), width=2)

    pil_img.show()
