# -*- coding: utf-8 -*-
import re

import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

from PIL import Image, ImageDraw, ImageFont
import numpy as np

from pathlib import Path

import torchvision.transforms as transforms


class ImgAug(object):
    def __init__(self, augmentations=[]):
        self.augmentations = augmentations

    def __call__(self, data):
        print("ImgAug")


class RelativateLabels(object):
    """
    目标框坐标根据宽高归一化
    """

    def __init__(self):
        pass

    def __call__(self, data):
        img, boxes = data
        h, w, _ = img.shape
        boxes[:, [1, 3]] /= w
        boxes[:, [2, 4]] /= h
        return img, boxes


# 14,36,205,180,289 10,51,160,150,292 10,295,138,450,290
def test():
    pil_img = Image.open("/Users/weimingan/work/dataset/VOCdevkit/VOC2007/JPEGImages/000030.jpg")
    np_img = np.array(pil_img)

    bbs = BoundingBoxesOnImage([
        BoundingBox(x1=36, y1=205, x2=180, y2=289, label="bicycle"),
        BoundingBox(x1=51, y1=160, x2=150, y2=292, label="person"),
        BoundingBox(x1=295, y1=138, x2=450, y2=290, label="person")
    ], shape=np_img.shape)

    bounding_boxes = bbs.clip_out_of_image()
    print(bounding_boxes)

    # Rescale image and bounding boxes
    image_rescaled = ia.imresize_single_image(np_img, (1000, 1000))
    bbs_rescaled = bbs.on(image_rescaled)  # 获得放缩后的坐标

    bounding_boxes = bbs_rescaled.clip_out_of_image()
    print(bounding_boxes)

    # Draw image before/after rescaling and with rescaled bounding boxes
    image_bbs = bbs.draw_on_image(np_img, size=2)
    image_rescaled_bbs = bbs_rescaled.draw_on_image(image_rescaled, size=2)

    pil_i = Image.fromarray(image_rescaled_bbs)
    pil_i.show()


def test2():
    # 先将图片进行增强，再进行图片和bbox的调整  最后返回归一化后的坐标

    pil_img = Image.open("/Users/weimingan/work/dataset/VOCdevkit/VOC2007/JPEGImages/000030.jpg")
    np_img = np.array(pil_img)

    aug = iaa.Sequential([
        # position="center-center" 两边填充
        iaa.PadToAspectRatio(aspect_ratio=1.0, position="center-center").to_deterministic()
    ])

    img_aug = aug(image=np_img)

    pil_i = Image.fromarray(img_aug)
    pil_i.show()


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

# 判断是否为中文字符
def is_chinese(s='人工智能'):
    # Is string composed of any Chinese characters?
    return re.search('[\u4e00-\u9fff]', s)


def check_font(font='../config/Arial.ttf', size=10):
    font = Path(font)
    return ImageFont.truetype(str(font) if font.exists() else font.name, size)

# pil库进行标记图片目标
def draw(img_path, boxes, label="person"):
    # 打开图片
    pil_img = Image.open(img_path)
    np_img = np.array(pil_img)
    default_fontsize = max(round(sum(pil_img.size) / 2 * 0.035), 12)
    font = check_font(size=default_fontsize)
    line_width = max(round(sum(np_img.shape) / 2 * 0.003), 2) #目标框的线条宽度
    draw = ImageDraw.Draw(pil_img)
    color = colors(0)
    txt_color = (255, 255, 255)

    for box in boxes:
        id, x1, y1, x2, y2 = box
        # 1.先画目标检测框
        draw.rectangle([x1, y1, x2, y2], width=line_width, outline=color)
        # 2.画左上角标签 包括框框和text
        # 获取标签字体的的宽度
        w, h = font.getsize(label)
        not_outside = y1 - h >= 0  # 判断label是否超出了图片范围
        # 先画label的框框
        draw.rectangle([x1,
                        y1 - h if not_outside else y1,
                        x1 + w + 1,
                        y1 + 1 if not_outside else y1 + h + 1], fill=color)
        # 画label的字体
        draw.text([x1, y1 - h if not_outside else not_outside], label, fill=txt_color,font=font)

    pil_img.show()


if __name__ == '__main__':
    import yaml
    import os
    # colors = Colors()
    # boxes = [[14.6, 36, 205, 180, 289], [10, 51, 160, 150, 292], [10, 295, 138, 450, 290]]
    # path = "/Users/weimingan/work/dataset/VOCdevkit/VOC2007/JPEGImages/000030.jpg"
    # draw(path, boxes)
    pass
