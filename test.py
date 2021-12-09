# -*- coding: utf-8 -*-
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

from PIL import Image
import numpy as np

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


if __name__ == '__main__':
    t = transforms.Compose([

    ])


