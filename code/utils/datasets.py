# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset

import numpy as np
from PIL import Image, ImageDraw

import torchvision.transforms as transforms


class ListDataSet(Dataset):
    def __init__(self, labels_file, input_shape):
        self.input_shape = input_shape  # (w h)
        super(ListDataSet, self).__init__()
        with open(labels_file, 'r') as f:
            self.lines = [line.strip() for line in f.readlines()]

    def __getitem__(self, index):
        # 读取一行标签
        line = self.lines[index % len(self.lines)]
        line_splited = line.split(" ")
        img_path = line_splited[0]
        pil_img = Image.open(img_path).convert('RGB')  # 将通道的转为3通道
        pad_w, pad_h, new_im = pad2square(pil_img)
        # 将图片resize成输入的尺寸
        resized_img = new_im.resize(self.input_shape, Image.BICUBIC)

        scale_w, scale_h = self.input_shape[0] / new_im.size[0], self.input_shape[1] / new_im.size[1]

        #  并调整坐标形式 xyxy -> xywh， 将根据padding调整box ,再归一化,
        boxes = np.array([box.split(",") for box in line_splited[1:]], dtype=np.float32)
        boxes = adjust_box(boxes, pad_w, pad_h, scale_w, scale_h, self.input_shape)

        # -------------验证调整情况-------------
        # draw = ImageDraw.Draw(resized_img)
        # for box in boxes:
        #     box[1:] = box[1:] * 416
        #     x, y, w, h = box[1], box[2], box[3], box[4]
        #     x1, y1 = x - w * 0.5, y - h * 0.5
        #     x2, y2 = x + w * 0.5, y + h * 0.5
        #     draw.rectangle((x1, y1, x2, y2), outline=(102, 255, 102), width=2)
        # resized_img.show()

        # 归一化图片并转换维度 np.array(PIL)是 H,W,3
        image_data = normalize_img(np.array(resized_img, dtype=np.float32))
        # 转为tensor数据 默认放在cpu上
        image_tensor = transforms.ToTensor()(image_data)  # 自动转换为 H,W,3 -> 3,H,W
        # 在bbox上增加一个值，用来表示当前数据属于当前batch的第几个，存batch_id
        bb_targets = torch.zeros(len(boxes), 6)
        bb_targets[:, 1:] = transforms.ToTensor()(boxes)
        # box xywh 是归一化数据
        return image_tensor, bb_targets

    def __len__(self):
        return len(self.lines)

    def collate_fn(self, batch):
        imgs, bb_targets = list(zip(*batch))
        # 设置数据在这个batch中的batchid
        imgs = torch.stack(imgs)

        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        # 将这个batch的多个数据，整合在一起 （nums,6）,因为数据里面设置了对应的batchid 所以后面可以找到属于哪个batch
        bb_targets = torch.cat(bb_targets, dim=0)

        # imgs (batchsize,3,416,416)
        # bb_targets (num,6)
        return imgs, bb_targets


def pad2square(pil_img):
    iw, ih = pil_img.size
    size = max(iw, ih)
    fill_color = (0, 0, 0)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(pil_img, ((size - iw) // 2, (size - ih) // 2))
    pad_w, pad_h = (size - iw) * 0.5, (size - ih) * 0.5
    return pad_w, pad_h, new_im


def adjust_box(bboxs, pw, ph, scale_w, scale_h, input_shape):
    # 调整点 xyxy
    bboxs[:, [1, 3]] = bboxs[:, [1, 3]] + pw
    bboxs[:, [2, 4]] = bboxs[:, [2, 4]] + ph
    # resize
    bboxs[:, [1, 3]] = bboxs[:, [1, 3]] * scale_w
    bboxs[:, [2, 4]] = bboxs[:, [2, 4]] * scale_h

    #  并调整坐标形式 xyxy -> xywh，,再归一化,
    bboxs[:, 3:5] = bboxs[:, 3:5] - bboxs[:, 1:3]  # w,h
    bboxs[:, 1:3] = bboxs[:, 1:3] + bboxs[:, 3:5] * 0.5  # x,y

    # 坐标值归一化
    bboxs[:, [1, 3]] = bboxs[:, [1, 3]] / input_shape[0]  # x w
    bboxs[:, [2, 4]] = bboxs[:, [2, 4]] / input_shape[1]  # y h

    return bboxs


def normalize_img(image):
    """
    #对图像 像素值进行归一化
    :param image:
    :return: numpy格式的图片数据
    """
    image /= 255.0
    return image


if __name__ == '__main__':
    lable = "/Users/weimingan/work/python_code/myyolo/yolov3_pytorch/code/voc2007_train.txt"
    dataset = ListDataSet(labels_file=lable, input_shape=[416, 416])
    dataset.__getitem__(5)
