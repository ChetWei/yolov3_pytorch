# -*- coding: utf-8 -*-
import glob
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import numpy as np
from PIL import Image



# 检测的时候读取图片路径
class ImageFolder(Dataset):
    def __init__(self, folder_path, transform=None):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.transform = transform  # 数据增强器

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        img = np.array(
            Image.open(img_path).convert('RGB'),
            dtype=np.uint8)

        # Label Placeholder
        boxes = np.zeros((1, 5))

        # Apply transforms
        if self.transform:
            img, _ = self.transform((img, boxes))

        return img_path, img

    def __len__(self):
        return len(self.files)


# 训练时候的数据集
class ListDataSet(Dataset):
    def __init__(self, labels_file, input_shape, transform):
        self.input_shape = input_shape  # (w h)
        self.transform = transform
        super(ListDataSet, self).__init__()
        with open(labels_file, 'r') as f:
            self.lines = [line.strip() for line in f.readlines()]

    def __getitem__(self, index):
        # 读取一行标签
        line = self.lines[index % len(self.lines)]
        line_splited = line.split(" ")
        img_path = line_splited[0]
        boxes = np.array([box.split(",") for box in line_splited[1:]], dtype=np.float32)
        pil_img = Image.open(img_path).convert('RGB')  # 将通道的转为3通道
        img = np.array(pil_img, dtype=np.uint8)
        if self.transform:
            img, bb_targets = self.transform((img, boxes))

        return img, bb_targets

    def __len__(self):
        return len(self.lines)


    def collate_fn(self, batch):
        imgs, bb_targets = list(zip(*batch))
        # 将pad正方形图片resize成模型输入的大小   拼接batchsize
        imgs = torch.stack([resize(img, self.input_shape[0]) for img in imgs])
        #设置属于一个batchsize中的第几个数据
        for i, boxes in enumerate(bb_targets):
            boxes[:, 0] = i
        # 将这个batch的多个数据，整合在一起 （nums,6）,因为数据里面设置了对应的batchid 所以后面可以找到属于哪个batch
        bb_targets = torch.cat(bb_targets, dim=0)

        # imgs (batchsize,3,416,416)
        # bb_targets (num,6)
        return imgs, bb_targets


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


if __name__ == '__main__':
    lable = "/Users/weimingan/work/python_code/myyolo/yolov3_pytorch/code/voc2007_train.txt"
    dataset = ListDataSet(labels_file=lable, input_shape=[416, 416])
    dataset.__getitem__(5)
