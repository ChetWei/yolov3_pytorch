# -*- coding: utf-8 -*-
import glob
import os.path

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import numpy as np
from PIL import Image

from pathlib import Path

from cv2 import cv2

IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes


# 检测的时候的数据加载
class LoadImages:
    def __init__(self, floder_path, transform=None):
        self.transform = transform
        p = str(Path(floder_path).resolve())  # 获得绝对路径
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))
        elif os.path.isfile(p):
            files = [p]
        else:
            raise Exception(f'ERROR: {p} does not exist')

        # 根据文件类型分类
        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        num_i, num_v = len(images), len(videos)

        self.files = images + videos  # 拼接文件路径列表
        self.num_files = num_i + num_v  # 总的文件数量
        self.video_flag = [False] * num_i + [True] * num_v  # 标记vedio
        self.mode = 'image'  # 默认的模式是图片
        if any(videos):
            # 如果有视频文件
            self.new_video(videos[0])
        else:
            self.cap = None

        # 断言文件数量要大于0
        assert self.num_files > 0, f'No images or videos found in {p}. ' \
                                   f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}'

    def __len__(self):
        return self.num_files

    def __iter__(self):
        # 第一次迭代进入的时候会调用
        self.count = 0  # 用count来决定是否进入下一个 迭代文件
        return self

    def __next__(self):
        if self.count == self.num_files:  # 是否所有文件的都遍历结束了 count从0开始
            raise StopIteration
        path = self.files[self.count]  # 获取文件路径

        if self.video_flag[self.count]:  # 读取视频文件
            self.mode = 'video'
            ret_val, img0 = self.cap.read()  # cap对象，读取视频帧图片, ret 是否正确读取文件到达读取完后为false，img为读取的帧
            while not ret_val:  # 如果视频读取结束了
                self.count += 1  # 文件递增
                self.cap.release()  # 释放
                if self.count == self.num_files:
                    # 如果文件都读取完了
                    raise StopIteration
                else:
                    path = self.files[self.count]  # 获得下一个文件
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1  # 用来输出的读取到第几帧
            s = f'video {self.count + 1}/{self.num_files} ({self.frame}/{self.frames}) {path}: '

        else:  # 读取图片文件
            self.count += 1
            img0 = cv2.imread(path)  # BGR

            assert img0 is not None, f'Image Not Found {path}'
            s = f'image {self.count}/{self.num_files} {path}: '

        # 对读取到的BGR图片进行处理 # HWC ,通道由 BGR to RGB
        img = img0[:, :, ::-1]
        # img2 = np.array(
        #     Image.open(path).convert('RGB'),
        #     dtype=np.uint8)
        # 无效填充
        boxes = np.zeros((1, 5))

        # 返回 图片路径和增强后的图片
        if self.transform:
            img, _ = self.transform((img, boxes))

        # 如果是视频要对识别后的图片进行拼接为视频
        return path, img, img0, self.cap, s

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)  # 视频对象
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总的帧数


# 检测的时候加载图片或视频
class ImageFolder2(Dataset):
    def __init__(self, folder_path, transform=None):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.transform = transform  # 数据增强器

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        img = np.array(
            Image.open(img_path).convert('RGB'),
            dtype=np.uint8)

        # Label Placeholder  填充类别
        boxes = np.zeros((1, 5))

        # Apply transforms
        if self.transform:
            img, _ = self.transform((img, boxes))

        return img_path, img

    def __len__(self):
        return len(self.files)


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

        # Label Placeholder  填充类别
        boxes = np.zeros((1, 5))

        # Apply transforms
        if self.transform:
            img, _ = self.transform((img, boxes))

        return img_path, img

    def __len__(self):
        return len(self.files)


# 训练时候的数据集
class ListDataSet(Dataset):
    def __init__(self, labels_file, img_home, input_shape, transform):
        self.img_home = img_home  # 图片存放目录路径
        self.input_shape = input_shape  # (w h)
        self.transform = transform

        super(ListDataSet, self).__init__()
        with open(labels_file, 'r') as f:
            self.lines = [line.strip() for line in f.readlines()]

    def __getitem__(self, index):
        # 读取一行标签
        line = self.lines[index % len(self.lines)]
        line_splited = line.split(" ")
        img_path = os.path.join(self.img_home, line_splited[0])
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
        # 设置属于一个batchsize中的第几个数据
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
    # lable = "/Users/weimingan/work/python_code/myyolo/yolov3_pytorch/code/voc2007_train.txt"
    # dataset = ListDataSet(labels_file=lable, input_shape=[416, 416])
    # dataset.__getitem__(5)
    # dataset = LoadImages(floder_path="/Users/weimingan/work/dataset/voc2012/VOC2012/JPEGImages",transform=DEFAULT_TRANSFORMS)
    #
    # for data in dataset:
    #     print(data)
    pass
