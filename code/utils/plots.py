# -*- coding: utf-8 -*-

from PIL import ImageFont,Image,ImageDraw
from pathlib import Path
from cv2 import cv2
import re
import numpy as np


class Colors:
    def __init__(self):
        hex = ('3366ff', '99ff99', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '36b8c9', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()


def check_font(font='../config/Arial.ttf', size=10):
    # Return a PIL TrueType Font
    font = Path(font)
    return ImageFont.truetype(str(font) if font.exists() else font.name, size)


def is_ascii(s=''):
    # Is string composed of all ASCII (no UTF) characters? (note str().isascii() introduced in python 3.7)
    s = str(s)  # convert list, tuple, None, etc. to str
    return len(s.encode().decode('ascii', 'ignore')) == len(s)


def is_chinese(s='人工智能'):
    # Is string composed of any Chinese characters?
    return re.search('[\u4e00-\u9fff]', s)


class Annotator:
    def __init__(self, im, line_width=None, font_size=None, font='../config/Arial.ttf', pil=False, example='abc'):
        """
        :param im: 传入的numpy图片数据
        :param line_width: 线条宽度
        :param font_size: 字体大小
        :param font: 字体
        :param pil: 传入的是否为PIL读取的图片 否则为cv2 读取的图片
        :param example:
        """
        assert im.data.contiguous, 'Image not contiguous'
        self.pil = pil or not is_ascii(example) or is_chinese(example)
        # 判断传入的图片数据类型是 PIL 还是 cv2
        if self.pil:  # use PIL
            self.im = im if isinstance(im, Image.Image) else Image.fromarray(im)
            self.draw = ImageDraw.Draw(self.im)
            self.font = check_font(font='Arial.Unicode.ttf' if is_chinese(example) else font,
                                   size=font_size or max(round(sum(self.im.size) / 2 * 0.035), 12))
        else:  # use cv2
            self.im = im
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width

    # 画目标框和标签
    def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
        """
        :param box: [x1,y1,x2,y2]
        :param label: 标签名称
        :param color: 目标框的颜色
        :param txt_color: 字体的颜色 默认白色
        :return:
        """
        if self.pil or not is_ascii(label):
            self.draw.rectangle(box, width=self.lw, outline=color)  # 画框
            if label:
                w, h = self.font.getsize(label)  # 获取字体的尺寸
                outside = box[1] - h >= 0  # 避免字体超出图片
                self.draw.rectangle([box[0],
                                     box[1] - h if outside else box[1],
                                     box[0] + w + 1,
                                     box[1] + 1 if outside else box[1] + h + 1], fill=color)
                self.draw.text((box[0], box[1] - h if outside else box[1]), label, fill=txt_color, font=self.font)

        else:  # cv2
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)  # 画目标框
            if label:
                tf = max(self.lw - 1, 1)  # font thickness
                w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]  # text width, height
                outside = p1[1] - h - 3 >= 0  # label fits outside box
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                # 画标签
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(self.im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), 0, self.lw / 3, txt_color,
                            thickness=tf, lineType=cv2.LINE_AA)

    # 使用PIL画目标框
    def rectangle(self, xy, fill=None, outline=None, width=1):
        """
        :param xy: [x1,y1,x2,y2]
        :param fill: 填充的颜色
        :param outline: 目标框的颜色
        :param width: 默认为1
        :return:
        """
        self.draw.rectangle(xy, fill, outline, width)

    # 使用PIL写标签
    def text(self, xy, text, txt_color=(255, 255, 255)):
        """
        :param xy: [x1,y1,x2,y2]
        :param text: 字的内容
        :param txt_color: 字的颜色 默认白色
        :return:
        """
        w, h = self.font.getsize(text)  # text width, height
        self.draw.text((xy[0], xy[1] - h + 1), text, fill=txt_color, font=self.font)

    # 返回numpy图片
    def result(self):
        return np.asarray(self.im)

