# -*- coding: utf-8 -*-
import argparse
import os
import tqdm
import random
import numpy as np

from PIL import Image, ImageDraw, ImageFont

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from utils.util import get_classes_name, non_max_suppression, rescale_boxes,Colors
from utils.datasets import ImageFolder
from utils.transforms import Resize, DEFAULT_TRANSFORMS
from Yolo3Body import YOLOV3

from pathlib import Path




def _create_data_loader(img_path, batch_size, img_size, n_cpu):
    dataset = ImageFolder(
        img_path,
        transform=transforms.Compose([DEFAULT_TRANSFORMS, Resize(img_size)]))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu,
        pin_memory=True)
    return dataloader


def detect(model, dataloader, output_path, img_size, conf_thres, nms_thres):
    """

    :param model: 训练好的模型
    :param dataloader: 数据加载器
    :param output_path: 图片保存目录
    :param img_size:
    :param conf_thres:
    :param nms_thres:
    :return:
    """
    os.makedirs(output_path, exist_ok=True)
    model.eval()  # 设置模型为评估

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    img_detections = []  # Stores detections for each image index
    imgs = []  # 存储图片路径

    for (img_paths, input_imgs) in tqdm.tqdm(dataloader, desc="Detecting"):
        # dataloader 加载 img_path:检测图片的路径, input_img:待检测的tensor图片数据
        input_imgs = Variable(input_imgs.type(Tensor))
        # 预测
        with torch.no_grad():
            # (batch_size,num,25)
            detections = model(input_imgs)
            # Non-Maximum Suppression 非极大抑制
            detections = non_max_suppression(detections, conf_thres, nms_thres)

        # Store image and detections
        img_detections.extend(detections)
        imgs.extend(img_paths)

    return img_detections, imgs


def _draw_and_save_output_images(img_detections, imgs, img_size, output_path, classes):
    for (image_path, detections) in zip(imgs, img_detections):
        print(f"Image {image_path}:")
        _draw_and_save_output_image(
            image_path, detections, img_size, output_path, classes)



def check_font(font='../config/Arial.ttf', size=10):
    font = Path(font)
    return ImageFont.truetype(str(font) if font.exists() else font.name, size)

colors = Colors()
def _draw_and_save_output_image(image_path, detections, img_size, output_path, classes):
    pil_img = Image.open(image_path)
    np_img = np.array(pil_img)
    default_fontsize = max(round(sum(pil_img.size) / 2 * 0.035), 12)
    font = check_font(size=default_fontsize)
    line_width = max(round(sum(np_img.shape) / 2 * 0.003), 2)  # 目标框的线条宽度
    draw = ImageDraw.Draw(pil_img)
    txt_color = (255, 255, 255) #默认是白色的字体
    img = np.array(Image.open(image_path))  # 打开原图像
    # Rescale boxes to original image
    detections = rescale_boxes(detections, img_size, img.shape[:2])

    for x1, y1, x2, y2, conf, cls_pred in detections:
        print(f"\t+ Label: {classes[int(cls_pred)]} | Confidence: {conf.item():0.4f}")
        color = colors(int(cls_pred))
        # 1.先画目标检测框
        #x1, y1, x2, y2 = x1.detach().numpy().tolist(), y1.detach().numpy().tolist(), x2.detach().numpy().tolist(), y2.detach().numpy().tolist()
        draw.rectangle([x1, y1, x2, y2], width=line_width, outline=color)
        # 2.画左上角标签 包括框框和text
        # 获取标签字体的的宽度
        class_name = classes[int(cls_pred)]
        label = class_name+" "+ str(round(conf.numpy().tolist(),2))
        w, h = font.getsize(label)
        not_outside = y1 - h >= 0  # 判断label是否超出了图片范围
        # 2.1先画label的框框
        draw.rectangle([x1,
                        y1 - h if not_outside else y1,
                        x1 + w + 1,
                        y1 + 1 if not_outside else y1 + h + 1], fill=color)
        # 2.2画label的字体
        draw.text([x1,
                   y1 - h if not_outside else y1], label, fill=txt_color, font=font)

    filename = os.path.basename(image_path).split(".")[0]
    output_path = os.path.join(output_path, f"{filename}.png")
    # 保存图片 bbox_inches="tight" figure自适应图片尺寸 pad_inches去掉空白
    pil_img.save(output_path)

def run():
    from config import anchors_mask_list

    parser = argparse.ArgumentParser(description="检测图片目标")

    parser.add_argument("-w", "--weight_path", type=str, default="/Users/weimingan/work/weights/yolov3_voc_200.pth",
                        help="权重文件")
    parser.add_argument("-i", "--img_dir", type=str, default="../data/sample", help="待检测图片存放路径")
    parser.add_argument("-c", "--classes", type=str, default="../config/voc_names.txt", help="类别文件")
    parser.add_argument("-o", "--output_path", type=str, default="../output", help="输出文件夹")
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("--img_size", type=int, default=416, help="输入Yolo的图片尺度")

    parser.add_argument("--n_cpu", type=int, default=1, help="dataloader的线程")
    parser.add_argument("--conf_thres", type=float, default=0.3, help="有物体置信度阈值")
    parser.add_argument("--nms_thres", type=float, default=0.6, help="NMS阈值")

    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    # 获取制作数据集时候设置的类别列表
    classes = get_classes_name(args.classes)

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设置非训练模型加载
    model = YOLOV3(cls_num=len(classes), anchors=anchors_mask_list, img_size=args.img_size, training=False).to(device)
    # 加载模型参数
    model.load_state_dict(torch.load(args.weight_path, map_location=device))
    # 创建dataloader
    dataloader = _create_data_loader(args.img_dir, args.batch_size, args.img_size, args.n_cpu)

    # 检测图片
    img_detections, imgs = detect(model, dataloader, args.output_path, args.img_size, args.conf_thres, args.nms_thres)

    # 画图并输出图片
    _draw_and_save_output_images(img_detections, imgs, args.img_size, args.output_path, classes)


if __name__ == '__main__':
    run()
