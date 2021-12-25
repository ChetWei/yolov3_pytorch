# -*- coding: utf-8 -*-
import argparse
import os
import tqdm
import numpy as np

from PIL import Image, ImageDraw

import yaml
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from utils.util import get_classes_name, non_max_suppression, rescale_boxes
from utils.datasets import ImageFolder, LoadImages
from utils.transforms import Resize, DEFAULT_TRANSFORMS
from yolov3_pytorch.models.Yolo3Body import YOLOV3

from pathlib import Path

from utils.plots import Annotator, check_font, colors

from cv2 import cv2


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
            # Non-Maximum Suppression 非极大抑制 shape（num,6）
            detections = non_max_suppression(detections, conf_thres, nms_thres)

        # Store image and detections
        img_detections.extend(detections)
        imgs.extend(img_paths)

    return img_detections, imgs


# 使用视频检测的dataset
def detect2(model, input_dir, output_dir, img_size, conf_thres, nms_thres, classes):
    """
    :param model: 训练好的模型
    :param input_dir: 输入图片的目录
    :param output_dir: 图片保存目录
    :param img_size: 模型的输入尺寸
    :param conf_thres: 置信度阈值
    :param nms_thres: nms阈值
    :return:
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()  # 设置模型为评估

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # Dataloader
    transform = transforms.Compose([DEFAULT_TRANSFORMS, Resize(img_size)])
    dataset = LoadImages(input_dir, transform)

    vid_path, vid_writer = None, None  # 视频输出标记
    # 加载数据
    for path, image, im0s, vid_cap, s in dataset:
        input_image = Variable(image.type(Tensor))
        input_image = input_image.unsqueeze(0)
        # 预测
        with torch.no_grad():
            # (batch_size,num,25)
            detections = model(input_image, training=False)
            # Non-Maximum Suppression 非极大抑制 shape（num,6）
            detections = non_max_suppression(detections, conf_thres, nms_thres)

        # 处理预测结果
        # 获取原始媒体的路径，原始cv2读取的图片，第几帧
        p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
        p = Path(p)
        save_path = os.path.join(output_dir, p.name)  # 保存的文件名称
        target_num = detections[0].shape[0]
        if target_num > 0:  # 判断是否有预测目标
            # 将预测的bbox 转换为原始图像尺寸的比例 img_size 为当前的尺寸  im0.shape[:2] 是原始的 （H,W）
            detections = rescale_boxes(detections[0], img_size, im0.shape[:2])
            # 使用标签工具
            annotator = Annotator(im0, line_width=2, pil=False)  # 传入的图片为cv2
            # 遍历这张图片的目标,并画目标框
            for *xyxy, conf, cls_pred in detections:
                color = colors(int(cls_pred), True)  # 获取颜色
                class_name = classes[int(cls_pred)]  # 获取标签名称 与 置信度 进行拼接
                label = class_name + " " + str(round(conf.numpy().tolist(), 2))
                annotator.box_label(xyxy, label, color)  # 画目标框和标签

        print(s, f"target {target_num}")
        # 保存结果
        if dataset.mode == 'image':
            cv2.imwrite(save_path, im0)  # 保存经过标记的图片

        # "video" or "stream"
        else:  # 视频文件
            if vid_path != save_path:  # 是否为新的检测视频
                vid_path = save_path  # 记录最新的视频输出路径
                if isinstance(vid_writer, cv2.VideoWriter):  # 如果有vid_writer实例先释放
                    vid_writer.release()  # release previous video writer
                if vid_cap:  # video
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path += '.mp4'

                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

            # 写入
            vid_writer.write(im0)


def _draw_and_save_output_images(img_detections, imgs, img_size, output_path, classes):
    for (image_path, detections) in zip(imgs, img_detections):
        print(f"Image {image_path}:")
        _draw_and_save_output_image(image_path, detections, img_size, output_path, classes)


def _draw_and_save_output_image(image_path, detections, img_size, output_path, classes):
    pil_img = Image.open(image_path)
    np_img = np.array(pil_img)
    default_fontsize = max(round(sum(pil_img.size) / 2 * 0.035), 12)
    font = check_font(size=default_fontsize)
    line_width = max(round(sum(np_img.shape) / 2 * 0.003), 2)  # 目标框的线条宽度
    draw = ImageDraw.Draw(pil_img)
    txt_color = (255, 255, 255)  # 默认是白色的字体
    img = np.array(Image.open(image_path))  # 打开原图像
    # Rescale boxes to original image
    detections = rescale_boxes(detections, img_size, img.shape[:2])

    for x1, y1, x2, y2, conf, cls_pred in detections:
        print(f"\t+ Label: {classes[int(cls_pred)]} | Confidence: {conf.item():0.4f}")
        color = colors(int(cls_pred))
        # 1.先画目标检测框
        # x1, y1, x2, y2 = x1.detach().numpy().tolist(), y1.detach().numpy().tolist(), x2.detach().numpy().tolist(), y2.detach().numpy().tolist()
        draw.rectangle([x1, y1, x2, y2], width=line_width, outline=color)
        # 2.画左上角标签 包括框框和text
        # 获取标签字体的的宽度
        class_name = classes[int(cls_pred)]
        label = class_name + " " + str(round(conf.numpy().tolist(), 2))
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
    parser = argparse.ArgumentParser(description="检测图片目标")

    parser.add_argument("-w", "--weight_path", type=str,
                        default="/Users/weimingan/work/weights/yolov3_voc_200_2021-12-25_08:46:33.pth",
                        help="权重文件")
    parser.add_argument("-i", "--img_dir", type=str, default="./data/sample", help="待检测图片存放路径")
    parser.add_argument("-o", "--output_path", type=str, default="./output", help="输出文件夹")
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("--img_size", type=int, default=416, help="输入Yolo的图片尺度")
    parser.add_argument("--n_cpu", type=int, default=1, help="dataloader的线程")
    parser.add_argument("--data", type=str, default="./data/voc2007.yaml", help="训练的数据集配置")
    parser.add_argument("--model", type=str, default="./models/yolov3.yaml", help="训练的模型配置")

    parser.add_argument("--conf_thres", type=float, default=0.2, help="有物体置信度阈值")
    parser.add_argument("--nms_thres", type=float, default=0.3, help="NMS阈值")

    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    # 数据集配置
    with open(args.data, errors='ignore') as f:
        data_params = yaml.safe_load(f)

    # 模型基本配置
    with open(args.model, errors='ignore') as f:
        model_params = yaml.safe_load(f)

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设置非训练模型加载
    model = YOLOV3(cls_num=data_params['num_classes'], anchors=model_params['anchors'], img_size=args.img_size).to(
        device)
    # 加载模型参数
    model.load_state_dict(torch.load(args.weight_path, map_location=device))

    # ============新版本============
    detect2(model, args.img_dir, args.output_path, args.img_size, args.conf_thres, args.nms_thres,
            data_params['classes_names'])


if __name__ == '__main__':
    run()
