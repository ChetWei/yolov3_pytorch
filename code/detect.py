# -*- coding: utf-8 -*-
import argparse
import os
import tqdm
import random
import numpy as np

from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

from utils.util import get_classes_name, non_max_suppression, rescale_boxes
from utils.datasets import ImageFolder
from utils.transforms import Resize, DEFAULT_TRANSFORMS
from Yolo3Body import YOLOV3

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator


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
    os.makedirs(output_path, exist_ok=True)
    model.eval()
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    img_detections = []  # Stores detections for each image index
    imgs = []  # Stores image paths

    for (img_paths, input_imgs) in tqdm.tqdm(dataloader, desc="Detecting"):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # 预测
        with torch.no_grad():
            detections = model(input_imgs)
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


def _draw_and_save_output_image(image_path, detections, img_size, output_path, classes):
    img = np.array(Image.open(image_path))
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    # Rescale boxes to original image
    detections = rescale_boxes(detections, img_size, img.shape[:2])
    unique_labels = detections[:, -1].cpu().unique()
    n_cls_preds = len(unique_labels)
    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, n_cls_preds)]
    bbox_colors = random.sample(colors, n_cls_preds)
    for x1, y1, x2, y2, conf, cls_pred in detections:
        print(f"\t+ Label: {classes[int(cls_pred)]} | Confidence: {conf.item():0.4f}")

        box_w = x2 - x1
        box_h = y2 - y1

        color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
        # Create a Rectangle patch
        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
        # Add the bbox to the plot
        ax.add_patch(bbox)
        # Add label
        plt.text(
            x1,y1,
            s=classes[int(cls_pred)],
            color="white",
            verticalalignment="top",
            bbox={"color": color, "pad": 0})

    # Save generated image with detections
    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    filename = os.path.basename(image_path).split(".")[0]
    output_path = os.path.join(output_path, f"{filename}.png")
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
    plt.close()


def run():
    from config import anchors_mask_list

    parser = argparse.ArgumentParser(description="检测图片目标")

    parser.add_argument("-w", "--weight_path", type=str, default="/Users/weimingan/work/weights/yolov3_voc_300.pth", help="权重文件")
    parser.add_argument("-i", "--img_dir", type=str, default="../data/sample", help="待检测图片存放路径")
    parser.add_argument("-c", "--classes", type=str, default="../config/voc_names.txt", help="类别文件")
    parser.add_argument("-o", "--output_path", type=str, default="../output", help="输出文件夹")
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("--img_size", type=int, default=416, help="输入Yolo的图片尺度")
    parser.add_argument("--n_cpu", type=int, default=1, help="dataloader的线程")
    parser.add_argument("--conf_thres", type=float, default=0.1, help="有物体置信度阈值")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="NMS阈值")

    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    # 获取制作数据集时候设置的类别列表
    classes = get_classes_name(args.classes)

    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = YOLOV3(cls_num=len(classes), anchors=anchors_mask_list, img_size=args.img_size, training=False).to(device)

    model.load_state_dict(torch.load(args.weight_path, map_location=device))

    dataloader = _create_data_loader(args.img_dir, args.batch_size, args.img_size, args.n_cpu)

    # 检测图片
    img_detections, imgs = detect(model, dataloader, args.output_path, args.img_size, args.conf_thres, args.nms_thres)

    # 画图并输出图片
    _draw_and_save_output_images(img_detections, imgs, args.img_size, args.output_path, classes)

if __name__ == '__main__':
    run()
