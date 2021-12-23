# -*- coding: utf-8 -*-
import json
import os
import xml.etree.ElementTree as ET

import yaml

from util import get_classes_name

project_name = 'yolov3_pytorch'
root_path = os.path.abspath(os.path.dirname(__file__)).split(project_name)[0]
project_path = os.path.join(root_path, project_name)

"""
根据提供的标签文件转换 为自己的格式
图片名称 cls_id,x1,y1,x2,y2 cls_id,x1,y1,x2,y2 cls_id,x1,y1,x2,y2

"""


def convert_voc_annotaion(voc_annotations_path, class_names, segment_files, target_files):
    """
    :param voc_annotations_path: 存放原始标签目录
    :param class_names: 类别列表
    :param segment_files: 划分数据集的文件 列表 tain val test
    :param target_files: 保存的目标标签 列表 tain val test
    :return: 制作三个类别的标签  tain val test
    """
    for idx, segment_file in enumerate(segment_files):
        with open(segment_file, 'r') as f:
            file_ids = [line.strip() for line in f.readlines()]
        # 读取xml文件
        label_lines = []
        for file_id in file_ids:
            label_line = []
            label_file = file_id + ".xml"
            # 读取标签文件
            with open(os.path.join(voc_annotations_path, label_file), 'r') as f:
                tree = ET.parse(f)
                root = tree.getroot()
                image_name = root.find("filename").text
                label_line.append(image_name)
                # 遍历图片中的对象 获得类别 位置坐标信息
                for obj in root.iter("object"):
                    difficult = obj.find("difficult").text
                    class_name = obj.find("name").text
                    if class_name not in class_names:
                        continue
                    class_id = class_names.index(class_name)  # 从0开始编号
                    xmlbox = obj.find('bndbox')
                    xmin = xmlbox.find('xmin').text
                    ymin = xmlbox.find('ymin').text
                    xmax = xmlbox.find('xmax').text
                    ymax = xmlbox.find('ymax').text
                    box = str(class_id) + "," + xmin + "," + ymin + "," + xmax + "," + ymax
                    label_line.append(box)
            label_lines.append(label_line)

        # 按行写入文件
        target_file = os.path.join(project_path,target_files[idx])
        with open(target_file, 'w') as f:
            for label_line in label_lines:
                l = " ".join(label_line)
                f.write(l + "\n")


# 读取本地的标签文件转换为自己的标签格式
def convert_coco_annotation(label_file, img_home, save_path="../../data/annotation/coco2017_train.txt",
                            cls_file="../../config/coco_names.txt"):
    # 获取类别
    coco_cls_names = get_classes_name(cls_file)
    # coco 的目标检测格式是xywh
    with open(label_file, "r") as f:
        coco_json = json.load(f)
        categories = coco_json.get("categories")
        images = coco_json.get("images")
        annotations = coco_json.get("annotations")

        category_dict = {}

        for c in categories:
            cls_name = c.get("name")
            id = c.get("id")
            category_dict[id] = cls_name

        img_dict = {}
        for img in images:
            img_id = img.get("id")
            img_dict[img_id] = []
            file_name = img.get("file_name")
            # 拼接完整的文件路径
            img_path = os.path.join(img_home, file_name)
            img_dict[img_id].append(img_path)

        for annotation in annotations:
            image_id = annotation.get("image_id")
            category_id = annotation.get("category_id")
            bbox = annotation.get("bbox")  # xywh
            # 从dict中获取类别名称
            category_name = category_dict.get(category_id)
            # 根据类别名称，从类别文件中获取下标
            cidx = coco_cls_names.index(category_name)

            if (len(bbox)) == 4:
                box_str = ",".join(str(i) for i in bbox)
                box = str(cidx) + "," + box_str

                img_dict.get(image_id).append(box)
    # 写入文件
    with open(save_path, "a") as f:
        for k, v in img_dict.items():
            if len(v) > 1:
                line = " ".join(v)
                f.write(line + "\n")


# 读取本地的标签文件转换为自己的标签格式
def run_voc_convert():
    # 配置远程服务器图片文件夹路径
    voc_img_home = "/media/root/5A72B65672B6371B/weimingan/dataset/VOCdevkit/VOC2007/JPEGImages"
    # voc_img_home = "/Users/weimingan/work/dataset/VOCdevkit/VOC2007/JPEGImages"
    # voc_img_home = "/root/autodl-tmp/VOC2007/JPEGImages"
    # 本地的路径
    voc_annotations_path = "/Users/weimingan/work/dataset/VOCdevkit/VOC2007/Annotations"
    segment_file = "/Users/weimingan/work/dataset/VOCdevkit/VOC2007/ImageSets/Main/train.txt"
    # segment_file = "/Users/weimingan/work/dataset/VOCdevkit/VOC2007/ImageSets/Main/val.txt"
    # segment_file = "/Users/weimingan/work/dataset/VOCdevkit/VOC2007/ImageSets/Main/test.txt"

    convert_voc_annotaion(voc_annotations_path, voc_img_home, segment_file)


def run_coco_convert():
    # 远程图片文件夹目录
    img_home = "/root/autodl-tmp/train2017"
    # 本地的标签目录
    label_file = "/Users/weimingan/work/dataset/coco2017_annotations/instances_train2017.json"

    convert_coco_annotation(label_file=label_file, img_home=img_home)


def parse_yaml(path):
    with open(path, errors='ignore') as f:
        # 参数字典
        args = yaml.safe_load(f)
    return args


if __name__ == '__main__':
    data_path = 'data/voc2007.yaml'
    args = parse_yaml(os.path.join(project_path, data_path))

    convert_voc_annotaion(
        voc_annotations_path=args['annotations_path'],
        class_names=args['classes_names'],
        segment_files=args['segments'],
        target_files=args['targets'])
