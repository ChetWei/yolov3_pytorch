# -*- coding: utf-8 -*-

import os
import xml.etree.ElementTree as ET


def get_classes_name(path):
    with open(path, "r") as f:
        names = [line.strip() for line in f.readlines()]
    return names


def convert_voc_annotaion(segment_file, classname_file, target_file):
    class_names = get_classes_name(classname_file)
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
            image_ab_path = voc_img_home + "/" + image_name
            label_line.append(image_ab_path)
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
    with open(target_file, 'w') as f:
        for label_line in label_lines:
            l = " ".join(label_line)
            f.write(l + "\n")


if __name__ == '__main__':

    voc_names = "../../config/voc_names.txt"
    VOCdevkt_path = "/Users/weimingan/work/dataset/VOCdevkit"  #需要修改的
    voc_annotations_path = VOCdevkt_path + "/VOC2007/Annotations"
    voc_img_home = VOCdevkt_path + "/VOC2007/JPEGImages"
    segment_train_file = VOCdevkt_path + "/VOC2007/ImageSets/Main/train.txt"
    segment_val_file = VOCdevkt_path + "/VOC2007/ImageSets/Main/val.txt"

    convert_voc_annotaion(
        segment_file=segment_train_file,
        classname_file=voc_names,
        target_file='../voc2007_train.txt')
