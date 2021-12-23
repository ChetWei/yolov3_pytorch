# -*- coding: utf-8 -*-
import argparse

import tqdm

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import numpy as np

from terminaltables import AsciiTable

from yolov3_pytorch.models.Yolo3Body import YOLOV3
from utils.util import get_classes_name, xywh2xyxy, non_max_suppression, get_batch_statistics, ap_per_class
from config import anchors_mask_list
from utils.datasets import ListDataSet
from utils.transforms import DEFAULT_TRANSFORMS


def _create_validation_data_loader(label_path, input_shape, batch_size, num_workers):
    dataset = ListDataSet(labels_file=label_path, input_shape=input_shape, transform=DEFAULT_TRANSFORMS)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )

    return dataloader,len(dataset)


def print_eval_stats(metrics_output, class_names, verbose):
    if metrics_output is not None:
        precision, recall, AP, f1, ap_class = metrics_output
        if verbose:
            # Prints class AP and mean AP
            ap_table = [["Index", "Class", "AP"]]
            for i, c in enumerate(ap_class):
                ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
            print(AsciiTable(ap_table).table)
        print(f"---- mAP {AP.mean():.5f} ----")
    else:
        print("---- mAP not measured (no detections found by model) ----")


def _evaluate(model, dataloader, class_names, img_size, iou_thres, conf_thres, nms_thres, verbose):
    """Evaluate model on validation dataset.

    :param model: Model to evaluate
    :type model: models.Darknet
    :param dataloader: Dataloader provides the batches of images with targets
    :type dataloader: DataLoader
    :param class_names: List of class names
    :type class_names: [str]
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param iou_thres: IOU threshold required to qualify as detected
    :type iou_thres: float
    :param conf_thres: Object confidence threshold
    :type conf_thres: float
    :param nms_thres: IOU threshold for non-maximum suppression
    :type nms_thres: float
    :param verbose: If True, prints stats of model
    :type verbose: bool
    :return: Returns precision, recall, AP, f1, ap_class
    """
    model.eval()  # Set model to evaluation mode

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)


    for imgs, targets in tqdm.tqdm(dataloader, desc="Validating"):
        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, iou_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    if len(sample_metrics) == 0:  # No detections over whole validation set.
        print("---- No detections over whole validation set ----")
        return None

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [
        np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    metrics_output = ap_per_class(
        true_positives, pred_scores, pred_labels, labels)

    print_eval_stats(metrics_output, class_names, verbose)

    return metrics_output


def run():
    parser = argparse.ArgumentParser(description="Evaluate validation data.")

    parser.add_argument("-w", "--weight_path", type=str, default="/Users/weimingan/work/weights/yolov3_vocc_50 (1).pth",
                        help="权重文件")
    parser.add_argument("-c", "--classes", type=str, default="../config/voc_names.txt", help="类别文件")
    parser.add_argument("--label_path", type=str, default="../data/annotation/voc2007_test.txt", help="标签文件")
    parser.add_argument("-b", "--batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=416, help="输入Yolo的图片尺度")
    parser.add_argument('--input_shape', type=list, default=[416, 416], help="输入图片的尺寸 w h")
    parser.add_argument("--num_workers", type=int, default=1, help="dataloader的线程")

    parser.add_argument("--iou_thres", type=float, default=0.5, help="IOU threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.01, help="Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="IOU threshold for non-maximum suppression")
    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    # 获取制作数据集时候设置的类别列表
    classes = get_classes_name(args.classes)

    # 加载测试数据集
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 设置非训练模型加载
    model = YOLOV3(cls_num=len(classes), anchors=anchors_mask_list, img_size=args.img_size, training=False).to(device)
    # 加载模型参数
    model.load_state_dict(torch.load(args.weight_path, map_location=device))
    # 创建dataloader
    dataloader,dataset_len = _create_validation_data_loader(args.label_path, args.input_shape, args.batch_size, args.num_workers)
    # 开始检测和评估
    metrics_output = _evaluate(
        model,
        dataloader,
        classes,
        args.img_size,
        args.iou_thres,
        args.conf_thres,
        args.nms_thres,
        verbose=True)

    precision, recall, AP, f1, ap_class = metrics_output


if __name__ == '__main__':
    run()
