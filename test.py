# -*- coding: utf-8 -*-
import argparse

import tqdm

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

import numpy as np
import yaml

from terminaltables import AsciiTable

from models.Yolo3Body import YOLOV3
from utils.util import xywh2xyxy, non_max_suppression, get_batch_statistics, ap_per_class
from utils.datasets import ListDataSet
from utils.transforms import DEFAULT_TRANSFORMS


# 创建验证数据集
def _create_validation_data_loader(label_path, img_home, input_shape, batch_size, num_workers):
    dataset = ListDataSet(label_path, img_home, input_shape, transform=DEFAULT_TRANSFORMS)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )

    return dataloader, len(dataset)


# 输出评估的结果状态
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


def _evaluate(model, dataloader, class_names, img_size, iou_thres, conf_thres, nms_thres, verbose, device):
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

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)

    for imgs, targets in tqdm.tqdm(dataloader, desc="Validating"):
        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.to(device, non_blocking=True), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs, training=False)
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

    parser.add_argument('--cuda_id', type=int, default=0, help="使用的gpu")
    parser.add_argument("-w", "--weight_path", type=str, default="/Users/weimingan/work/weights/yolov3_voc_1100.pth",
                        help="权重文件")
    parser.add_argument("-b", "--batch_size", type=int, default=8)
    parser.add_argument("--img_size", type=int, default=416, help="输入Yolo的图片尺度")
    parser.add_argument('--input_shape', type=list, default=[416, 416], help="输入图片的尺寸 w h")
    parser.add_argument("--num_workers", type=int, default=2, help="dataloader的线程")

    parser.add_argument("--data", type=str, default="./data/voc2007.yaml", help="训练的数据集配置")
    parser.add_argument("--model", type=str, default="./models/yolov3.yaml", help="训练的模型配置")

    parser.add_argument("--iou_thres", type=float, default=0.5, help="IOU threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.01, help="Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="IOU threshold for non-maximum suppression")
    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    # 选择使用的设备
    cuda_name = 'cuda:' + str(args.cuda_id)
    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')

    # 数据集配置
    with open(args.data, errors='ignore') as f:
        data_params = yaml.safe_load(f)

    # 模型基本配置
    with open(args.model, errors='ignore') as f:
        model_params = yaml.safe_load(f)

    # 加载测试数据集
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 设置非训练模型加载
    model = YOLOV3(cls_num=data_params['num_classes'], anchors=model_params['anchors'], img_size=args.img_size,
                   training=False).to(device)
    # 加载模型参数
    model.load_state_dict(torch.load(args.weight_path, map_location=device))
    # 创建dataloader
    dataloader, dataset_len = _create_validation_data_loader(
        label_path=data_params['targets'][1],  # 验证数据集
        img_home=data_params['img_home'],
        input_shape=args.input_shape,
        batch_size=args.batch_size,
        num_workers=args.num_workers)
    # 开始检测和评估
    metrics_output = _evaluate(
        model=model,
        dataloader=dataloader,
        class_names=data_params['classes_names'],
        img_size=args.img_size,
        iou_thres=args.iou_thres,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres,
        verbose=True,
        device=device)

    precision, recall, AP, f1, ap_class = metrics_output


if __name__ == '__main__':
    run()
