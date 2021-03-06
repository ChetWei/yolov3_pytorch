# -*- coding: utf-8 -*-
import argparse
import math

import torch
import yaml
import time

from tqdm import tqdm
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from utils.datasets import ListDataSet
from utils.loss import compute_loss
from utils.util import weights_init, get_lr, save_model,str2bool,update_lr
from utils.augmentations import AUGMENTATION_TRANSFORMS

from models.Yolo3Body import YOLOV3
from test import _create_validation_data_loader, _evaluate


def create_dataloader(label_path, img_home, input_shape, batch_size, num_workers):
    """
    :param label_path: 待训练的标签文件路径
    :param img_home: 训练的图片home路径
    :param input_shape: 输入模型的图片尺寸
    :param batch_size:
    :param num_workers:
    :return:
    """

    dataset = ListDataSet(label_path, img_home, input_shape, transform=AUGMENTATION_TRANSFORMS)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )
    return dataloader, len(dataset)



def train_one_epoch(model, dataloader, optimizer, device, epoch, epoches, epoch_steps, warmup_steps, hyp_params,
                    writer):
    """
    训练一个世代
    :param model: 训练的模型
    :param dataloader: 数据加载器
    :param optimizer: 优化器
    :param device: 设备
    :param epoch: 当前训练的epoch
    :param epoches: 累计训练多少个epoch
    :param epoch_steps:  每个epoch有多少个batch步
    :param loss : 损失函数
    :param writer:  tensorboard写入器
    :return:
    """
    model.train()
    total_loss = 0
    box_loss = 0
    obj_loss = 0
    cls_loss = 0

    # 当训练轮数快结束了，开始降低学习率
    if (epoch + 1) / epoches > hyp_params['decay_pos']:
        new_lr = hyp_params['lr'] * math.pow(hyp_params['lr_decay'],
                                             int((epoch + 1) - (epoches * hyp_params['decay_pos'])))
        update_lr(optimizer, new_lr)

    with tqdm(total=epoch_steps, postfix=dict, mininterval=0.3) as pbar:
        for step, (imgs, targets) in enumerate(dataloader):

            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device)
            # 梯度清零
            optimizer.zero_grad()
            currnt_step = (epoch_steps * epoch) + (step + 1)  # 所有轮次中当前的step

            # warmup 阶段的学习率
            if warmup_steps > currnt_step:
                #  更新学习率
                new_lr = hyp_params['lr'] * (currnt_step / warmup_steps)
                update_lr(optimizer, new_lr)

            # 向前传播
            outputs = model(imgs)
            # 计算损失
            loss, loss_components = compute_loss(outputs, targets, model, hyp_params)
            # 反向传播
            loss.backward()
            # 优化参数
            optimizer.step()

            total_loss += loss.item()
            box_loss += loss_components[0]
            obj_loss += loss_components[1]
            cls_loss += loss_components[2]

            # 计算当前epoch的平均损失
            mean_loss = total_loss / (step + 1)
            mean_box = box_loss / (step + 1)
            mean_obj = obj_loss / (step + 1)
            mean_cls = cls_loss / (step + 1)

            writer.add_scalar("box_mean/epoch", mean_box, currnt_step)
            writer.add_scalar("obj_mean/epoch", mean_obj, currnt_step)
            writer.add_scalar("cls_mean/epoch", mean_cls, currnt_step)
            writer.add_scalar("mean_loss/epoch", mean_loss, currnt_step)
            writer.add_scalar("learning_rate/step", get_lr(optimizer), currnt_step)

            # 控制台日志
            c_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            info_desc = f"[{c_time}] Epoch {epoch + 1}/{epoches}"

            info_postfix = f'loss_v:{loss.item():.6f}|' \
                           f'box_mean:{mean_box:.6f}|' \
                           f'obj_mean:{mean_obj:.6f}|' \
                           f'cls_mean:{mean_cls:.6f}|' \
                           f'mean_loss:{mean_loss:.6f}|' \
                           f'lr:{get_lr(optimizer)}'

            pbar.set_description_str(info_desc)
            pbar.set_postfix_str(info_postfix)

            pbar.update(1)





def parse_args():
    parser = argparse.ArgumentParser(description="Trains the YOLO3 model.")

    parser.add_argument('--cuda_id', type=int, default=1, help="使用的gpu")
    parser.add_argument('--input_shape', type=list, default=[416, 416], help="输入图片的尺寸 w h")
    parser.add_argument("--img_size", type=int, default=416, help="输入Yolo的图片尺度")
    parser.add_argument("--epochs", type=int, default=1200, help="训练的轮次")
    parser.add_argument("--save_interval", type=int, default=50, help="每多少轮保存一次权重")
    parser.add_argument('--batch_size', type=int, default=4, help="batch的大小")
    parser.add_argument('--num_workers', type=int, default=3, help="加载数据进程数量")

    parser.add_argument("--evaluation_interval", type=int, default=100, help="每多少轮验证一次")

    parser.add_argument('--adam', type=str2bool, default=True, help="使用Adam or SGD")
    parser.add_argument('--pre_trained', type=str2bool, default=False, help="使用预训练模型")
    parser.add_argument("--weight_path", type=str, default="./weights/yolov3_voc_80_2021-12-24_23:15:02.pth",
                        help="预训练权重路径")

    parser.add_argument("--hyp", type=str, default="./config/hyps/hyp.yaml", help="超参数配置文件")
    parser.add_argument("--data", type=str, default="./data/voc2007.yaml", help="训练的数据集配置")
    parser.add_argument("--model", type=str, default="./models/yolov3.yaml", help="训练的模型配置")
    parser.add_argument("--weight_dir", type=str, default="./weights", help="模型权重保存目录")
    parser.add_argument("--logdir", type=str, default="./data/logs", help="tensorboard保存目录")
    parser.add_argument('--model_name', type=str, default="yolov3_voc", help="保存模型名称")

    parser.add_argument("--iou_thres", type=float, default=0.5, help="IOU threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.01, help="Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="IOU threshold for non-maximum suppression")

    args = parser.parse_args()
    print(f"Command line arguments: {args}")
    return args


def load_params(args):
    # 加载超参数
    with open(args.hyp, errors='ignore') as f:
        hyp_params = yaml.safe_load(f)

    # 数据集配置
    with open(args.data, errors='ignore') as f:
        data_params = yaml.safe_load(f)

    # 模型基本配置
    with open(args.model, errors='ignore') as f:
        model_params = yaml.safe_load(f)

    return hyp_params, data_params, model_params


def main(args):
    hyp_params, data_params, model_params = load_params(args)

    # tensorboard 日志
    writer = SummaryWriter(log_dir=args.logdir)

    # 选择使用的设备
    cuda_name = 'cuda:' + str(args.cuda_id)
    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = YOLOV3(
        cls_num=data_params['num_classes'],
        anchors=model_params['anchors'],
        img_size=args.input_shape[0]).to(device)  # 训练的图片尺寸大小

    # 权重参数初始化
    if args.pre_trained:
        model.load_state_dict(torch.load(args.weight_path, map_location=device))

    # 优化器
    if args.adam:
        optimizer = optim.Adam(
            model.parameters(),
            lr=hyp_params['lr'],
            weight_decay=hyp_params['weight_decay'])
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=hyp_params['lr'],
            momentum=hyp_params['momentum'],
            weight_decay=hyp_params['weight_decay'],
            nesterov=True)

    # 加载数据 dataset_len 数据集的数量
    train_dataloader, train_len = create_dataloader(
        label_path=data_params['targets'][0],  # 训练的标签文件路径
        img_home=data_params['img_home'],  # 训练的图片home目录
        input_shape=args.input_shape,  # 输入尺寸
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    # 创建验证dataloader
    val_dataloader, val_len = _create_validation_data_loader(
        label_path=data_params['targets'][1],  # 验证数据集
        img_home=data_params['img_home'],
        input_shape=args.input_shape,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    epoch_steps = len(train_dataloader)  # 每轮有多少个step
    warmup_steps = hyp_params['warmup_epochs'] * epoch_steps  # warmup step 的数量
    for epoch in range(args.epochs):
        # 训练一个epoch
        train_one_epoch(model, train_dataloader, optimizer, device, epoch, args.epochs, epoch_steps, warmup_steps,
                        hyp_params, writer)

        # 每save_per_epoch轮就保存一次权重
        if ((epoch + 1) % args.save_interval == 0):
            save_model(model, args.model_name, (epoch + 1), weights_dir=args.weight_dir)
            # optimizer.state_dict() 保存训练器状态，方便下一次训练

        # 验证模型
        if (epoch + 1) % args.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            metrics_output = _evaluate(
                model=model,
                dataloader=val_dataloader,  # 使用验证数据集
                class_names=data_params['classes_names'],
                img_size=args.img_size,
                iou_thres=args.iou_thres,
                conf_thres=args.conf_thres,
                nms_thres=args.nms_thres,
                verbose=True,
                device=device)


if __name__ == '__main__':
    # 获得代码参数
    args = parse_args()

    # 运行主程序
    main(args)
