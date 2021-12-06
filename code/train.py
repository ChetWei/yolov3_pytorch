# -*- coding: utf-8 -*-
import argparse

import torch
from utils.datasets import ListDataSet
from torch.utils.data import DataLoader
import torch.optim as optim
from Yolo3Body import YOLOV3
from utils.loss import compute_loss
from utils.util import weights_init, get_lr,save_model
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def create_dataloader(label_path, input_shape, batch_size, num_workers):
    dataset = ListDataSet(labels_file=label_path, input_shape=input_shape)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )
    return dataloader, len(dataset)


def train_one_epoch(model, dataloader, optimizer, device, epoch, epoches, epoch_steps, writer):
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
    with tqdm(total=epoch_steps, desc=f'Epoch {epoch + 1}/{epoches}', postfix=dict, mininterval=0.3) as pbar:
        for step, (imgs, targets) in enumerate(dataloader):
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device)
            # 梯度清零
            optimizer.zero_grad()
            # 向前传播
            outputs = model(imgs)
            # 计算损失
            loss, loss_components = compute_loss(outputs, targets, model)
            # 反向传播
            loss.backward()
            # 优化参数
            optimizer.step()

            total_loss += loss.item()
            # 计算当前epoch的平均损失
            mean_loss = total_loss / (step + 1)
            writer.add_scalar("loss/step", mean_loss, (step + 1) * (epoch + 1))
            writer.add_scalar("learning_rate/step", get_lr(optimizer), (step + 1) * (epoch + 1))
            info = "loss_v={},loss_mean={},lr={}".format(round(loss.item(), 6), round(mean_loss, 6), get_lr(optimizer))
            pbar.set_postfix_str(info)
            pbar.update(1)



def run():
    parser = argparse.ArgumentParser(description="Trains the YOLO model.")
    parser.add_argument('--cuda_id', type=int, default=0, help="使用的gpu")
    parser.add_argument('--model_name', type=str, default="yolov3", help="保存模型名称")
    parser.add_argument('--label_path', type=str, default="./voc2007_train.txt", help="设置label文件的路径")
    parser.add_argument('--num_workers', type=int, default=0, help="加载数据进程数量")
    parser.add_argument('--batch_size', type=int, default=1, help="batch的大小")
    parser.add_argument('--lr', type=float, default=0.0001, help="学习率")
    parser.add_argument('--input_shape', type=list, default=[416, 416], help="输入图片的尺寸 w h")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮次")
    parser.add_argument("--num_classes", type=int, default=20, help="训练数据集的类别个数")
    parser.add_argument("--weight_dir", type=str, default="./weights", help="模型权重保存目录")
    parser.add_argument("--logdir", type=str, default="./logs", help="tensorboard保存目录")
    parser.add_argument("--iou_thres", type=float, default=0.5,
                        help="Evaluation: IOU threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.1, help="Evaluation: Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5,
                        help="Evaluation: IOU threshold for non-maximum suppression")

    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    anchors_mask1 = [[116, 90], [156, 198], [373, 326]]  # 对应 13x13
    anchors_mask2 = [[30, 61], [62, 45], [59, 119]]  # 对应 26x26
    anchors_mask3 = [[10, 13], [16, 30], [33, 23]]  # 对应 52x52
    anchors_mask_list = [anchors_mask1, anchors_mask2, anchors_mask3]
    # tensorboard log
    writer = SummaryWriter(log_dir=args.logdir)

    # 选择使用的设备
    cuda_name = 'cuda:' + str(args.cuda_id)
    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = YOLOV3(args.num_classes, anchors_mask_list, img_size=args.input_shape[0]).to(device)
    # 权重参数固定初始化
    weights_init(model)
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    dataloader, epoch_steps = create_dataloader(args.label_path, args.input_shape, batch_size=args.batch_size,
                                                num_workers=args.num_workers)

    for epoch in range(args.epochs):
        train_one_epoch(model, dataloader, optimizer, device, epoch, args.epochs, epoch_steps, writer)

        if ((epoch + 1) % 100 == 0):
            save_model(model, args.model_name, (epoch + 1), weights_dir=args.weight_dir)


if __name__ == '__main__':
    run()
