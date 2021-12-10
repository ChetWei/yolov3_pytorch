# -*- coding: utf-8 -*-
import argparse
import torch

from tqdm import tqdm
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from Yolo3Body import YOLOV3
from config import anchors_mask_list
from utils.datasets import ListDataSet
from utils.loss import compute_loss
from utils.util import weights_init, get_lr, save_model
from utils.augmentations import AUGMENTATION_TRANSFORMS


def create_dataloader(label_path, input_shape, batch_size, num_workers):
    dataset = ListDataSet(labels_file=label_path, input_shape=input_shape, transform=AUGMENTATION_TRANSFORMS)
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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Trains the YOLO3 model.")
    parser.add_argument('--cuda_id', type=int, default=0, help="使用的gpu")
    parser.add_argument('--pre_trained', type=str2bool, default=False, help="使用预训练模型")
    parser.add_argument("--weight_path", type=str, default="../weights/yolov3_coco_20.pth", help="预训练权重路径")
    parser.add_argument("--weight_dir", type=str, default="../weights", help="模型权重保存目录")
    parser.add_argument('--model_name', type=str, default="yolov3_voc", help="保存模型名称")
    parser.add_argument("--save_per_epoch", type=int, default=100, help="每多少轮保存一次权重")
    parser.add_argument("--logdir", type=str, default="./logs_voc", help="tensorboard保存目录")
    parser.add_argument('--label_path', type=str, default="../data/annotation/voc2007_train.txt", help="设置label文件的路径")
    parser.add_argument("--num_classes", type=int, default=20, help="训练数据集的类别个数")

    parser.add_argument('--freeze', type=str2bool, default=False, help="是否冻结骨干网络")

    parser.add_argument('--batch_size', type=int, default=16, help="batch的大小")
    parser.add_argument('--lr', type=float, default=0.0001, help="学习率")
    parser.add_argument('--decay', type=float, default=0.0005, help="decay")
    parser.add_argument('--input_shape', type=list, default=[416, 416], help="输入图片的尺寸 w h")
    parser.add_argument("--epochs", type=int, default=1000, help="训练轮次")
    parser.add_argument('--num_workers', type=int, default=4, help="加载数据进程数量")

    #========检测时候使用==========
    parser.add_argument("--iou_thres", type=float, default=0.5,
                        help="Evaluation: IOU threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.1, help="Evaluation: Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5,
                        help="Evaluation: IOU threshold for non-maximum suppression")

    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    # tensorboard log
    writer = SummaryWriter(log_dir=args.logdir)

    # 选择使用的设备
    cuda_name = 'cuda:' + str(args.cuda_id)
    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = YOLOV3(args.num_classes, anchors_mask_list, img_size=args.input_shape[0]).to(device)
    # 权重参数固定初始化
    if args.pre_trained:
        model.load_state_dict(torch.load(args.weight_path, map_location=device))
    else:
        weights_init(model)

    # 是否冻结骨干网络
    for param in model.backbone.parameters():
        param.requires_grad = not args.freeze

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    dataloader, epoch_steps = create_dataloader(args.label_path, args.input_shape, batch_size=args.batch_size,
                                                num_workers=args.num_workers)

    for epoch in range(args.epochs):
        train_one_epoch(model, dataloader, optimizer, device, epoch, args.epochs, epoch_steps // args.batch_size,
                        writer)

        if ((epoch + 1) % args.save_per_epoch == 0):
            save_model(model, args.model_name, (epoch + 1), weights_dir=args.weight_dir)
