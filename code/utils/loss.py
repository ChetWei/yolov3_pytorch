# -*- coding: utf-8 -*-
import math

import torch
import torch.nn as nn
from .util import to_cpu


def compute_loss(predictions, targets, model):
    """
    :param predictions: 三个尺度的输出 (b,3,13,13,85) ,(b,3,26,26,85),(b,3,52,52,85)
    :param targets: 真实框
    :param model:
    :return:
    """
    device = targets.device
    # 存储损失
    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
    # 获得标签的 target
    # Build yolo targets
    tcls, tbox, indices, anchors = build_targets(predictions, targets, model)  # targets

    # 这里的bce包含了 sigmoid + BCE 两个过程
    BCEcls = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([1.0], device=device))
    BCEobj = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([1.0], device=device))

    # 计算每个特征层的输出损失
    for layer_index, layer_predictions in enumerate(predictions):
        # Get image ids, anchors, grid index i and j for each target in the current yolo layer
        b, anchor, grid_j, grid_i = indices[layer_index]
        # 创建有目标的标记初始为0 [1,3,13,13]
        tobj = torch.zeros_like(layer_predictions[..., 0], device=device)  # target obj
        # Get the number of targets for this layer.
        # Each target is a label box with some scaling and the association of an anchor box.
        # Label boxes may be associated to 0 or multiple anchors. So they are multiple times or not at all in the targets.
        num_targets = b.shape[0]
        # Check if there are targets for this batch
        if num_targets:
            # (num,85)
            ps = layer_predictions[b, anchor, grid_j, grid_i]

            # 有目标存在的 box 回归
            # xy偏移 sigmoid 保持在(0,1)之间  shape(num,2)
            pxy = ps[:, :2].sigmoid()
            # Apply exponent to wh predictions and multiply with the anchor box that matched best with the label for each cell that has a target
            # ahchors[layer_index] 之前gettarge得到的当前layer的先验框
            # 预测的比例 * 之前选中的负责预测的先验框尺寸 #shape（num,2）
            pwh = torch.exp(ps[:, 2:4]) * anchors[layer_index]
            # (num,4)
            pbox = torch.cat((pxy, pwh), 1)
            # Calculate CIoU or GIoU for each target with the predicted box for its cell + anchor
            iou = bbox_iou(pbox.T, tbox[layer_index], x1y1x2y2=False, CIoU=True)
            # We want to minimize our loss so we and the best possible IoU is 1 so we take 1 - IoU and reduce it with a mean
            # 得到的iou是一个 长度为 num 的tensor ，1-iou  再求平均
            lbox += (1.0 - iou).mean()  # iou loss

            # 目标分类损失
            # 设置单元格有物体的概率
            # 将之前有目标点的真实框 与预测框(根据anchor的)之间的iou作为 有目标的概率 ？？？ 为什么不直接用1
            tobj[b, anchor, grid_j, grid_i] = iou.detach().clamp(0).type(
                tobj.dtype)  # Use cells with iou > 0 as object targets

            # Check if we need to do a classification (number of classes > 1)
            if ps.size(1) - 5 > 1:  # 判断类别个数是否大于1
                # Hot one class encoding 用 0 1 编码标记target
                # (num,classnum)
                t = torch.zeros_like(ps[:, 5:], device=device)  # targets
                t[range(num_targets), tcls[layer_index]] = 1
                # Use the tensor to calculate the BCE loss
                lcls += BCEcls(ps[:, 5:], t)  # BCE

        # 置信度损失，正负样本一起计算
        lobj += BCEobj(layer_predictions[..., 4], tobj)  # obj loss

    lbox *= 0.05
    lobj *= 1.0  # 是否有物体非常重要
    lcls *= 0.5

    # Merge losses
    loss = lbox + lobj + lcls

    return loss, to_cpu(torch.cat((lbox, lobj, lcls, loss)))


def build_targets(p, targets, model):
    """
    :param p:
    :param targets: # (batchid,classid,x,y,w,h)  batch为这个数据在这个batch的的序号
    :param model:
    :return:
    """
    na, nt = 3, targets.shape[0]  # anchors数量, targets 真实目标数量
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
    # [0,1,2] 来标记三个anchor
    ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)
    # 如果有1个真实框tensor([[0.],[1.],[2.]])
    # 如果有2个真实框tensor([[0., 0.],[1., 1.],[2., 2.]])

    # 将真实框复制3份，并拓展最后一个维度为对应的anchor index 分别用来对应三个anchor的计算
    # (img id, class, x, y, w, h, anchor id)
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)
    # 一个真实目标框 tensor([[[0.0000, 0.0000, 0.5422, 0.5520, 0.1203, 0.1513, 0.0000]],
    # [[0.0000, 0.0000, 0.5422, 0.5520, 0.1203, 0.1513, 1.0000]],
    # [[0.0000, 0.0000, 0.5422, 0.5520, 0.1203, 0.1513, 2.0000]]])
    # 两个真实目标框tensor([[[0.0000, 0.0000, 0.4662, 0.5317, 0.1663, 0.2092, 0.0000],
    #  [0.0000, 2.0000, 0.3129, 0.6461, 0.2430, 0.2092, 0.0000]],
    # [[0.0000, 0.0000, 0.4662, 0.5317, 0.1663, 0.2092, 1.0000],
    #  [0.0000, 2.0000, 0.3129, 0.6461, 0.2430, 0.2092, 1.0000]],
    # [[0.0000, 0.0000, 0.4662, 0.5317, 0.1663, 0.2092, 2.0000],
    #  [0.0000, 2.0000, 0.3129, 0.6461, 0.2430, 0.2092, 2.0000]]])
    for i, yolo_layer in enumerate(model.yolo_out_layer):
        anchors = yolo_layer.anchors.to(targets.device)
        anchors = anchors / yolo_layer.stride  # anchor以单元格为长度单元
        # 设置 feature_w,feature_h,feature_w,feature_h
        gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain[2:6]  tensor([13, 13, 13, 13])
        # 计算坐标在特征图上的位置,其他属性乘以1了，不会有影响
        t = targets * gain  # xywh * feature_w,feature_h,feature_w,feature_h
        # 如果有真实框
        if nt:
            # 计算真实框宽高与anchor宽高的比例
            r = t[:, :, 4:6] / anchors[:, None]  # w,h
            # 选择比例比较合适的anchor来负责预测， 用 max(r,1./r) 是因为不知道真实框和anchor谁大谁小
            j = torch.max(r, 1. / r).max(2)[0] < 4  # compare
            # 这意味着我们只保留那些比较合适的先验框负责预测，并且放弃了其他的先验框
            t = t[j]
            # 筛选后的t  (num,7)
            # tensor([[0.0000, 0.0000, 6.6443, 6.5083, 2.1944, 2.7607, 0.0000],
            #         [0.0000, 2.0000, 8.6672, 8.0180, 3.2058, 2.7607, 0.0000],
            #         [0.0000, 0.0000, 6.6443, 6.5083, 2.1944, 2.7607, 1.0000],
            #         [0.0000, 2.0000, 8.6672, 8.0180, 3.2058, 2.7607, 1.0000],
            #         [0.0000, 2.0000, 8.6672, 8.0180, 3.2058, 2.7607, 2.0000]])
        else:
            t = targets[0]

        # 抽取 image id in batch 和 class id  长度是t.num
        b, c = t[:, :2].long().T
        # We isolate the target cell associations.
        # x, y, w, h are allready in the cell coordinate system meaning an x = 1.2 would be 1.2 times cellwidth
        gxy = t[:, 2:4]
        gwh = t[:, 4:6]  # grid wh
        # 将以单元格为长度单位的 xy值取整，就可以获得所在的单元格位置
        gij = gxy.long()
        # 分离 i j
        gi, gj = gij.T  # grid xy indices

        # 将anchor的标记也转为int
        a = t[:, 6].long()
        #  clamp 以防超出单元格界限
        indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))
        # gxy-gij 获得相对单元格的偏移量 与 wh拼接 组成box的位置信息 添加到list中
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
        # 添加负责预测每个目标的先验框 到list中
        anch.append(anchors[a])
        # 添加每个目标的类别id 到list中
        tcls.append(c)
    # 返回都是list，每个list的长度都是3，每个元素分别代表了对应的3个尺度的信息
    # cls[i] 表示这个尺度的几个真实框分别对应的classid
    # tbox[i] 表示这个尺度的几个真实框的xywh信息，其中xy是单个cell的偏移量
    # indices[i] 表示这个尺度的几个真实框的单位格位置信息 和 这个真实框是由哪几个anchor来预测，以及imageid
    # anch[i] 表示这个尺度的几个真实框，需要用到的 先验框

    return tcls, tbox, indices, anch


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        # convex (smallest enclosing box) width
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * \
                    torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU
