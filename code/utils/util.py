# -*- coding: utf-8 -*-
import os
import torch


def to_cpu(tensor):
    return tensor.detach().cpu()


def save_model(model, model_name, epoch, weights_dir):
    """
    保存第epoch个网络的参数
    :param model: torch.nn.Module, 需要训练的网络
    :param model_name: 保存模型前缀名称
    :param epoch: int, 表明当前训练的是第几个epoch
    :param models_pth_path: 模型保存路径
    """
    name = "{}_{}.pth".format(model_name, epoch)
    if not os.path.exists(weights_dir):
        # 创建目录
        os.makedirs(weights_dir)

    save_path = os.path.join(weights_dir, name)
    torch.save(model.state_dict(), save_path)
    print("saved model finished", name)



#固定初始化权重
def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    net.apply(init_func)


#---------------------------------------------------#
#   从优化器中获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']