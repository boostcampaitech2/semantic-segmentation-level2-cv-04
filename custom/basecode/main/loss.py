import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


# 미리 인자 지정해두기!!
_loss_entrypoints = {
    'cross_entropy': nn.CrossEntropyLoss(),
    'dice' : smp.losses.DiceLoss('multiclass'),
    'focal' : smp.losses.FocalLoss('multiclass'),
    'jaccard' : smp.losses.JaccardLoss('multiclass'),
    'lovasz' : smp.losses.LovaszLoss('multiclass'),
    'softBCE' : smp.losses.SoftBCEWithLogitsLoss(),
    'softCE' : smp.losses.SoftCrossEntropyLoss()
}


def loss_entrypoint(loss_name):
    return _loss_entrypoints[loss_name]


def is_loss(loss_name):
    return loss_name in _loss_entrypoints


def get_loss(loss_name):
    if is_loss(loss_name):
        loss = loss_entrypoint(loss_name)
    else:
        raise RuntimeError('Unknown loss (%s)' % loss_name)
    return loss

