"""
pstage level1 image classification baseline code 참고
"""
import torch.nn as nn
from segmentation_models_pytorch.losses.soft_ce import SoftCrossEntropyLoss
from segmentation_models_pytorch.losses.focal import FocalLoss
from segmentation_models_pytorch.losses.dice import DiceLoss
from segmentation_models_pytorch.losses.soft_ce import SoftCrossEntropyLoss
from segmentation_models_pytorch.losses.jaccard import JaccardLoss 

# Dice + Focal loss
class DiceFocalLoss(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.focal_loss = FocalLoss(mode='multiclass')
        self.dice_loss =  DiceLoss(mode='multiclass')

    def forward(self, outputs, masks):
        return self.focal_loss(outputs, masks) + self.dice_loss(outputs, masks)

# Jaccard + Soft CrossEntropy
class JaccardSoftCE(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.jaccard_loss = JaccardLoss(mode='multiclass')
        self.soft_ce = SoftCrossEntropyLoss(smooth_factor=0.1)

    def forward(self, outputs, masks):
        return self.jaccard_loss(outputs, masks) + self.soft_ce(outputs, masks)

# DiceLoss
class DiceLoss_(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.dice_loss =  DiceLoss(mode='multiclass')

    def forward(self, outputs, masks):
        return self.dice_loss(outputs, masks)

# FocalLoss
class FocalLoss_(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.focal_loss = FocalLoss(mode='multiclass')

    def forward(self, outputs, masks):
        return self.focal_loss(outputs, masks)

#Soft CrossEntropy LOss
class SoftCrossEntropyLoss_(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.soft_ce = SoftCrossEntropyLoss(smooth_factor=0.1)

    def forward(self, outputs, masks):
        return self.soft_ce(outputs, masks)

# JaccardLoss
class JaccardLoss_(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.jaccard_loss = JaccardLoss(mode='multiclass')

    def forward(self, outputs, masks):
        return self.jaccard_loss(outputs, masks)

_criterion_entrypoints = {
    'cross_entropy': nn.CrossEntropyLoss,
    'soft_ce' : SoftCrossEntropyLoss_,
    'focal' : FocalLoss_,
    'dice' : DiceLoss_,
    'dice_focal' : DiceFocalLoss,
    'jaccard' : JaccardLoss_,
    'jaccard_soft_ce' : JaccardSoftCE
}


def criterion_entrypoint(criterion_name):
    return _criterion_entrypoints[criterion_name]

# Loss 이름 존재하는지 확인
def is_criterion(criterion_name):
    return criterion_name in _criterion_entrypoints

# get Loss
def create_criterion(criterion_name, **kwargs):
    if is_criterion(criterion_name):
        create_fn = criterion_entrypoint(criterion_name)
        criterion = create_fn(**kwargs)
    else:
        raise RuntimeError('Unknown loss (%s)' % criterion_name)
    return criterion