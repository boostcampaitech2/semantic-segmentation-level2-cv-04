import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import segmentation_models_pytorch as smp

def fcn_resnet50():
    # model 정의
    model = models.segmentation.fcn_resnet50(pretrained=True)

    # output class를 data set에 맞도록 수정
    model.classifier[4] = nn.Conv2d(512, 11, kernel_size=1)

    return model