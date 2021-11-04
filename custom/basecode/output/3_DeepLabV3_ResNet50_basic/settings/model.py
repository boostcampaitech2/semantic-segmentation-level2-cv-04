# import segmentation_models_pytorch as smp
import torch.nn as nn
from torchvision import models

def getModel():
	
	model = models.segmentation.deeplabv3_resnet50(pretrained=True)
	model.classifier[4] = nn.Conv2d(256, 11, kernel_size=1)
	model.aux_classifier[4] = nn.Conv2d(256, 11, kernel_size=1)

	return  model
