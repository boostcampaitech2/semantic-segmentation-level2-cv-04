# import segmentation_models_pytorch as smp
import torch.nn as nn
from torchvision import models

def getModel():
	
	model = models.segmentation.fcn_resnet50(pretrained=True)
	model.classifier[4] = nn.Conv2d(512, 11, kernel_size=1)
	model.aux_classifier[4] = nn.Conv2d(256, 11, kernel_size=1)

	return  model


# from efficientnet_pytorch import EfficientNet


# https://smp.readthedocs.io/en/latest/index.html

# def getModel():
	
# 	model = smp.Unet(
# 			encoder_name="timm-efficientnet-b1",
# 			encoder_weights="imagenet",
# 			in_channels=3,
# 			classes=11
# 		)
	
# 	return model