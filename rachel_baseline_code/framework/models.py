import torch.nn as nn
from torchvision import models
import segmentation_models_pytorch as smp


def seg_model(model_name):
    is_aux = False

    if model_name == 'DeepLabV3Plus_xception71':
        model = smp.DeepLabV3Plus(
            encoder_name="tu-xception71",
            encoder_weights="imagenet",
            in_channels=3,
            classes=11)

    elif model_name == 'DeepLabV3Plus_xception71_aux':
        is_aux=True
        aux_params = dict(
            pooling='avg',             # one of 'avg', 'max'
            dropout=0.5,               # dropout ratio, default is None
            activation='sigmoid',      # activation function, default is None
            classes=11)                # define number of output labels

        model = smp.DeepLabV3Plus(
            encoder_name="tu-xception71",
            encoder_weights="imagenet",
            in_channels=3,
            classes=11,
            aux_params=aux_params)

    return model, is_aux
