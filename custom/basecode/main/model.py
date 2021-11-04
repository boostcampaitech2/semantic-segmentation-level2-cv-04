from segmentation_models_pytorch.unetplusplus.model import UnetPlusPlus
import torch.nn as nn
from torchvision import models
import segmentation_models_pytorch as smp

# ------------------- torchvision -------------------
class FCN:
    def __init__(self, num_classes=11):
        self.num_classes = num_classes
        self.backbone = {
            "resnet50" : self.resnet50(),
            "resnet101" : self.resnet101(),
        }


    def resnet50(self):
        model = models.segmentation.fcn_resnet50(pretrained=True)

        model.classifier[4] = nn.Conv2d(512, self.num_classes, kernel_size=1)
        model.aux_classifier[4] = nn.Conv2d(256, self.num_classes, kernel_size=1)

        return model


    def resnet101(self):
        model = models.segmentation.fcn_resnet101(pretrained=True)

        model.classifier[4] = nn.Conv2d(512, self.num_classes, kernel_size=1)
        model.aux_classifier[4] = nn.Conv2d(256, self.num_classes, kernel_size=1)

        return model


    def backbone_entrypoint(self, backbone_name):
        return self.backbone[backbone_name]


    def is_backbone(self, backbone_name):
        return backbone_name in self.backbone


    def get_backbone(self, backbone_name):
        if self.is_backbone(backbone_name):
            backbone = self.backbone_entrypoint(backbone_name)
        else:
            raise RuntimeError('Unknown backbone (%s)' % backbone_name)
        return backbone


class DeepLabV3:
    def __init__(self, num_classes=11):
        self.num_classes = num_classes
        self.backbone = {
            "resnet50" : self.resnet50(),
            "resnet101" : self.resnet101(),
        }


    def resnet50(self):
        model = models.segmentation.deeplabv3_resnet50(pretrained=True)
        model.classifier[4] = nn.Conv2d(256, self.num_classes, kernel_size=1)
        model.aux_classifier[4] = nn.Conv2d(256, self.num_classes, kernel_size=1)

        return model


    def resnet101(self):
        model = models.segmentation.deeplabv3_resnet101(pretrained=True)
        model.classifier[4] = nn.Conv2d(256, self.num_classes, kernel_size=1)
        model.aux_classifier[4] = nn.Conv2d(256, self.num_classes, kernel_size=1)

        return model


    def backbone_entrypoint(self, backbone_name):
        return self.backbone[backbone_name]


    def is_backbone(self, backbone_name):
        return backbone_name in self.backbone


    def get_backbone(self, backbone_name):
        if self.is_backbone(backbone_name):
            backbone = self.backbone_entrypoint(backbone_name)
        else:
            raise RuntimeError('Unknown backbone (%s)' % backbone_name)
        return backbone


# ------------------- smp -------------------
class DeepLabV3Plus:
    def __init__(self, num_classes=11):
        self.num_classes = num_classes
        self.backbone = {
            "xception65" : self.xception65(),
            "xception71" : self.xception71(),
            "resnest101e" : self.resnest101e(),
            "resnest269e" : self.resnest269e()
        }


    def xception65(self):
        model = smp.DeepLabV3Plus(
            encoder_name="tu-xception65",
            encoder_weights="imagenet",
            in_channels=3,
            classes=11
        )

        return model


    def xception71(self):
        model = smp.DeepLabV3Plus(
            encoder_name="tu-xception71",
            encoder_weights="imagenet",
            in_channels=3,
            classes=11
        )

        return model

    
    def resnest101e(self):
        model = smp.UnetPlusPlus(
            encoder_name="timm-resnest101e",
            encoder_weights="imagenet",
            in_channels=3,
            classes=11
        )

        return model

    
    def resnest269e(self):
        model = smp.UnetPlusPlus(
            encoder_name="timm-resnest269e",
            encoder_weights="imagenet",
            in_channels=3,
            classes=11
        )

        return model


    def backbone_entrypoint(self, backbone_name):
        return self.backbone[backbone_name]


    def is_backbone(self, backbone_name):
        return backbone_name in self.backbone


    def get_backbone(self, backbone_name):
        if self.is_backbone(backbone_name):
            backbone = self.backbone_entrypoint(backbone_name)
        else:
            raise RuntimeError('Unknown backbone (%s)' % backbone_name)
        return backbone


class UNetPlusPlus:
    def __init__(self, num_classes=11):
        self.num_classes = num_classes
        self.backbone = {
            "resnet101" : self.resnet101(),
            "efficientnetb7" : self.efficientnetb7()
        }

    def resnet101(self):
        model = smp.UnetPlusPlus(
            encoder_name="resnet101",
            encoder_weights="imagenet",
            in_channels=3,
            classes=11
        )

        return model


    def efficientnetb7(self):
        model = smp.PSPNet(
            encoder_name="timm-efficientnet-b7",
            encoder_weights="imagenet",
            in_channels=3,
            classes=11
        )

        return model


    def backbone_entrypoint(self, backbone_name):
        return self.backbone[backbone_name]


    def is_backbone(self, backbone_name):
        return backbone_name in self.backbone


    def get_backbone(self, backbone_name):
        if self.is_backbone(backbone_name):
            backbone = self.backbone_entrypoint(backbone_name)
        else:
            raise RuntimeError('Unknown backbone (%s)' % backbone_name)
        return backbone


class PSPNet:
    def __init__(self, num_classes=11):
        self.num_classes = num_classes
        self.backbone = {
            "resnet101" : self.resnet101(),
            "efficientnetb7" : self.efficientnetb7()
        }


    def resnet101(self):
        model = smp.PSPNet(
            encoder_name="resnet101",
            encoder_weights="imagenet",
            in_channels=3,
            classes=11
        )

        return model


    def efficientnetb7(self):
        model = smp.PSPNet(
            encoder_name="timm-efficientnet-b7",
            encoder_weights="imagenet",
            in_channels=3,
            classes=11
        )

        return model


    def backbone_entrypoint(self, backbone_name):
        return self.backbone[backbone_name]


    def is_backbone(self, backbone_name):
        return backbone_name in self.backbone


    def get_backbone(self, backbone_name):
        if self.is_backbone(backbone_name):
            backbone = self.backbone_entrypoint(backbone_name)
        else:
            raise RuntimeError('Unknown backbone (%s)' % backbone_name)
        return backbone
