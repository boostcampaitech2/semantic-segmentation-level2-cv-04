from abc import ABC, ABCMeta
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.base.model import SegmentationModel

class BaseModel(metaclass=ABCMeta):
	def __init__(self, encoderName, encoderWeights="imagenet", imgChannels=3, classes=11):
			self.model : SegmentationModel = None

	def __call__(self,*args,**kwargs):
			self.model(*args,**kwargs)

class UNetPlusPlus(BaseModel):
	def __init__(self, encoderName, encoderWeights="imagenet", imgChannels=3, classes=11):
		self.model = smp.UnetPlusPlus(
			encoder_name=encoderName,
			encoder_weights=encoderWeights,
			in_channels=imgChannels,
			classes=classes
		)
class DeepLabV3Plus(BaseModel):
	def __init__(self, encoderName, encoderWeights="imagenet", imgChannels=3, classes=11):
		self.model = smp.DeepLabV3Plus(
			encoder_name=encoderName,
			encoder_weights=encoderWeights,
			in_channels=imgChannels,
			classes=classes
		)