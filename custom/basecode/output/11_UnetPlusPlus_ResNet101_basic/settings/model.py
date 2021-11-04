import segmentation_models_pytorch as smp

def getModel():
	
	# model = smp.DeepLabV3Plus(
	# 	encoder_name="tu-xception65",
	# 	encoder_weights="imagenet",
	# 	in_channels=3,
	# 	classes=11
	# )

	model = smp.UnetPlusPlus(
	encoder_name="resnet101",
	encoder_weights="imagenet",
	in_channels=3,
	classes=11
	)

	return  model
