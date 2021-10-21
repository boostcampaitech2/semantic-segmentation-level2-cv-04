import segmentation_models_pytorch as smp

def getModel():
	
	model = smp.Unet(
			encoder_name="timm-efficientnet-b1",
			encoder_weights="imagenet",
			in_channels=3,
			classes=11
		)
	
	return model