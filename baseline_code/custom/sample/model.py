import segmentation_models_pytorch as smp

# https://smp.readthedocs.io/en/latest/index.html

def getModel():
	
	model = smp.Unet(
			encoder_name="timm-efficientnet-b1",
			encoder_weights="imagenet",
			in_channels=3,
			classes=11
		)
	
	return model