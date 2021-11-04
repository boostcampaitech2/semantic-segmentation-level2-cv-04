from math import sqrt, floor
import wandb
import random
import cv2
import numpy as np

class WandBMethod:

	@staticmethod
	def login(arg, model, criterion, log="gradients", log_freq=10):
		wandb.login()
		wandb.init(project=arg.wandb_project, entity=arg.wandb_entity, name=arg.wandb_custom_name, config=arg)
		wandb.watch(model,criterion, log=log,log_freq=log_freq)
	
	@staticmethod
	def trainLog(loss, acc, lr):
		wandb.log({"train/loss":loss.item(),"train/decode.loss_ce":loss.item(),"train/decode.acc_seg":acc.item(), "learning_rate_torch":lr})
	
	@classmethod
	def validLog(cls, clsIoU, clsAcc,clsMeanAcc, mAcc, mIoU, images, outputs, masks):
		randIdx = random.randint(0,len(images)-1)
		categoryDict = {i:category for i, category in enumerate(['Background','General trash','Paper','Paper pack','Metal','Glass','Plastic','Styrofoam','Plastic bag','Battery','Clothing'])}
		
		# image = cls.concatImages(images)
		# output = cls.concatImages(outputs)
		# mask = cls.concatImages(masks)
		
		wandb.log({
			"val/IoU.Background":clsIoU[0],
			"val/IoU.Battery":clsIoU[10],
			"val/IoU.Clothing":clsIoU[9],
			"val/IoU.General trash":clsIoU[1],
			"val/IoU.Glass":clsIoU[5],
			"val/IoU.Metal":clsIoU[4],
			"val/IoU.Paper":clsIoU[2],
			"val/IoU.Paper pack":clsIoU[3],
			"val/IoU.Plastic":clsIoU[6],
			"val/IoU.Plastic bag":clsIoU[8],
			"val/IoU.Styrofoam":clsIoU[7],
			"val/Acc.Background":clsAcc[0],
			"val/Acc.Battery":clsAcc[10],
			"val/Acc.Clothing":clsAcc[9],
			"val/Acc.General trash":clsAcc[1],
			"val/Acc.Glass":clsAcc[5],
			"val/Acc.Metal":clsAcc[4],
			"val/Acc.Paper":clsAcc[2],
			"val/Acc.Paper pack":clsAcc[3],
			"val/Acc.Plastic":clsAcc[6],
			"val/Acc.Plastic bag":clsAcc[8],
			"val/Acc.Styrofoam":clsAcc[7],
			"val/aAcc":mAcc.item(),
			"val/mAcc":clsMeanAcc.item(), #wandb 맞추는중
			"val/mIoU":mIoU.item()
			# "image" : wandb.Image(image, masks={
			# 		"predictions" : {
			# 				"mask_data" : output,
			# 				"class_labels":categoryDict
			# 		},
			# 		"ground_truth" : {
			# 				"mask_data" : mask,
			# 				"class_labels":categoryDict
			# 		}}),
		})

	@staticmethod
	def pickImageStep(length):
		return random.randint(0,length-1)	

	@staticmethod
	def concatImages(images):
		length = len(images)
		squareSide = floor(sqrt(length))

		hConcatImgs = []
		for i in range(0,squareSide*squareSide,squareSide):
			imgList = []
			for j in range(i,i+squareSide):
				if images[j].shape == (512,512):
					nowImage = np.expand_dims(images[j],axis=2) 
				else:
					nowImage = np.transpose(images[j],(1,2,0))
				imgList.append(nowImage)

			hConcatImgs.append(cv2.hconcat(imgList))
		return cv2.vconcat(hConcatImgs)