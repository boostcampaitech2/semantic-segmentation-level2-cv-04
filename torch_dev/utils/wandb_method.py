from math import sqrt, floor
import wandb
import random
import cv2
import numpy as np

class WandBMethod:
	'''
	WandB에 관련된 method들이 모여있는 helper 클래스 입니다.
	'''

	@staticmethod
	def login(arg, model, criterion, log="gradients", log_freq=10):
		'''
		초기화 함수 모음
		'''
		wandb.login()
		wandb.init(project=arg.wandb_project, entity=arg.wandb_entity, name=arg.custom_name, config=arg)
		wandb.watch(model,criterion, log=log,log_freq=log_freq)
	
	@staticmethod
	def trainLog(loss, acc, lr):
		'''
		train에서 각 batch마다 보내는 정보로, mmsegmentation format에 최대한 맞춤
		'''
		wandb.log({"train/loss":loss.item(),"train/decode.loss_ce":loss.item(),"train/decode.acc_seg":acc.item(), "learning_rate_torch":lr})
	
	@classmethod
	def validLog(cls, clsIoU, clsAcc,clsMeanAcc, mAcc, mIoU, images, outputs, masks):
		'''
		각 epoch당 valid가 끝난 뒤 한번 보내지는 정보들로, mmsegmentation format에 최대한 맞춰서 정보가 좀 직관적이지 않고 많음
		'''
		categoryDict = {i:category for i, category in enumerate(['Background','General trash','Paper','Paper pack','Metal','Glass','Plastic','Styrofoam','Plastic bag','Battery','Clothing'])}
		
		image = cls.concatImages(images)
		output = cls.concatImages(outputs)
		mask = cls.concatImages(masks)
		
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
			"val/mIoU":mIoU.item(),
			"image" : wandb.Image(image, masks={
					"predictions" : {
							"mask_data" : output,
							"class_labels":categoryDict
					},
					"ground_truth" : {
							"mask_data" : mask,
							"class_labels":categoryDict
					}}),
		})

	@staticmethod
	def pickImageStep(length):
		'''
		valid할 때 여러 batch 중 하나를 선택
		'''
		return random.randint(0,length-1)	

	@staticmethod
	def concatImages(images):
		'''
		여러개의 이미지를 하나의 이미지로 합쳐주는 과정입니다.
		batch size에서 가장 가까운 정사각형으로 설정되게 해놨습니다

		ex) 
		batch 8 -> 2x2 
		batch 16 -> 4x4
		batch 32 -> 5x5
		'''

		length = len(images)
		squareSide = floor(sqrt(length))

		hConcatImgs = []
		for i in range(0,squareSide*squareSide,squareSide):
			imgList = []
			for j in range(i,i+squareSide):
				if images[j].shape == (512,512): # mask의 경우 채널이 1개이기 때문에 따로 정제 필요
					nowImage = np.expand_dims(images[j],axis=2) 
				else:
					nowImage = np.transpose(images[j],(1,2,0)) # tensor -> cv2 포맷으로 변경
				imgList.append(nowImage)

			hConcatImgs.append(cv2.hconcat(imgList))

		fullImg = cv2.vconcat(hConcatImgs)
		resizedImg = cv2.resize(fullImg.astype('float32'), dsize=(512,512), interpolation=cv2.INTER_AREA) # 최종 출력을 512x512로 축소
		return resizedImg
