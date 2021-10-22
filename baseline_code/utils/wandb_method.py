import wandb
import random

class WandBMethod:

	@staticmethod
	def login(arg, model, criterion, log="gradients", log_freq=10):
		wandb.login()
		wandb.init(project=arg.wandb_project, entity=arg.wandb_entity, name=arg.custom_name, config=arg)
		wandb.watch(model,criterion, log=log,log_freq=log_freq)
	
	@staticmethod
	def trainLog(loss, acc, lr):
		wandb.log({"train/loss":loss.item(),"train/decode.loss_ce":loss.item(),"train/decode.acc_seg":acc.item(), "learning_rate_torch":lr})
	
	@staticmethod
	def validLog(clsIoU, clsAcc,clsMeanAcc, mAcc, mIoU, images, outputs, masks):
		randIdx = random.randint(0,len(images)-1)
		categoryDict = {i:category for i, category in enumerate(['Background','General trash','Paper','Paper pack','Metal','Glass','Plastic','Styrofoam','Plastic bag','Battery','Clothing'])}
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
			# "val/aAcc":clsMeanAcc.item(),
			# "val/mAcc":mAcc.item(),
			"val/aAcc":mAcc.item(),
			"val/mAcc":clsMeanAcc.item(), #wandb 맞추는중
			"val/mIoU":mIoU.item(),
			"image" : wandb.Image(images[randIdx], masks={
					"predictions" : {
							"mask_data" : outputs[randIdx],
							"class_labels":categoryDict
					},
					"ground_truth" : {
							"mask_data" : masks[randIdx],
							"class_labels":categoryDict
					}}),
		})		
