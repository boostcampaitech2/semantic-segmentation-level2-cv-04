import torch
import argparse
import os
import shutil
from importlib import import_module

from dataset.base_dataset import CustomDataset
from utils.train_method import train
from utils.set_seed import setSeed

def getArgument():
	parser = argparse.ArgumentParser()
	parser.add_argument('--custom_arg',type=str ,required=True)

	# parser.parse_known_args() >> (Namespace(custom_dir='sample'), [])
	# parser.parse_known_args()[0] >> Namespace(custom_dir='sample')
	# parser.parse_known_args()[0].custom_dir >> 'sample'
	return parser.parse_known_args()[0].custom_arg


def main(custom_arg):

	arg = getattr(import_module(f"arg.{custom_arg}"), "getArg")()

	device = "cuda" if torch.cuda.is_available() else "cpu"
	setSeed(arg.seed)

	outputPath = os.path.join(arg.output_path, arg.custom_name)
	os.makedirs(outputPath, exist_ok=False)
	shutil.copy(f"arg/{custom_arg}.py", outputPath) # copytree(a, b) : a에 있는 파일과 폴더를 b로 복사 

	train_transform, val_transform = getattr(import_module("main.transform"), "get_transfrom")(arg.transform)

	train_dataset = CustomDataset(data_dir=addPath([arg.image_root,arg.train_json]),image_root=arg.image_root, mode='train', transform=train_transform)
	val_dataset = CustomDataset(data_dir=addPath([arg.image_root,arg.val_json]),image_root=arg.image_root, mode='val', transform=val_transform)

	trainLoader, valLoader = getattr(import_module("main.dataloader"), "getDataloader")(
		train_dataset, val_dataset, arg.batch, arg.train_worker, arg.valid_worker)

	model = getattr(import_module("main.model"), arg.model)().get_backbone(arg.backbone)
	criterion = getattr(import_module("main.loss"), "get_loss")(arg.loss)
	optimizer, scheduler = getattr(import_module("main.opt_scheduler"), "getOptAndScheduler")(model, arg.lr)

	# torchvision 모델 이용 시, outputs 형태 바꿔줘야 함
	tv = False
	if arg.model in ["FCN", "DeepLabV3"]:
		tv = True
	
	# wandb
	from utils.wandb_method import WandBMethod
	if arg.wandb:
			WandBMethod.login(arg, model, criterion)

	train(arg.epoch, model, trainLoader, valLoader, criterion, optimizer,scheduler, outputPath, arg.save_capacity, device, arg.wandb, arg.patience, arg.object_best, tv)

def addPath(pathList):
	return os.path.join(*pathList)

if __name__=="__main__":
	custom_arg = getArgument()
	main(custom_arg)