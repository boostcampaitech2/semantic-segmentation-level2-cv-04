import torch
import argparse
import os
import shutil
from importlib import import_module

from dataset.base_dataset import CustomDataset
from utils.train_method import train
# from utils.train_method_filter_pseudo import train
from utils.set_seed import setSeed

def getArgument():
	parser = argparse.ArgumentParser()
	parser.add_argument('--custom_dir',type=str ,required=True)
	return parser.parse_known_args()[0].custom_dir


def main(custom_dir):

	arg = getattr(import_module(f"custom.{custom_dir}.arg"), "getArg")()

	device = "cuda" if torch.cuda.is_available() else "cpu"
	setSeed(arg.seed)

	outputPath = os.path.join(arg.output_path, arg.custom_name)
	os.makedirs(outputPath, exist_ok=False)
	shutil.copytree(f"custom/{custom_dir}",outputPath+"/settings")

	train_transform, val_transform = getattr(import_module(f"custom.{custom_dir}.transform"), "getTransform")()

	# train_dataset = CustomDataset(data_dir=addPath([arg.image_root,arg.train_json]),image_root=arg.image_root, mode='train', transform=train_transform)
	# val_dataset = CustomDataset(data_dir=addPath([arg.image_root,arg.val_json]),image_root=arg.image_root, mode='val', transform=val_transform)

# 수정부분
	train_dataset = CustomDataset(data_dir=addPath([arg.image_root,arg.train_json]),image_root=arg.image_root, mode='train', transform=train_transform, zscore=arg.zscore)
	val_dataset = CustomDataset(data_dir=addPath([arg.image_root,arg.val_json]),image_root=arg.image_root, mode='val', transform=val_transform, zscore=arg.zscore)

	trainLoader, valLoader = getattr(import_module(f"custom.{custom_dir}.dataloader"), "getDataloader")(
		train_dataset, val_dataset, arg.batch, arg.train_worker, arg.valid_worker)

	model = getattr(import_module(f"custom.{custom_dir}.model"), "getModel")()
	criterion = getattr(import_module(f"custom.{custom_dir}.loss"), "getLoss")()

	optimizer, scheduler = getattr(import_module(f"custom.{custom_dir}.opt_scheduler"), "getOptAndScheduler")(model, arg.lr)
	
	# wandb
	from utils.wandb_method import WandBMethod
	if arg.wandb:
			WandBMethod.login(arg, model, criterion)

	train(arg.epoch, model, trainLoader, valLoader, criterion, optimizer,scheduler, outputPath, arg.save_capacity, device, arg.wandb)

def addPath(pathList):
	return os.path.join(*pathList)

if __name__=="__main__":
	custom_dir = getArgument()
	main(custom_dir)