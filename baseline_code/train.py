import torch
import argparse

from model.base_model import UNetPlusPlus
from dataset.base_dataset import CustomDataset
from dataloader.base_dataloader import BaseDataLoader
from utils.train_method import train
from torch import nn
import os


def getArg():
	parser = argparse.ArgumentParser()

	parser.add_argument('--batch', default=16,type=int, required=False)
	parser.add_argument('--epoch', default=20,type=int, required=False)
	parser.add_argument('--lr',default=1e-5,type=float, required=False)
	parser.add_argument('--seed', default=21,type=int,  required=False)
	parser.add_argument('--save_period', default=1, type=int, required=False)

	parser.add_argument('--image_root', default="../input/data", type=str, required=False)
	parser.add_argument('--train_json', default="train.json", type=str, required=False)
	parser.add_argument('--val_json', default="val.json", type=str, required=False)
	# parser.add_argument('--train_json', default="/test.json", type=str, required=False)

	parser.add_argument('--output_path', default="../output", type=str, required=False)

	return parser.parse_known_args()

def main(arg):

	print('pytorch version: {}'.format(torch.__version__))
	print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))
	device = "cuda" if torch.cuda.is_available() else "cpu"


	if not os.path.isdir(arg.output_path):                                                           
		os.mkdir(arg.output_path)

	from transform import train_transform, val_transform

	train_dataset = CustomDataset(data_dir=addPath([arg.image_root,arg.train_json]),image_root=arg.image_root, mode='train', transform=train_transform)
	val_dataset = CustomDataset(data_dir=addPath([arg.image_root,arg.val_json]),image_root=arg.image_root, mode='val', transform=val_transform)

	trainLoader = BaseDataLoader(dataset=train_dataset, batch_size=arg.batch,shuffle=True,num_workers=4)
	valLoader = BaseDataLoader(dataset=val_dataset, batch_size=arg.batch,shuffle=False,num_workers=4)

	model = UNetPlusPlus(encoderName="timm-efficientnet-b8").model

	# Loss function 정의
	criterion = nn.CrossEntropyLoss()

	# Optimizer 정의
	optimizer = torch.optim.Adam(params = model.parameters(), lr = arg.lr, weight_decay=1e-5)

	# TODO 이름 넣어라
	train(arg.epoch, model, trainLoader, valLoader, criterion, optimizer, arg.output_path, arg.save_period, device)

def setSeed(seed):
	import numpy as np
	import random
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if use multi-GPU
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	np.random.seed(seed)
	random.seed(seed)

def addPath(pathList):
	return os.path.join(*pathList)

if __name__=="__main__":
	arg, _ = getArg()
	setSeed(arg.seed)
	main(arg)