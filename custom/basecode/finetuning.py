import torch
import argparse
import os
from importlib import import_module

from dataset.base_dataset import CustomDataset
from utils.finetuning_method import train
from utils.set_seed import setSeed

def getArgument():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--custom_name',type=str ,required=True)
    parser.add_argument('--custom_name',type=str ,default="20_DeepLabV3Plus_ResNest269e_recommendAug")
    parser.add_argument('--epoch',type=int ,default=15)
    parser.add_argument('--batch',type=int ,default=4)
    return parser.parse_known_args()[0].custom_name, parser.parse_known_args()[0].epoch, parser.parse_known_args()[0].batch


def main(custom_name, epoch, batch):
    arg = getattr(import_module(f"output.{custom_name}.sample"), "getArg")()

    # finetuning 위한 epoch 변경
    arg.batch = batch
    arg.epoch = epoch
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setSeed(arg.seed)

    outputPath = os.path.join(arg.output_path, arg.custom_name, "finetuning")
    os.makedirs(outputPath, exist_ok=True)

    train_transform, _ = getattr(import_module("main.transform"), "get_transfrom")(arg.transform)

    # valid set 추가학습 (train_datset에 val_json으로 변경함)
    train_dataset = CustomDataset(data_dir=addPath([arg.image_root,arg.val_json]),image_root=arg.image_root, mode='train', transform=train_transform)

    trainLoader = getattr(import_module("main.dataloader"), "getTrainDataloader")(
        train_dataset, arg.batch, arg.train_worker)

    for name in os.listdir(f"./output/{custom_name}"):
        if 'best' in name:
            best_model = name
            break
    
    model_path = os.path.join(arg.output_path, custom_name, best_model)

    checkpoint = torch.load(model_path, map_location=device)
    model = getattr(import_module("main.model"), arg.model)().get_backbone(arg.backbone).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    criterion = getattr(import_module("main.loss"), "get_loss")(arg.loss)
    optimizer, scheduler = getattr(import_module("main.opt_scheduler"), "getOptAndScheduler")(model, arg.lr)
    optimizer.load_state_dict(checkpoint['optimizer'])

    tv = False
    if arg.model in ["FCN", "DeepLabV3"]:
        tv = True

    train(arg.epoch, model, trainLoader, criterion, optimizer, scheduler, outputPath, arg.save_capacity, device, tv)


def addPath(pathList):
	return os.path.join(*pathList)

if __name__=="__main__":
	custom_name, epoch, batch = getArgument()
	main(custom_name, epoch, batch)
