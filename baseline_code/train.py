import os
import argparse
import torch
import numpy as np
import random
import yaml
from importlib import import_module
import segmentation_models_pytorch as smp

from tool.tools import train
from data_set.data_set import CustomDataSet, collate_fn
from data_set.data_augmentation import get_transform
from model.loss import create_criterion
from model.custom_encoder import register_encoder
from logger.wandb_logger import wandb_init
from logger.logger import yaml_logger, make_dir

# Seed 고정
def seed_everything(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def run(args, cfg, device):
    seed_everything(cfg['seed'])
    wandb_init(cfg['exp_name'])

    # Custom encoder(Swin) smp에 등록
    register_encoder()

    # cfg saved 폴더에 저장
    cfg['saved_dir'] = make_dir(cfg['saved_dir'], cfg['exp_name'])
    yaml_logger(args, cfg)

    # Transform 불러오기
    train_transform = get_transform(cfg['transforms']['name'])
    val_transform = get_transform('valid_transform')

    # DataSet 설정
    train_dataset = CustomDataSet(data_dir=cfg['train_path'], dataset_path=cfg['dataset_path'], mode='train', transform=train_transform)
    val_dataset = CustomDataSet(data_dir=cfg['val_path'], dataset_path=cfg['dataset_path'], mode='val', transform=val_transform)

    # DataLoader 설정
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, collate_fn=collate_fn,
                                                **cfg['train_dataloader']['params'])

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, collate_fn=collate_fn,
                                                **cfg['valid_dataloader']['params'])
    # Model 불러오기
    model = smp.__dict__[cfg['model']['name']]
    model = model(**cfg['model']['params'])
    
    # Loss function 설정
    criterion = create_criterion(cfg['criterion']['name'])

    # Optimizer 설정
    opt_module = getattr(import_module("torch.optim"), cfg['optimizer']['name'])
    optimizer = opt_module(params = model.parameters(), **cfg['optimizer']['params'])

    # Scheduler 설정
    scheduler_module = getattr(import_module("torch.optim.lr_scheduler"), cfg['scheduler']['name'])
    scheduler=scheduler_module(optimizer, **cfg['scheduler']['params'])

    # train
    train(cfg['epochs'], model, train_loader, val_loader, criterion, optimizer, scheduler, cfg['saved_dir'], cfg['val_every'], cfg['exp_name'], device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='./config/base_test.yaml', help='yaml file path')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Yaml 파일에서 config 가져오기
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run(args, cfg, device)