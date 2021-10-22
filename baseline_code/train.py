import os
import argparse
import torch
import numpy as np
import random
import yaml
from importlib import import_module
import segmentation_models_pytorch as smp

from tools import train
from data_set.data_set import CustomDataSet, collate_fn
from data_set.data_augmentation import get_transform
from model.loss import create_criterion
from logger.wandb_logger import wandb_init
from logger.logger import yaml_logger, make_dir

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

    # config logging
    cfg['saved_dir'] = make_dir(cfg['saved_dir'], cfg['exp_name'])
    yaml_logger(args, cfg)

    train_transform = get_transform(cfg['transforms']['name'])
    val_transform = get_transform('valid_transform')

    train_dataset = CustomDataSet(data_dir=cfg['train_path'], dataset_path=cfg['dataset_path'], mode='train', transform=train_transform)
    val_dataset = CustomDataSet(data_dir=cfg['val_path'], dataset_path=cfg['dataset_path'], mode='val', transform=val_transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, collate_fn=collate_fn,
                                                **cfg['train_dataloader']['params'])

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, collate_fn=collate_fn,
                                                **cfg['valid_dataloader']['params'])
    # model 불러오기
    model = smp.__dict__[cfg['model']['name']]
    model = model(**cfg['model']['params'])
    
    # Loss function 정의
    criterion = create_criterion(cfg['criterion']['name'])

    # Optimizer 정의
    opt_module = getattr(import_module("torch.optim"), cfg['optimizer']['name'])
    optimizer = opt_module(params = model.parameters(), **cfg['optimizer']['params'])

    #Scheduler 정의
    scheduler_module = getattr(import_module("torch.optim.lr_scheduler"), cfg['scheduler']['name'])
    scheduler=scheduler_module(optimizer, **cfg['scheduler']['params'])
    train(cfg['epochs'], model, train_loader, val_loader, criterion, optimizer, scheduler, cfg['saved_dir'], cfg['val_every'], cfg['exp_name'], device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default='./config/base_test.yaml', help='yaml file path')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run(args, cfg, device)