import os
import random
import argparse
import warnings
from torch.serialization import save
import yaml
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import segmentation_models_pytorch as smp

import numpy as np
import pandas as pd
from tqdm import tqdm

from train import train
import models
from dataset import create_dataloader
from annotation import annotation
from transform import transform
from arg_parser import arg_parser
from logger import make_logger
from inference import inference

from datetime import datetime
from pytz import timezone


def seed_everything(random_seed: int = 42):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def main(args):

    print('pytorch version: {}'.format(torch.__version__))
    print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))

    print('Device name:', torch.cuda.get_device_name(0))
    print('Number of GPU:', torch.cuda.device_count())

    # GPU 사용 가능 여부에 따라 device 정보 저장
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_path = '../input/data'

    sorted_df = annotation(dataset_path)

    train_loader, val_loader, _ = create_dataloader(args.transform, args.batch_size)

    # model 정의
    model = models.fcn_resnet50()

    # Loss function 정의
    criterion = nn.CrossEntropyLoss()

    # Optimizer 정의
    optimizer = optim.Adam(params=model.parameters(),
                           lr=args.learning_rate, weight_decay=1e-6)

    # lr_scheduler 정의
    if args.scheduler == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.001)
    elif args.scheduler == 'multiply':
        def lmbda(epoch): return 0.98739
        lr_scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)

    # 모델 저장 이름 정의
    saved_dir = os.path.join('./saved', args.exp_name)

    # If saved_dir already exits, create a new dir
    i = 2
    while os.path.exists(saved_dir):
        saved_dir = os.path.join('./saved', args.exp_name + "_" + str(i))
        i += 1
    os.makedirs(saved_dir, exist_ok=True)
    
    # Model pickle file name
    file_name = args.exp_name + ".pt"

    # 인자 로그에 저장
    logger = make_logger(saved_dir, "main")
    logger.info(' '.join(f'{k}={v}' for k, v in vars(args).items()))

    # 인자 yaml 파일에 저장
    dict_file = {k:v for k,v in vars(args).items()}
    dict_file['output_path'] = os.path.join(saved_dir, file_name)

    with open(f"{saved_dir}/config.yaml", 'w') as file:
        yaml.dump(dict_file, file)
    
    exit(0)

    # 모델 학습
    train(args.num_epochs, model, train_loader, val_loader, criterion,
          optimizer, lr_scheduler, saved_dir, file_name, device, sorted_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Semantic Segmentation', parents=[arg_parser()])
    args = parser.parse_args()
    seed_everything(args.seed)
    main(args)