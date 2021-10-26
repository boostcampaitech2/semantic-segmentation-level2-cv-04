import os
import random
import argparse
import time
import json
import warnings

from torch.utils import data 
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from utils import label_accuracy_score, add_hist
import cv2

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim
from torchvision import models
import segmentation_models_pytorch as smp

from train import train
from dataset import CustomDataLoader
from annotation import annotation
from transform import transform
from arg_parser import arg_parser

# 전처리를 위한 라이브러리
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 시각화를 위한 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from matplotlib.patches import Patch
import webcolors


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

    # train.json / validation.json / test.json 디렉토리 설정
    dataset_path = '../input/data'
    train_path = dataset_path + '/train.json'
    val_path = dataset_path + '/val.json'
    test_path = dataset_path + '/test.json'

    # class (Categories) 에 따른 index 확인 (0~10 : 총 11개)
    sorted_df = annotation(dataset_path)
    category_names = list(sorted_df.Categories)

    # collate_fn needs for batch
    def collate_fn(batch):
        return tuple(zip(*batch))

    # Data Augmentation
    train_transform = transform(args.transform)

    val_transform = transform(args.transform)

    test_transform = transform(args.transform)

    # create own Dataset 1 (skip)
    # validation set을 직접 나누고 싶은 경우
    # random_split 사용하여 data set을 8:2 로 분할
    # train_size = int(0.8*len(dataset))
    # val_size = int(len(dataset)-train_size)
    # dataset = CustomDataLoader(data_dir=train_path, mode='train', transform=transform)
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # create own Dataset 2
    # train dataset
    train_dataset = CustomDataLoader(data_dir=train_path,
                                     dataset_path=dataset_path,
                                     category_names=category_names,
                                     mode='train',
                                     transform=train_transform)

    # validation dataset
    val_dataset = CustomDataLoader(data_dir=val_path,
                                   dataset_path=dataset_path,
                                   category_names=category_names,
                                   mode='val',
                                   transform=val_transform)

    # test dataset
    test_dataset = CustomDataLoader(data_dir=test_path,
                                    dataset_path=dataset_path,
                                    category_names=category_names,
                                    mode='test',
                                    transform=test_transform)

    # DataLoader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=4,
                                               collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=4,
                                             collate_fn=collate_fn)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=args.batch_size,
                                              num_workers=4,
                                              collate_fn=collate_fn)

    # model 정의
    model = models.segmentation.fcn_resnet50(pretrained=True)

    # output class를 data set에 맞도록 수정
    model.classifier[4] = nn.Conv2d(512, 11, kernel_size=1)

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

    # 모델 저장 함수 정의
    val_every = 1

    saved_dir = './saved'
    if not os.path.isdir(saved_dir):
        os.mkdir(saved_dir)

    # 모델 학습
    train(args.num_epochs, model, train_loader, val_loader, criterion,
          optimizer, lr_scheduler, saved_dir, val_every, device, sorted_df)

    # best model 저장된 경로
    model_path = './saved/fcn_resnet50_best_model(pretrained).pt'

    # best model 불러오기
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.state_dict()
    model.load_state_dict(state_dict)

    model = model.to(device)
    # 추론을 실행하기 전에는 반드시 설정 (batch normalization, dropout 를 평가 모드로 설정)
    # model.eval()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Semantic Segmentation', parents=[arg_parser()])
    args = parser.parse_args()
    print(' '.join(f'{k}={v}' for k, v in vars(args).items()))
    seed_everything(args.seed)
    main(args)