import os
import random
import argparse
import time
import json
import warnings 
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

from train import train

# 전처리를 위한 라이브러리
from pycocotools.coco import COCO
import torchvision
import torchvision.transforms as transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2

# 시각화를 위한 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from matplotlib.patches import Patch
import webcolors




plt.rcParams['axes.grid'] = False

print('pytorch version: {}'.format(torch.__version__))
print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))

print(torch.cuda.get_device_name(0))
print(torch.cuda.device_count())

# GPU 사용 가능 여부에 따라 device 정보 저장
device = "cuda" if torch.cuda.is_available() else "cpu"

def arg_parser():
    parser = argparse.ArgumentParser('Semantic Segmentation', add_help=False)

    # Hyper-parameters
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_epochs', default=30, type=int)
    parser.add_argument('--learning_rate', default=0.0001, type=float)

def seed_everything(random_seed: int = 42):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)    

def main(args):

    model = models.segmentation.fcn_resnet50(pretrained=True)

    # output class를 data set에 맞도록 수정
    model.classifier[4] = nn.Conv2d(512, 11, kernel_size=1)

    # Loss function 정의
    criterion = nn.CrossEntropyLoss()

    # Optimizer 정의
    optimizer = torch.optim.Adam(params = model.parameters(), lr = args.learning_rate, weight_decay=1e-6)

    if args.scheduler == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.001)
    elif args.scheduler == 'multiply':
        lmbda = lambda epoch: 0.98739
        lr_scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)

    train(args.num_epochs, model, train_loader, val_loader, criterion, optimizer, lr_scheduler, saved_dir, val_every, device)


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
    parser = argparse.ArgumentParser(description='Semantic Segmentation', parents=[arg_parser()])
    args = parser.parse_args()
    seed_everything(args.seed)
    main(args)