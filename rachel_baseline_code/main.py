import os
import argparse
import warnings
import yaml
warnings.filterwarnings('ignore')

import torch

from train import train
from framework.models import seg_model
from framework.dataset import create_dataloader
from framework.criterion import seg_loss
from framework.optimizer import seg_optimizer
from framework.scheduler import seg_scheduler
from utils.annotation import annotation
from utils.arg_parser import arg_parser
from utils.logger import make_logger
from utils.utils import seed_everything
from utils.wandb import WandBMethod


def main(args):

    print('pytorch version: {}'.format(torch.__version__))
    print('GPU 사용 가능 여부: {}'.format(torch.cuda.is_available()))

    print('Device name:', torch.cuda.get_device_name(0))
    print('Number of GPU:', torch.cuda.device_count())

    # GPU 사용 가능 여부에 따라 device 정보 저장
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_path = '../input/data'

    sorted_df = annotation(dataset_path)

    train_loader, val_loader, _ = create_dataloader(args.transform, args.batch_size, args.train_path, args.valid_path)

    # model 정의
    model, is_aux = seg_model(args.model)

    # Loss function 정의
    criterion = seg_loss(args.loss)

    # Optimizer 정의
    optimizer = seg_optimizer(args.opt_name, model, args.learning_rate)

    # lr_scheduler 정의
    lr_scheduler = seg_scheduler(args, optimizer)
    
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

    if args.wandb:
        WandBMethod.login(args, model, criterion)

    # 모델 학습
    train(args.num_epochs, model, train_loader, val_loader, criterion,
          optimizer, lr_scheduler, saved_dir, file_name, device, sorted_df, args.wandb, is_aux)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Semantic Segmentation', parents=[arg_parser()])
    args = parser.parse_args()
    seed_everything(args.seed)
    main(args)