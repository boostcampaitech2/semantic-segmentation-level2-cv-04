import os
import random
import time
import yaml
import warnings 
import argparse
from utils.arg_parser import arg_parser_infer
warnings.filterwarnings('ignore')

import torch

import numpy as np
import pandas as pd
from tqdm import tqdm

import utils.models as models
from utils.dataset import create_dataloader
from utils.utils import seed_everything

import albumentations as A


def inference(model, test_loader, device):
    size = 256
    transform = A.Compose([A.Resize(size, size)])
    print('Start prediction.')

    model.eval()

    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)

    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(tqdm(test_loader)):

            # inference (512 x 512)
            outs = model(torch.stack(imgs).to(device))['out']
            oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()

            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)

            oms = np.array(temp_mask)

            oms = oms.reshape([oms.shape[0], size*size]).astype(int)
            preds_array = np.vstack((preds_array, oms))

            file_name_list.append([i['file_name'] for i in image_infos])
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]

    return file_names, preds_array


def main(args):

    model = models.fcn_resnet50()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, _, test_loader = create_dataloader(args.transform, args.batch_size)

    # best model 불러오기
    checkpoint = torch.load(args.model_path, map_location=device)
    state_dict = checkpoint.state_dict()
    model.load_state_dict(state_dict)

    model = model.to(device)
    # 추론을 실행하기 전에는 반드시 설정 (batch normalization, dropout 를 평가 모드로 설정)
    # model.eval()

    # sample_submisson.csv 열기
    submission = pd.read_csv(
        './submission/sample_submission.csv', index_col=None)

    # test set에 대한 prediction
    file_names, preds = inference(model, test_loader, device)

    # PredictionString 대입
    for file_name, string in zip(file_names, preds):
        submission = submission.append({"image_id": file_name, "PredictionString": ' '.join(str(e) for e in string.tolist())},
                                       ignore_index=True)

    # submission.csv로 저장
    submission.to_csv(f"./submission/{args.exp_name}.csv", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Semantic Segmentation', parents=[arg_parser_infer()])
    args = parser.parse_args()
    print(args)
    seed_everything(args.seed)
    main(args)
