
import os
import argparse
import torch
import pandas as pd
import numpy as np
import yaml
import ttach as tta

from tools import test
from data_set.data_augmentation import get_transform
from data_set.data_set import CustomDataSet, collate_fn
from model.custom_encoder import register_encoder

import segmentation_models_pytorch as smp

@torch.no_grad()
def inference(cfg, device):
    # custom encoder smp에 등록
    register_encoder()

    # TTA transform
    tta_transforms = tta.Compose(
    [
        tta.HorizontalFlip(),
        tta.VerticalFlip(),
        tta.Scale(scales=[0.5, 0.75, 1.0, 1.25, 1.5])
    ])

    dataset_path  = '../input/data'
    test_path = '../input/data/test.json'
    test_transform = get_transform('test_transform')
    test_dataset = CustomDataSet(data_dir=test_path, dataset_path=dataset_path, mode='test', transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=16,
                                          shuffle=False,
                                          num_workers=4,
                                          collate_fn=collate_fn)

    model_path =  f"{cfg['saved_dir']}/{cfg['exp_name']}/{cfg['exp_name']}.pt"                                    
    
    # model 불러오기
    model = smp.__dict__[cfg['model']['name']]
    model = model(**cfg['model']['params'])
    # best model 불러오기
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.state_dict()
    model.load_state_dict(state_dict)

    model = model.to(device)
    # 추론을 실행하기 전에는 반드시 설정 (batch normalization, dropout 를 평가 모드로 설정)
    model.eval()

    tta_model = tta.SegmentationTTAWrapper(model, tta_transforms, merge_mode='mean')
    # sample_submisson.csv 열기
    submission = pd.read_csv('./submission/sample_submission.csv', index_col=None)

    # test set에 대한 prediction
    file_names, preds = test(tta_model, test_loader, device)

    # PredictionString 대입
    for file_name, string in zip(file_names, preds):
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                    ignore_index=True)

    # submission.csv로 저장
    submission.to_csv(f"./submission/submission_{cfg['exp_name']}.csv", index=False)

    print(f'Inference Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp', type=str, default='3_Crop_Test')
    args = parser.parse_args()

    yaml_path = f"./saved/{args.exp}/{args.exp}.yaml"

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    inference(cfg, device)