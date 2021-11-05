import warnings 
import argparse
from utils.arg_parser import arg_parser_infer
warnings.filterwarnings('ignore')

import torch

import numpy as np
import pandas as pd
from tqdm import tqdm

from framework.models import seg_model
from framework.dataset import create_dataloader
from utils.utils import seed_everything

import albumentations as A


def inference(model, test_loader, device, is_aux):
    size = 256
    transform = A.Compose([A.Resize(size, size)])
    print('Start prediction.')

    model.eval()

    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)

    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(tqdm(test_loader)):

            # inference (512 x 512)
            if is_aux:
                outs, labels = model(torch.stack(imgs).to(device))
            else:
                outs = model(torch.stack(imgs).to(device))
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

    model, is_aux = seg_model(args.model)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, _, test_loader = create_dataloader(args.tta, args.batch_size)

    # best model 불러오기
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)

    model = model.to(device)

    # sample_submisson.csv 열기
    submission = pd.read_csv(
        './submission/sample_submission.csv', index_col=None)

    # test set에 대한 prediction
    file_names, preds = inference(model, test_loader, device, is_aux)

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