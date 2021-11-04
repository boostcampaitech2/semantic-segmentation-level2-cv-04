import torch
import pandas as pd
import argparse
import os
from importlib import import_module

from dataset.base_dataset import CustomDataset
from utils.inference_method_ensemble import test

def getArgument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--custom_name',type=str ,required=True)
    parser.add_argument('--tta',type=bool, default=False)
    return parser.parse_known_args()[0].custom_name, parser.parse_known_args()[0].tta

def main(custom_name, tta):
    arg = getattr(import_module(f"output.{custom_name}.sample"), "getArg")()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # tta 적용 시, batch 사이즈 줄이기 (cuda out of memory 발생)
    if tta:
        arg.batch = 4

    test_transform = getattr(import_module("main.transform"), "get_transfrom")(arg.inference_transform)
    test_dataset = CustomDataset(data_dir=os.path.join(arg.image_root,arg.test_json),image_root=arg.image_root, mode='test', transform=test_transform)
    testLoader = getattr(import_module("main.dataloader"), "getTestDataloader")(
		test_dataset, arg.batch, arg.test_worker)
    
    # best model 저장된 경로 (save_helper 수정 전)
    # model_path = os.path.join(arg.output_path, custom_dir, "best.pth")

    # best model 저장된 경로 (save_helper 수정 후)
    for name in os.listdir(f"./output/{arg.custom_name}"):
        if 'best' in name:
            best_model = name
        elif 'general_trash' in name:
            general_trash = name
        elif 'paper_pack' in name:
            paper_pack = name
        elif 'battery' in name:
            battery = name
        elif 'clothing' in name:
            clothing = name

    model_path_list = []
    for model in [best_model, general_trash, paper_pack, battery, clothing]:
        model_path_list.append(os.path.join(arg.output_path, arg.custom_name, model))

    model_list = []
    
    for i in range(len(model_path_list)):
        checkpoint = torch.load(model_path_list[i], map_location=device)
        model = getattr(import_module("main.model"), arg.model)().get_backbone(arg.backbone)
        model.load_state_dict(checkpoint['state_dict'])
        model_list.append(model)

    if tta:
        for i in range(len(model_list)):
            model_list[i] = getattr(import_module("utils.tta"), "get_tta")(model_list[i])

        arg.custom_name = arg.custom_name + "_tta"

    for i in range(len(model_list)):
        model_list[i] = model_list[i].to(device)
    
    # submission 파일 생성
    # sample_submisson.csv 열기
    submission = pd.read_csv('./submission/sample_submission.csv', index_col=None)

    # test set에 대한 prediction
    file_names, preds = test(model_list, testLoader, device)

    # PredictionString 대입
    for file_name, string in zip(file_names, preds):
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                    ignore_index=True)

    # submission.csv로 저장
    submission.to_csv(f"./submission/{arg.custom_name}_classEnsemble.csv", index=False)

if __name__=="__main__":
	custom_name, tta = getArgument()
	main(custom_name, tta)


