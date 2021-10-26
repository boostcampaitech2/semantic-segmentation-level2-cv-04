import torch
import argparse
import os
import shutil
from importlib import import_module

from torch.serialization import save

from dataset.base_dataset import CustomDataset

def getArgument():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dir',type=str ,required=True)
	parser.add_argument('--model', type=str, required=True)

	args = parser.parse_known_args()[0]

	return args.dir, args.model 


def main(custom_dir, model_name):
	
	device = "cuda" if torch.cuda.is_available() else "cpu"

	if os.path.isdir("./tempSettings"):
		shutil.rmtree("./tempSettings")
	shutil.copytree(f"../output/{custom_dir}/settings/", "./tempSettings/")

	arg = getattr(import_module(f"tempSettings.arg"), "getArg")()


	transfroms = getattr(import_module(f"tempSettings.transform"), "getInferenceTransform")()
	dataset = CustomDataset(data_dir=addPath([arg.image_root,arg.test_json]),image_root=arg.image_root, mode="test", transform=transfroms)
	model = getattr(import_module(f"tempSettings.model"), "getModel")()
	dataloader = getattr(import_module(f"tempSettings.dataloader"), "getInferenceDataloader")(dataset, arg.batch, arg.test_worker)

	stateDict = torch.load(f"../output/{custom_dir}/{model_name}.pth", map_location=device)
	model.load_state_dict(stateDict)
	
	file_names, preds_array = test(model, dataloader, device)
	saveSubmission(file_names,preds_array,arg.output_path, arg.custom_name)

	shutil.rmtree("./tempSettings")

import numpy as np
from tqdm import tqdm
import albumentations as A

def test(model, data_loader, device):
	model.to(device)
	size = 256
	transform = A.Compose([A.Resize(size, size)])
	print('Start prediction.')
	
	model.eval()
	
	file_name_list = []
	preds_array = np.empty((0, size*size), dtype=np.long)
	
	with torch.no_grad():
			for step, (imgs, image_infos) in enumerate(tqdm(data_loader)):
					
					# inference (512 x 512)
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

import pandas as pd
def saveSubmission(file_names, preds_array, output_path, custom_name):
	submission = pd.read_csv(f'{output_path}/{custom_name}/submission/sample_submission.csv', index_col=None)	
	
	for file_name, string in zip(file_names, preds_array):
		submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                   ignore_index=True)

	submission.to_csv(f"{output_path}/{custom_name}/submission/submission.csv", index=False)


def addPath(pathList):
	return os.path.join(*pathList)

if __name__=="__main__":
	custom_dir, model_name = getArgument()
	main(custom_dir, model_name)