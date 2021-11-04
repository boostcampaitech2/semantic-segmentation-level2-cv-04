import torch
import numpy as np
from tqdm import tqdm
import albumentations as A

def test(model, data_loader, device, size):
	model.to(device)
	transform = A.Compose([A.Resize(size, size)])
	print('Start prediction.')
	
	model.eval()
	
	file_name_list = []
	preds_array = np.empty((0, size*size), dtype=np.compat.long)
	
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