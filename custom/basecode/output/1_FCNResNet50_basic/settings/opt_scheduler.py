import torch

#https://sanghyu.tistory.com/113
#https://gaussian37.github.io/dl-pytorch-lr_scheduler/

def getOptAndScheduler(model, lr):

	optimizer = torch.optim.Adam(params = model.parameters(), lr=lr, weight_decay=1e-5)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=10,eta_min=1e-5)

	return optimizer, scheduler	