import torch

#https://sanghyu.tistory.com/113
#https://gaussian37.github.io/dl-pytorch-lr_scheduler/

def getOptAndScheduler(model, lr):

	optimizer = torch.optim.Adam(params = model.parameters(), lr=lr, weight_decay=1e-5)
	scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, pct_start=0.05, steps_per_epoch=1, epochs=40)
	return optimizer, scheduler	