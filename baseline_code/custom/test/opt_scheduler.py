import torch

def getOptAndScheduler(model, lr):

	optimizer = torch.optim.Adam(params = model.parameters(), lr=lr, weight_decay=1e-5)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=100,eta_min=1e-4)

	return optimizer, scheduler	