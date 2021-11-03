from torch import nn

def getLoss():
	criterion = nn.CrossEntropyLoss(reduce=False)

	return criterion