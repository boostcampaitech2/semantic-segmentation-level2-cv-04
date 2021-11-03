from torch import nn

def getLoss():
	criterion = nn.CrossEntropyLoss()

	return criterion