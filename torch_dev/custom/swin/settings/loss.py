from torch import nn
from segmentation_models_pytorch.losses.soft_ce import SoftCrossEntropyLoss
from segmentation_models_pytorch.losses.jaccard import JaccardLoss 

def getLoss():
	criterion = JaccardSoftCE()

	return criterion

class JaccardSoftCE(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.jaccard_loss = JaccardLoss(mode='multiclass')
        self.soft_ce = SoftCrossEntropyLoss(smooth_factor=0.1)

    def forward(self, outputs, masks):
        return self.jaccard_loss(outputs, masks) + self.soft_ce(outputs, masks)