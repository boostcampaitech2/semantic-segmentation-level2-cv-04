import torch.nn as nn


def seg_loss(loss_name):

    if loss_name == "CE":
        criterion = nn.CrossEntropyLoss()

    return criterion
