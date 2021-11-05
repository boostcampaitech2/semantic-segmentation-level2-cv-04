import torch.nn as nn


def seg_loss(loss_name):

    if loss_name == "CE":
        criterion = nn.CrossEntropyLoss()
    
    """If you want to add more options...

    elif loss_name == 'loss name to use in argparser':
        criterion = any loss you want to use
    """

    return criterion
