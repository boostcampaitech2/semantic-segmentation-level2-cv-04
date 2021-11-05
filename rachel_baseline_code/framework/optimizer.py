import torch.optim as optim


def seg_optimizer(opt_name, model, lr):

    if opt_name == 'Adam':
        opt = optim.Adam(params=model.parameters(),
                         lr=lr, weight_decay=1e-6)

    """If you want to add more options...

    elif opt_name == 'optimizer name to use in argparser':
        opt = any optimizer you want to use
    """

    return opt
