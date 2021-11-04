import torch.optim as optim


def seg_optimizer(opt_name, model, lr):

    if opt_name == 'Adam':
        opt = optim.Adam(params=model.parameters(),
                        lr=lr, weight_decay=1e-6)

    return opt
