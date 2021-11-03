import wandb

# wandb init
def wandb_init(exp_name):
    wandb.init(project="segmentation", entity = "cv4", reinit=True)
    wandb.run.name = exp_name

# wandb lr 기록
def lr_logger(lr):
    wandb.log({
        "learning rate" : lr
    })

# wandb train 값 기록
def train_logger(loss, mIoU, acc):
    wandb.log({
        "train/loss" : loss.item(),
        "train/mIoU" : mIoU,
        "train/acc" : acc.item()
    })

# wandb valid 값 기록
def valid_logger(mean_acc, acc, acc_cls, mIoU, IoU):
    wandb.log({
        "val/mIoU" : mIoU.item(),
        "val/mAcc" : mean_acc.item(),
        "val/aAcc" : acc.item(),
        "val/Acc.Background" : acc_cls[0],
        "val/Acc.General Trash" : acc_cls[1],
        "val/Acc.Paper" : acc_cls[2],
        "val/Acc.Paper Pack" : acc_cls[3],
        "val/Acc.Metal" : acc_cls[4],
        "val/Acc.Glass" : acc_cls[5],
        "val/Acc.Plastic" : acc_cls[6],
        "val/Acc.Styrofoam" : acc_cls[7],
        "val/Acc.Plastic bag" : acc_cls[8],
        "val/Acc.Battery" : acc_cls[9],
        "val/Acc.Clothing" : acc_cls[10],
        "val/IoU.Background" : IoU[0],
        "val/IoU.General Trash" : IoU[1],
        "val/IoU.Paper" : IoU[2],
        "val/IoU.Paper Pack" : IoU[3],
        "val/IoU.Metal" : IoU[4],
        "val/IoU.Glass" : IoU[5],
        "val/IoU.Plastic" : IoU[6],
        "val/IoU.Styrofoam" : IoU[7],
        "val/IoU.Plastic bag" : IoU[8],
        "val/IoU.Battery" : IoU[9],
        "val/IoU.Clothing" : IoU[10]       
    })