import torch
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from utils.utils import add_hist, label_accuracy_score
from utils.tqdm import TQDM
from utils.save_helper import SaveHelper

def train(num_epochs, model, train_loader, criterion, optimizer, scheduler, saved_dir, save_capacity, device, tv):
    n_class = 11
    scaler = GradScaler(enabled=True)
    
    saveHelper = SaveHelper(save_capacity, saved_dir)
    mainPbar = TQDM.makeMainProcessBar(num_epochs)

    for epoch in mainPbar:
        model.train()
        pbar = TQDM.makePbar(train_loader, epoch, True)

        hist = np.zeros((n_class, n_class))
        for step, (images, masks, _) in enumerate(pbar):
            images = torch.stack(images)       
            masks = torch.stack(masks).long() 
            
            # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device)

            # device 할당
            model = model.to(device)
            
            with autocast(enabled=True):
            # inference
                if tv:
                    outputs = model(images)['out']
                else:
                    outputs = model(images)
                
                # loss 계산 (cross entropy loss)
                loss = criterion(outputs, masks)

            # Mixed-Precision 사용 (FP16 이용)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            # defalut precision 사용 (FP32)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            # scheduler.step()
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)
            acc, acc_cls, acc_clsmean, mIoU, fwavacc, IoU = label_accuracy_score(hist)

            # pbar.update()
            TQDM.setPbarPostInStep(pbar, acc,acc_clsmean,loss,mIoU)

    saveHelper.saveModel(epoch, model, optimizer)
    saveHelper.renameBestModel()