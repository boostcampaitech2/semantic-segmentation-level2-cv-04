import torch
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from utils.utils import add_hist, label_accuracy_score
from utils.wandb_method import WandBMethod
from utils.tqdm import TQDM
from utils.save_helper import SaveHelper

def train(num_epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, saved_dir, save_capacity, device, doWandb, patience, object_best, tv):
    n_class = 11
    counter = 0
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

            if doWandb:
                WandBMethod.trainLog(loss, acc, scheduler.get_last_lr())

        val_avrg_loss , val_mIoU, val_IoU = validation(epoch, model, val_loader, criterion, device, doWandb, tv)
        TQDM.setMainPbarPostInValid(mainPbar, val_avrg_loss)

        if saveHelper.checkBestmIoU(val_mIoU, epoch):
            TQDM.setMainPbarDescInSaved(mainPbar, epoch, val_mIoU)
            saveHelper.removeModel()
            saveHelper.saveModel(epoch, model, optimizer)
            counter = 0
        else:
            counter += 1

        if object_best:
            saveHelper.saveBestIoUObject(val_IoU, epoch, object_best, model)
        
        if counter > patience:
            print("Early Stopping...")
            break
        
    saveHelper.renameBestModel()
    
    if object_best:
        saveHelper.renameBestObjectModel(object_best)

    

def validation(epoch, model, valid_loader, criterion, device, doWandb, tv):
    model.eval()
    with torch.no_grad():
        n_class = 11
        total_loss = 0
        cnt = 0
        
        hist = np.zeros((n_class, n_class))

        pbar = TQDM.makePbar(valid_loader,epoch,False)

        for step, (images, masks, _) in enumerate(pbar):
            
            images = torch.stack(images)       
            masks = torch.stack(masks).long()  

            images, masks = images.to(device), masks.to(device)            
            
            # device 할당
            model = model.to(device)
            
            if tv:
                outputs = model(images)['out']
            else:
                outputs = model(images)

            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)
            acc, acc_cls, acc_clsmean, mIoU, fwavacc, IoU = label_accuracy_score(hist)

            TQDM.setPbarPostInStep(pbar,acc,acc_clsmean,loss,mIoU)

        if doWandb:
            WandBMethod.validLog(IoU, acc_cls, acc_clsmean, acc, mIoU, images, outputs, masks)
      
        avrg_loss = total_loss / cnt
        # TQDM.setMainPbarPostInValid(mainPbar,avrg_loss)
    return avrg_loss, mIoU, IoU
