import torch
import numpy as np
from utils.utils import add_hist, label_accuracy_score
from utils.wandb_method import WandBMethod
from utils.tqdm import TQDM
from utils.save_helper import SaveHelper

def train(num_epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, saved_dir, save_capacity, device, doWandb):
    n_class = 11
    
    saveHelper = SaveHelper(save_capacity)
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
            
            # inference
            outputs = model(images)
            
            # loss 계산 (cross entropy loss)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)
            acc, acc_cls, acc_clsmean, mIoU, fwavacc, IoU = label_accuracy_score(hist)

            # pbar.update()
            TQDM.setPbarPostInStep(pbar, acc,acc_clsmean,loss,mIoU)

            if doWandb:
                WandBMethod.trainLog(loss, acc, scheduler.get_last_lr())

        avrg_loss , mIoU= validation(epoch, model, val_loader, criterion, device, mainPbar, doWandb)
        
        if saveHelper.checkBestLoss(avrg_loss, epoch):
            TQDM.setMainPbarDescInSaved(mainPbar,epoch,mIoU)
            saveHelper.removeModel(saved_dir)
            saveHelper.saveModel(epoch,saved_dir,model)
            
    saveHelper.renameBestModel()

def validation(epoch, model, valid_loader, criterion, device, mainPbar, doWandb):
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
        TQDM.setMainPbarPostInValid(mainPbar,avrg_loss)
    return avrg_loss, mIoU
