from utils.utils import *
import torch
import os
import numpy as np
from collections import deque
from utils.wandb_method import WandBMethod
from utils.tqdm import TQDM

category_names = ['Background','General trash','Paper','Paper pack','Metal','Glass','Plastic','Styrofoam','Plastic bag','Battery','Clothing']

def train(num_epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, saved_dir, save_capacity, device, doWandb):
    n_class = 11
    best_loss = 9999999
    
    savedEpochList = deque()
    bestEpoch=0
    # IoU_by_class = [{classes : round(IoU,2)} for IoU, classes in zip([0 for i in range(len(category_names))], category_names)]  
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
        
        if avrg_loss < best_loss:
            TQDM.setMainPbarDescInSaved(mainPbar,epoch,mIoU)
            best_loss = avrg_loss

            savedEpochList.append(epoch)
            bestEpoch = epoch
            if len(savedEpochList) > max(save_capacity,2):
                delTartget = savedEpochList.popleft()

                # 파일 탐색 후 삭제
                os.remove(os.path.join(saved_dir,f"epoch{delTartget}.pth"))
            
            torch.save(model.state_dict(), os.path.join(saved_dir,f"epoch{epoch}.pth"))
    os.rename(f"epoch{bestEpoch}.pth","best.pth")

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
            WandBMethod.validLog(IoU, acc_cls, acc_clsmean, acc, mIoU, images, outputs, masks, {i:category for i, category in enumerate(category_names)})
      
        avrg_loss = total_loss / cnt
        TQDM.setMainPbarPostInValid(mainPbar,avrg_loss)
    return avrg_loss, mIoU
