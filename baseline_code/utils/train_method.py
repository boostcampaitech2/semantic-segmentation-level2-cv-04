import random
from utils.utils import *
import torch
import os
import numpy as np
from tqdm import tqdm
from collections import deque
import wandb

category_names = ['Background','General trash','Paper','Paper pack','Metal','Glass','Plastic','Styrofoam','Plastic bag','Battery','Clothing']

def train(num_epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, saved_dir, save_capacity, device, doWandb):
    n_class = 11
    best_loss = 9999999
    
    savedEpochList = deque()
    bestEpoch=0
    # IoU_by_class = [{classes : round(IoU,2)} for IoU, classes in zip([0 for i in range(len(category_names))], category_names)]  
    mainPbar = tqdm(range(num_epochs), desc="Start Training")

    for epoch in mainPbar:
        model.train()
        pbar = tqdm(train_loader, desc=f"Train #{epoch} ")

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
            pbar.set_postfix_str(f"Acc: {acc.item():.2f}, AccCls: {acc_clsmean.item():.2f}, fwavacc: {fwavacc.item():.2f}, Loss: {loss.item():.2f}, mIoU: {mIoU.item():.2f}")

            if doWandb:
                wandb.log({"train/loss":loss.item(),"train/decode.loss_ce":loss.item(),"train/decode.acc_seg":acc.item(), "learning_rate":scheduler.get_last_lr()})


        avrg_loss , mIoU= validation(epoch, model, val_loader, criterion, device, mainPbar, doWandb)
        if avrg_loss < best_loss:
            mainPbar.set_description_str(f"Last save epoch #{epoch}, mIoU: {mIoU:.3f}")
            best_loss = avrg_loss

            savedEpochList.append(epoch)
            bestEpoch = epoch
            if len(savedEpochList) > max(save_capacity,2):
                delTartget = savedEpochList.popleft()

                # 파일 탐색 후 삭제
                os.remove(os.path.join(saved_dir,f"epoch{delTartget}.pth"))
            
            torch.save(model.state_dict(), os.path.join(saved_dir,f"epoch{epoch}.pth"))
    os.rename(f"epoch{bestEpoch}.pth","best.pth")

def validation(epoch, model, train_loader, criterion, device, mainPbar, doWandb):
    model.eval()
    with torch.no_grad():
        n_class = 11
        total_loss = 0
        cnt = 0
        
        hist = np.zeros((n_class, n_class))

        pbar = tqdm(train_loader,desc=f"Validation #{epoch} ")
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

            pbar.set_postfix_str(f"Acc: {acc.item():.2f}, AccCls: {acc_clsmean.item():.2f}, fwavacc: {fwavacc.item():.2f}, Loss: {loss.item():.2f}, mIoU: {mIoU.item():.2f}")

        if doWandb:
            randIdx = random.randint(0,len(images)-1)
            wandb.log({
                "val/IoU.Background":IoU[0],
                "val/IoU.Battery":IoU[10],
                "val/IoU.Clothing":IoU[9],
                "val/IoU.General trash":IoU[1],
                "val/IoU.Glass":IoU[5],
                "val/IoU.Metal":IoU[4],
                "val/IoU.Paper":IoU[2],
                "val/IoU.Paper pack":IoU[3],
                "val/IoU.Plastic":IoU[6],
                "val/IoU.Plastic bag":IoU[8],
                "val/IoU.Styrofoam":IoU[7],
                "val/Acc.Background":acc_cls[0],
                "val/Acc.Battery":acc_cls[10],
                "val/Acc.Clothing":acc_cls[9],
                "val/Acc.General trash":acc_cls[1],
                "val/Acc.Glass":acc_cls[5],
                "val/Acc.Metal":acc_cls[4],
                "val/Acc.Paper":acc_cls[2],
                "val/Acc.Paper pack":acc_cls[3],
                "val/Acc.Plastic":acc_cls[6],
                "val/Acc.Plastic bag":acc_cls[8],
                "val/Acc.Styrofoam":acc_cls[7],
                "val/aAcc":acc_clsmean.item(),
                "val/mAcc":acc.item(),
                "val/mIoU":mIoU.item(),
                "image" : wandb.Image(images[randIdx], masks={
                    "predictions" : {
                        "mask_data" : outputs[randIdx],
                        "class_labels":{i:category for i, category in enumerate(category_names)}
                    },
                    "ground_truth" : {
                        "mask_data" : masks[randIdx],
                        "class_labels":{i:category for i, category in enumerate(category_names)}
                    }}),
                })
      
        avrg_loss = total_loss / cnt
        mainPbar.set_postfix_str(f"Avrg Loss: {avrg_loss.item():.2f}")
        # ", IoU by class : {IoU_by_class}")
        
    return avrg_loss, mIoU
