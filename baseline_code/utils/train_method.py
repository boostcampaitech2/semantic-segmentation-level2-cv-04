from utils.utils import *
import torch
import os
import numpy as np
from tqdm import tqdm
from collections import deque

category_names = ['Back','General','Paper','Pack','Metal','Glass','Plastic','EPS','P-Bag','Battery','Cloth']

def train(num_epochs, model, train_loader, val_loader, criterion, optimizer, saved_dir, save_capacity, device):
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
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)
            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)

            # pbar.update()
            pbar.set_postfix_str(f"Acc: {acc.item():.2f}, AccCls: {acc_cls.item():.2f}, fwavacc: {fwavacc.item():.2f}, Loss: {loss.item():.2f}, mIoU: {mIoU.item():.2f}")


        avrg_loss , mIoU= validation(epoch, model, val_loader, criterion, device, mainPbar)
        if avrg_loss < best_loss:
            mainPbar.set_description_str(f"Last save epoch #{epoch}, mIoU: {mIoU}")
            best_loss = avrg_loss

            savedEpochList.append(epoch)
            bestEpoch = epoch
            if len(savedEpochList) >= max(save_capacity,2):
                delTartget = savedEpochList.popleft()

                # 파일 탐색 후 삭제
                os.remove(os.path.join(saved_dir,f"epoch{delTartget}.pth"))
            
            torch.save(model, os.path.join(saved_dir,f"epoch{epoch}.pth"))
    os.rename(f"epoch{bestEpoch}.pth","best.pth")

def validation(epoch, model, train_loader, criterion, device, mainPbar):
    model.eval()
    with torch.no_grad():
        n_class = 11
        total_loss = 0
        cnt = 0
        
        hist = np.zeros((n_class, n_class))

        IoU_by_class = None
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
            acc, acc_cls, mIoU, fwavacc, IoU = label_accuracy_score(hist)

            pbar.set_postfix_str(f"Acc: {acc.item():.2f}, AccCls: {acc_cls.item():.2f}, fwavacc: {fwavacc.item():.2f}, Loss: {loss.item():.2f}, mIoU: {mIoU.item():.2f}")
        
        IoU_by_class = [{classes : round(IoU,2)} for IoU, classes in zip(IoU , category_names)]       
        avrg_loss = total_loss / cnt
        mainPbar.set_postfix_str(f"Avrg Loss: {avrg_loss.item():.2f}")
        # ", IoU by class : {IoU_by_class}")
        
    return avrg_loss, mIoU
