import torch
import numpy as np
from utils.utils import add_hist, label_accuracy_score
from utils.wandb_method import WandBMethod
from utils.tqdm import TQDM
from utils.save_helper import SaveHelper
from torch.cuda.amp import GradScaler, autocast

def train(num_epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, saved_dir, save_capacity, device, doWandb):
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
            
            optimizer.zero_grad()

            with autocast(True):
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # auxiliary head가 포함됐을 때 mask만 추출
            if isinstance(outputs,tuple):
                outputs = outputs[0]    
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)
            acc, acc_cls, acc_clsmean, mIoU, fwavacc, IoU = label_accuracy_score(hist)

            TQDM.setPbarPostInStep(pbar, acc,acc_clsmean,loss,mIoU)

            if doWandb:
                WandBMethod.trainLog(loss, acc, scheduler.get_last_lr())

        avrg_loss ,mIoU = validation(epoch, model, val_loader, criterion, device, doWandb)
        TQDM.setMainPbarPostInValid(mainPbar,avrg_loss)

        if saveHelper.checkBestIoU(mIoU, epoch):
            TQDM.setMainPbarDescInSaved(mainPbar,epoch,mIoU)
            saveHelper.removeModel()
            saveHelper.saveModel(epoch,model,optimizer,scheduler)

        # Scheduler는 epoch당 step
        scheduler.step() 

def validation(epoch, model, valid_loader, criterion, device, doWandb):
    model.eval()
    with torch.no_grad():
        n_class = 11
        total_loss = 0
        total_mIoU = 0
        cnt = 0
        
        hist = np.zeros((n_class, n_class))

        pbar = TQDM.makePbar(valid_loader,epoch,False)


        # len(pbar) 중 랜덤하게 정수 뽑고 해당 step일 때 이미지 그룹 찝어다가 이미지를 doWandb에 넣어줘
        targetStep = WandBMethod.pickImageStep(len(pbar))
        targetImages, targetOutputs, targetMasks = None, None, None

        for step, (images, masks, _) in enumerate(pbar):
            
            images = torch.stack(images)       
            masks = torch.stack(masks).long()  

            images, masks = images.to(device), masks.to(device)            
            
            model = model.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # auxiliary head가 포함됐을 때 mask만 추출
            if isinstance(outputs,tuple):
                outputs = outputs[0]    
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)
            acc, acc_cls, acc_clsmean, mIoU, fwavacc, IoU = label_accuracy_score(hist)


            total_loss += loss
            total_mIoU += mIoU
            cnt += 1

            TQDM.setPbarPostInStep(pbar,acc,acc_clsmean,loss,total_mIoU/cnt)

            # 여러개의 epoch 중 랜덤으로 뽑아 wandb에 전송하는 용도
            if step==targetStep:
                targetImages, targetOutputs, targetMasks = images.detach().cpu().numpy(), outputs, masks

        avrg_loss = total_loss / cnt
        avrg_mIoU = total_mIoU / cnt

        if doWandb:
            WandBMethod.validLog(IoU, acc_cls, acc_clsmean, acc, avrg_mIoU, targetImages, targetOutputs, targetMasks)
      
        
    return avrg_loss, avrg_mIoU
