import torch
import numpy as np
from utils.utils import add_hist, label_accuracy_score
from utils.wandb_method import WandBMethod
from utils.tqdm import TQDM
from utils.save_helper import SaveHelper
from utils.lovasz_loss import lovasz_softmax, criterion_lovasz_softmax_non_empty

def train(num_epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, saved_dir, save_capacity, device, doWandb):
    n_class = 11
    
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
            
            # inference
            outputs = model(images)
            
            # modified-1
            # outputs, outputs_deeps = model(images)
            
            # loss 계산 (cross entropy loss)
            loss = criterion(outputs, masks)
            
            # modified-2
            # loss = criterion(outputs, masks)
            # loss += lovasz_softmax(outputs, masks)
            # for outputs_deep in outputs_deeps:
            #     loss += 0.1 * criterion_lovasz_softmax_non_empty(criterion, outputs_deep, masks)
                    
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

        avrg_loss , mIoU= validation(epoch, model, val_loader, criterion, device, doWandb)
        TQDM.setMainPbarPostInValid(mainPbar,avrg_loss)

        if saveHelper.checkBestLoss(avrg_loss, epoch):
            TQDM.setMainPbarDescInSaved(mainPbar,epoch,mIoU)
            saveHelper.removeModel()
            saveHelper.saveModel(epoch,model)
            
    # saveHelper.renameBestModel() #생각해보니 best일때만 저장되니까 젤높은숫자가 best임

def validation(epoch, model, valid_loader, criterion, device, doWandb):
    model.eval()
    with torch.no_grad():
        n_class = 11
        total_loss = 0
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
            
            # device 할당
            model = model.to(device)
            
            outputs = model(images)
            
            # modified-3
            # outputs, outputs_deeps = model(images)
            
            loss = criterion(outputs, masks)
            
            # modified -4
            # loss = criterion(outputs, masks)
            # loss += lovasz_softmax(outputs, masks)
            # for outputs_deep in outputs_deeps:
            #     loss += 0.1 * criterion_lovasz_softmax_non_empty(criterion, outputs_deep, masks)
                
            total_loss += loss
            cnt += 1
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)
            acc, acc_cls, acc_clsmean, mIoU, fwavacc, IoU = label_accuracy_score(hist)

            TQDM.setPbarPostInStep(pbar,acc,acc_clsmean,loss,mIoU)
            if step==targetStep:
                targetImages, targetOutputs, targetMasks = images.detach().cpu().numpy(), outputs, masks

        if doWandb:
            WandBMethod.validLog(IoU, acc_cls, acc_clsmean, acc, mIoU, targetImages, targetOutputs, targetMasks)
      
        avrg_loss = total_loss / cnt
        
    return avrg_loss, mIoU