import os
import warnings
warnings.filterwarnings('ignore')

import torch
from utils.utils import label_accuracy_score, add_hist
from torch.cuda.amp import GradScaler, autocast

from utils.logger import make_logger
from utils.wandb import WandBMethod
from utils.tqdm import TQDM

import numpy as np

def save_model(model, saved_dir, file_name):
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model.state_dict(), output_path)

def train(num_epochs, model, data_loader, val_loader, criterion, optimizer, lr_scheduler, saved_dir, file_name, device, sorted_df, doWandb, is_aux, val_every=1):
    logger = make_logger(saved_dir, "train")

    logger.info(f'Start training..')
    n_class = 11
    best_loss = 9999999
    best_mIoU = 0

    # mixed-precision
    scaler = GradScaler(enabled=True)

    # set up tqdm
    mainPbar = TQDM.makeMainProcessBar(num_epochs)

    for epoch in mainPbar:
        model.train()
        pbar = TQDM.makePbar(data_loader, epoch, True)

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
                if is_aux:
                    outputs, _ = model(images)
                else:
                    outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            lr_scheduler.step()
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)
            acc, acc_cls, acc_clsmean, mIoU, fwavacc, IoU = label_accuracy_score(hist)
            
            # pbar.update()
            TQDM.setPbarPostInStep(pbar, acc,acc_clsmean,loss,mIoU)

            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                logger.debug(
                    f'Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(data_loader)}], Loss: {round(loss.item(),4)}, mIoU: {round(mIoU,4)}')
            
            # logging to wandb
            if doWandb:
                WandBMethod.trainLog(loss, acc, lr_scheduler.get_last_lr())

        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            avrg_loss, mIoU_val = validation(epoch + 1, model, val_loader, criterion, saved_dir, device, sorted_df, doWandb, is_aux)
            if mIoU_val > best_mIoU:
                logger.info(f"Best performance at epoch: {epoch + 1}")
                logger.info(f"Save model in {saved_dir}")
                best_mIoU = mIoU_val
                save_model(model, saved_dir, file_name)


def validation(epoch, model, data_loader, criterion, saved_dir, device, sorted_df, doWandb, is_aux):
    logger = make_logger(saved_dir, "validation")
    
    logger.info(f'Start validation #{epoch}')
    model.eval()

    with torch.no_grad():
        n_class = 11
        total_loss = 0
        total_mIoU = 0
        cnt = 0
        
        hist = np.zeros((n_class, n_class))

        pbar = TQDM.makePbar(data_loader,epoch,False)

        targetStep = WandBMethod.pickImageStep(len(pbar))
        targetImages, targetOutputs, targetMasks = None, None, None

        for step, (images, masks, _) in enumerate(pbar):
            
            images = torch.stack(images)       
            masks = torch.stack(masks).long()  

            images, masks = images.to(device), masks.to(device)            
            
            # device 할당
            model = model.to(device)
            
            if is_aux:
                outputs, _ = model(images)
            else:
                outputs = model(images)
            loss = criterion(outputs, masks)
            
            outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            masks = masks.detach().cpu().numpy()
            
            hist = add_hist(hist, masks, outputs, n_class=n_class)
            acc, acc_cls, acc_clsmean, mIoU, fwavacc, IoU = label_accuracy_score(hist)

            total_loss += loss
            total_mIoU += mIoU
            cnt += 1

            TQDM.setPbarPostInStep(pbar,acc,acc_clsmean,loss,total_mIoU/cnt)
            
            if step==targetStep:
                targetImages, targetOutputs, targetMasks = images.detach().cpu().numpy(), outputs, masks
        
        IoU_by_class = [{classes : round(IoU,4)} for IoU, classes in zip(IoU , sorted_df['Categories'])]
        
        avrg_loss = total_loss / cnt
        avrg_mIoU = total_mIoU / cnt

        logger.info(f'Validation #{epoch}  Average Loss: {round(avrg_loss.item(), 4)}, Accuracy : {round(acc, 4)}, mIoU: {round(mIoU, 4)}')
        logger.info(f'IoU by class : {IoU_by_class}')
        
        if doWandb:
            WandBMethod.validLog(IoU, acc_cls, acc_clsmean, acc, avrg_mIoU, targetImages, targetOutputs, targetMasks)
        
    return avrg_loss, mIoU
