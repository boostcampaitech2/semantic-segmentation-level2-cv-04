from utils.utils import *
import torch
import os
import numpy as np
from tqdm import tqdm

category_names = ['Background','General trash','Paper','Paper pack','Metal','Glass','Plastic','Styrofoam','Plastic bag','Battery','Clothing']

# FIXME 수정해라 이거
def save_model(model, saved_dir, file_name='fcn_resnet50_best_model(pretrained).pt'):
    check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model, output_path)

def train(num_epochs, model, train_loader, val_loader, criterion, optimizer, saved_dir, val_every, device):
    print(f'Start training..')
    n_class = 11
    best_loss = 9999999
    
    for epoch in range(num_epochs):
        model.train()

        pbar = tqdm(train_loader,desc=f"train epoch {epoch} ")

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

            pbar.set_postfix_str(f"Acc: {acc.item():.2f}, AccCls: {acc_cls.item():.2f}, fwavacc: {fwavacc.item():.2f}, Loss: {loss.item():.2f}, mIoU: {mIoU.item():.2f}")

        #TODO 저장주기 바까
        # validation 주기에 따른 loss 출력 및 best model 저장
        if (epoch + 1) % val_every == 0:
            avrg_loss = validation(epoch + 1, model, val_loader, criterion, device)
            if avrg_loss < best_loss:
                print(f"Best performance at epoch: {epoch + 1}")
                print(f"Save model in {saved_dir}")
                best_loss = avrg_loss
                save_model(model, saved_dir)


def validation(epoch, model, train_loader, criterion, device):
    print(f'Start validation #{epoch}')
    model.eval()

    with torch.no_grad():
        n_class = 11
        total_loss = 0
        cnt = 0
        
        hist = np.zeros((n_class, n_class))

        pbar = tqdm(train_loader,desc=f"validating ")
        IoU_by_class = None
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
        
        IoU_by_class = [{classes : round(IoU,4)} for IoU, classes in zip(IoU , category_names)]
        
        avrg_loss = total_loss / cnt
        print(f'Average Loss : {avrg_loss.item():.4f}, IoU by class : {IoU_by_class}')
        
    return avrg_loss
