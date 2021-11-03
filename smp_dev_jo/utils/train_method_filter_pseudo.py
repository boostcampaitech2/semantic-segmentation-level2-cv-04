import torch
import numpy as np
from utils.utils import add_hist, label_accuracy_score
from utils.wandb_method import WandBMethod
from utils.tqdm import TQDM
from utils.save_helper import SaveHelper
import os
import pickle
import gzip
import glob

def train(num_epochs, model, train_loader, val_loader, criterion, optimizer, scheduler, saved_dir, save_capacity, device, doWandb):
    n_class = 11
    
    saveHelper = SaveHelper(save_capacity, saved_dir)
    mainPbar = TQDM.makeMainProcessBar(num_epochs)

    # pickle 파일을 저장하기 위한 폴더 생성
    picklefolder = os.path.join(saved_dir, "pickle_folder")
    if not os.path.exists(picklefolder):
        os.mkdir(picklefolder)
    
    # 매 배치마다 filtering할 갯 수
    filter_num = 2
    
    for epoch in mainPbar:
        model.train()
        pbar = TQDM.makePbar(train_loader, epoch, True)

        hist = np.zeros((n_class, n_class))
        
        # pickle folder epoch별로 생성
        pickle_epoch_folder = os.path.join(picklefolder, "{}".format(epoch))
        if not os.path.exists(pickle_epoch_folder):
            os.mkdir(pickle_epoch_folder)
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
            loss = torch.mean(loss, axis=(1,2)) # 16, 512, 512에서 512, 512를 reduction
            # [2.8536, 2.9212, 2.6025, 2.8111, 2.8378, 2.8253, 2.5458, 2.7472, 2.9457, 2.4796, 2.8635, 2.8808, 2.6545, 2.5723, 2.8518, 2.7062]
            _, loss_rank2 = torch.topk(loss, filter_num) # 2개 미 반영 예시
            loss[loss_rank2] = 0 # loss를 날려!
            # [2.8536, 0.0000, 2.6025, 2.8111, 2.8378, 2.8253, 2.5458, 2.7472, 0.0000, 2.4796, 2.8635, 2.8808, 2.6545, 2.5723, 2.8518, 2.7062],
            loss = torch.mean(loss) # 이렇게 할 경우, 14/16이 반영됨, 나머지 2/16은 Pseudo labeling 후 학습시 반영
            
            # pickle파일로 이미지 저장
            save_pickle_path = os.path.join(pickle_epoch_folder, "{}.pickle".format(step)) # epoch 폴더 내 배치 순서로 저장
            select_images = images[loss_rank2] #  torch.Size([2, 3, 512, 512])
            select_images = select_images.detach().cpu().numpy()
            data = {'images' : select_images}
            with gzip.open(save_pickle_path, 'wb') as f:
                pickle.dump(data, f)
            
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
            
    # pickle 파일 하나씩 불러와서 모델로 예측 후, 예측한 마스크를 pikle에 추가로 저장 (Pesudo-labeling 과정)
    print("Pesudo Labeling Start")
    for epoch in range(num_epochs):
        print("{} / {}".format(epoch + 1, num_epochs))
        pickle_epoch_folder = os.path.join(picklefolder, "{}".format(epoch))
        steps_pickle = glob.glob(os.path.join(pickle_epoch_folder, '*.pickle'))
        for pkl_file in steps_pickle:
            data = '' # init
            with gzip.open(pkl_file,'rb') as f:
                data = pickle.load(f) # data = {'images' : select_images}
                
            # 불러온 이미지에 대해 마스크 이미지 생성
            model.eval()
            with torch.no_grad():
                images = torch.from_numpy(data['images'])

                # gpu 연산을 위해 device 할당
                images = images.to(device)

                # device 할당
                model = model.to(device)

                # inference
                outputs = model(images)
                
                # 마스크화 및 피클로 저장
                outputs = torch.argmax(outputs, dim=1).detach().cpu().numpy()
                data['masks'] = outputs
                with gzip.open(pkl_file, 'wb') as f: # 마스크 추가 # data = {'images' : select_images, 'masks' : match_masks}
                    pickle.dump(data, f)
            
    # pickle 에서 이미지와 마스크 페어를 가져와 추가학습을 진행
    print("Pseudo Train Start")
    for epoch in range(num_epochs):
        model.train()
        print("{} / {}".format(epoch + 1, num_epochs))
        pickle_epoch_folder = os.path.join(picklefolder, "{}".format(epoch))
        steps_pickle = glob.glob(os.path.join(pickle_epoch_folder, '*.pickle'))
        for pkl_file in steps_pickle:
            data = '' # init
            with gzip.open(pkl_file,'rb') as f:
                data = pickle.load(f) # data = {'images' : select_images}
                
            images = torch.from_numpy(data['images'])
            masks = torch.from_numpy(data['masks'])

            # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device)

            # device 할당
            model = model.to(device)

            # inference
            outputs = model(images)

            # loss 계산 (cross entropy loss)
            loss = criterion(outputs, masks)
            loss = torch.mean(loss, axis=(0,1,2)) # reduction
            loss = loss * 0.125 # 나머지 2/16을 반영 (변경해줄 것 filter_num/batch_size로)

            # loss 반영
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
                
        # validation 셋으로 확인해줘야하는데 ,, 흠 validation이 없네
        # 일단 저장
        torch.save(model.state_dict(), os.path.join(saved_dir, 'Pseudo_epoch{}.pth'.format(epoch)))

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
            
            loss = criterion(outputs, masks)
            loss = torch.mean(loss, axis=(0,1,2)) # reduce를 진행안하므로 해주어야함
                
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