import os
import warnings 
warnings.filterwarnings('ignore')

import torch
from torch.utils.data import Dataset
import cv2
from pycocotools.coco import COCO

import numpy as np

from utils.annotation import annotation
from transform import transform


def get_classname(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id'] == classID:
            return cats[i]['name']
    return "None"


class CustomDataLoader(Dataset):
    """COCO format"""

    def __init__(self, data_dir, dataset_path, category_names, mode='train', transform=None):
        super().__init__()
        self.mode = mode
        self.transform = transform
        self.category_names = category_names
        self.coco = COCO(data_dir)
        self.dataset_path = dataset_path
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index: int):
        # dataset이 index되어 list처럼 동작
        image_id = self.coco.getImgIds(imgIds=self.ids[index])
        image_infos = self.coco.loadImgs(image_id)[0]

        # cv2 를 활용하여 image 불러오기
        images = cv2.imread(os.path.join(
            self.dataset_path, image_infos['file_name']))
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)

        if (self.mode in ('train', 'val')):
            ann_ids = self.coco.getAnnIds(imgIds=image_infos['id'])
            anns = self.coco.loadAnns(ann_ids)

            # Load the categories in a variable
            cat_ids = self.coco.getCatIds()
            cats = self.coco.loadCats(cat_ids)

            # masks : size가 (height x width)인 2D
            # 각각의 pixel 값에는 "category id" 할당
            # Background = 0
            masks = np.zeros((image_infos["height"], image_infos["width"]))
            # General trash = 1, ... , Cigarette = 10
            anns = sorted(anns, key=lambda idx: idx['area'], reverse=True)
            for i in range(len(anns)):
                className = get_classname(anns[i]['category_id'], cats)
                pixel_value = self.category_names.index(className)
                masks[self.coco.annToMask(anns[i]) == 1] = pixel_value
            masks = masks.astype(np.int8)

            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images, mask=masks)
                images = transformed["image"]
                masks = transformed["mask"]
            return images, masks, image_infos

        if self.mode == 'test':
            # transform -> albumentations 라이브러리 활용
            if self.transform is not None:
                transformed = self.transform(image=images)
                images = transformed["image"]
            return images, image_infos

    def __len__(self) -> int:
        # 전체 dataset의 size를 return
        return len(self.coco.getImgIds())


def create_dataloader(trans, batch_size):
    # train.json / validation.json / test.json 디렉토리 설정
    dataset_path = '../input/data'
    train_path = dataset_path + '/train.json'
    val_path = dataset_path + '/val.json'
    test_path = dataset_path + '/test.json'

    # class (Categories) 에 따른 index 확인 (0~10 : 총 11개)
    sorted_df = annotation(dataset_path)
    category_names = list(sorted_df.Categories)

    # collate_fn needs for batch
    def collate_fn(batch):
        return tuple(zip(*batch))

    # Data Augmentation
    train_transform = transform(trans)

    val_transform = transform(trans)

    test_transform = transform(trans)

    # create own Dataset 1 (skip)
    # validation set을 직접 나누고 싶은 경우
    # random_split 사용하여 data set을 8:2 로 분할
    # train_size = int(0.8*len(dataset))
    # val_size = int(len(dataset)-train_size)
    # dataset = CustomDataLoader(data_dir=train_path, mode='train', transform=transform)
    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # create own Dataset 2
    # train dataset
    train_dataset = CustomDataLoader(data_dir=train_path,
                                     dataset_path=dataset_path,
                                     category_names=category_names,
                                     mode='train',
                                     transform=train_transform)

    # validation dataset
    val_dataset = CustomDataLoader(data_dir=val_path,
                                   dataset_path=dataset_path,
                                   category_names=category_names,
                                   mode='val',
                                   transform=val_transform)

    # test dataset
    test_dataset = CustomDataLoader(data_dir=test_path,
                                    dataset_path=dataset_path,
                                    category_names=category_names,
                                    mode='test',
                                    transform=test_transform)

    # DataLoader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4,
                                               collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=4,
                                             collate_fn=collate_fn)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4,
                                              collate_fn=collate_fn)

    return train_loader, val_loader, test_loader
