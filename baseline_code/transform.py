import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
                            A.Flip(),
                            A.RandomCrop(width=256, height=256),
                            A.RandomBrightnessContrast(p=0.2),
                            # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ToTensorV2(),
                            ])

val_transform = A.Compose([
                            # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                          ToTensorV2(),
                          ])

test_transform = A.Compose([
                            # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                           ToTensorV2(),
                           ])