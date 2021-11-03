import albumentations as A
from albumentations.pytorch import ToTensorV2

def getTransform():

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

  return train_transform, val_transform


def getInferenceTransform():

  test_transform = A.Compose([
                            # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                           ToTensorV2(),
                           ])
                        
  return test_transform