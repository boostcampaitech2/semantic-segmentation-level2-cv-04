import albumentations as A
from albumentations.pytorch import ToTensorV2

def getTransform():

  train_transform = A.Compose([
                              # Basic
                              A.RandomRotate90(p=1),
                              A.HorizontalFlip(p=0.5),
      
                              # Morphology
                              A.ShiftScaleRotate(shift_limit=0, scale_limit=(-0.2,0.2), rotate_limit=(-30,30), 
                                                  interpolation=1, border_mode=0, value=(0,0,0), p=0.5),
                              A.GaussNoise(var_limit=(0,50.0), mean=0, p=0.5),
                              A.GaussianBlur(blur_limit=(3,7), p=0.5),
      
                              # Color
                              A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.5, 
                                                          brightness_by_max=True,p=0.5),
                              A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=30, 
                                                      val_shift_limit=0, p=0.5),
                              # A.Flip(),
                              # A.RandomCrop(width=256, height=256),
                              # A.RandomBrightnessContrast(p=0.2),
                              A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                              ToTensorV2(),
                              ])

  val_transform = A.Compose([
                            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ToTensorV2(),
                            ])

  return train_transform, val_transform


def getInferenceTransform():

  test_transform = A.Compose([
                            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                           ToTensorV2(),
                           ])
                        
  return test_transform