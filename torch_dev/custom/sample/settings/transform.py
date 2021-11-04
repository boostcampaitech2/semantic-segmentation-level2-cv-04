import albumentations as A
from albumentations.pytorch import ToTensorV2
import ttach as tta
# https://albumentations-demo.herokuapp.com/

def getTransform():

  train_transform = A.Compose([
                              A.Flip(),
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

  tta_transform = tta.Compose(
    [
        tta.HorizontalFlip(),
        tta.Rotate90(angles=[0, 180]),
        tta.Scale(scales=[1, 2, 4]),
        tta.Multiply(factors=[0.9, 1, 1.1]),        
    ]
)

                        
  return test_transform, tta_transform