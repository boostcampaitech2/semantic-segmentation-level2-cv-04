import albumentations as A
from albumentations.augmentations.transforms import Normalize
from albumentations.pytorch import ToTensorV2

def transform(t_name):
    if t_name == 'water':
        t = A.Compose(
            [
                A.Normalize(),
                ToTensorV2(),
            ])
        return t

    elif t_name == 'coffee':
        t = A.Compose(
            [
                A.CenterCrop(height=400, width=384),
                A.Resize(224, 224),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.Normalize(),
                ToTensorV2(),
            ])
        return t
