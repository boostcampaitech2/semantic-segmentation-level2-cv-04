"""
pstage level1 image classification baseline code 참고
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2

TrainAugmentation = A.Compose([
            A.Normalize(),
            ToTensorV2()
        ])

ValidAugmentation = A.Compose([
            A.Normalize(),
            ToTensorV2()
        ])

TestAugmentation = A.Compose([
            A.Normalize(), 
            ToTensorV2()
        ])

ALLAugmentation = A.Compose([
            A.OneOf([
                A.Flip(p=1.0),
                A.RandomRotate90(p=1.0),
                A.ShiftScaleRotate(p=1.0),
            ], p=0.75),
            A.RandomResizedCrop(512, 512, scale=(0.75, 1.0), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.15, p=0.5),
            A.GaussNoise(p=0.3),
            A.OneOf([
                A.Blur(p=1.0),
                A.GaussianBlur(p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
                A.MotionBlur(p=1.0),
            ], p=0.1),
            A.Normalize(), 
            ToTensorV2()
        ])

transform_entrypoints = {
    'train_transform' : TrainAugmentation,
    'valid_transform' : ValidAugmentation,
    'test_transform' : TestAugmentation,
    'all_transform' : ALLAugmentation
}

# transform 이름 존재하는지 확인
def is_transform(transform_name):
    return transform_name in transform_entrypoints

# get transform
def get_transform(transform_name):
    if is_transform(transform_name):
        transform = transform_entrypoints[transform_name]
    else:
        raise RuntimeError('Unknown transform (%s)' % transform_name)
    return transform
