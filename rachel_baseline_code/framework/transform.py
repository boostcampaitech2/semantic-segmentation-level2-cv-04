import albumentations as A
from albumentations.pytorch import ToTensorV2


def transform(t_name):
    if t_name == 'water':
        t = A.Compose(
            [
                A.Normalize(),
                ToTensorV2(),
            ])

    elif t_name == 'coffee':
        t = A.Compose(
            [
                A.Flip(),
                A.Normalize(),
                ToTensorV2(),
            ])
    
    elif t_name == 'latte':
        t = A.Compose(
            [
                A.OneOf([
                    A.Flip(1.0), 
                    A.RandomRotate90(1.0)
                    ], p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.15, p=0.5),
                A.HueSaturationValue(
                    hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=10, p=0.5),
                A.GaussNoise(p=0.3),
                A.Normalize(),
                ToTensorV2(),
            ])

    elif t_name == 'choco_chip':
        t = A.Compose(
            [
                A.OneOf([
                    A.Flip(1.0), 
                    A.RandomRotate90(1.0)
                    ], p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.15, p=0.5),
                A.HueSaturationValue(
                    hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=10, p=0.5),
                A.Cutout(num_holes=32, max_h_size=50, max_w_size=50, p=0.5),
                A.Normalize(),
                ToTensorV2(),
            ])

    elif t_name == 'frappe':
        t = A.Compose(
            [
                A.OneOf([
                    A.Flip(1.0), 
                    A.RandomRotate90(1.0)
                    ], p=0.5),
                A.RandomResizedCrop(height=512, width=512,
                                    scale=(0.5, 1.0), p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.15, p=0.5),
                A.HueSaturationValue(
                    hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=10, p=0.5),
                A.GaussNoise(p=0.3),
                A.OneOf([
                    A.Blur(p=1.0), 
                    A.GaussianBlur(p=1.0), 
                    A.MedianBlur(blur_limit=5, p=1.0), 
                    A.MotionBlur(p=1.0)
                    ], p=0.1),
                A.Normalize(),
                ToTensorV2(),
            ])

    elif t_name == 'choco_chip_frappe':
        t = A.Compose(
            [
                A.OneOf([
                    A.Flip(1.0), 
                    A.RandomRotate90(1.0)
                    ], p=0.5),
                A.RandomResizedCrop(height=512, width=512,
                                    scale=(0.5, 1.0), p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, contrast_limit=0.15, p=0.5),
                A.HueSaturationValue(
                    hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=10, p=0.5),
                A.GaussNoise(p=0.3),
                A.OneOf([
                    A.Blur(p=1.0), 
                    A.GaussianBlur(p=1.0), 
                    A.MedianBlur(blur_limit=5, p=1.0), 
                    A.MotionBlur(p=1.0)
                    ], p=0.1),
                A.Cutout(num_holes=32, max_h_size=50, max_w_size=50, p=0.5),
                A.Normalize(),
                ToTensorV2(),
            ])

    return t
