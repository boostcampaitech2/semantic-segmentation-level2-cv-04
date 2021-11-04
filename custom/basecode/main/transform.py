import albumentations as A
from albumentations.pytorch import ToTensorV2

def base_transform():

    train_transform = A.Compose([
                                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ToTensorV2(),
                                ])

    val_transform = A.Compose([
                                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                              ToTensorV2(),
                              ])

    return train_transform, val_transform

def rotate30_transform():
    train_transform = A.Compose([
                                A.OneOf([
                                            A.Rotate(p=1.0, limit=(-30, 30))
                                        ], p=0.5),
                                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ToTensorV2(),
                                ])

    val_transform = A.Compose([
                                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ToTensorV2(),
                                ])

    return train_transform, val_transform


def randomresizedcrop_transform():
    train_transform = A.Compose([
                                A.RandomResizedCrop(512, 512, scale=(0.75, 1.0), p=0.5),
                                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ToTensorV2()
                                ])

    val_transform = A.Compose([
                                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ToTensorV2(),
                                ])

    return train_transform, val_transform


def clahe_transform():
    train_transform = A.Compose([
                                A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=0.5),
                                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ToTensorV2()
                                ])

    val_transform = A.Compose([
                                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ToTensorV2(),
                                ])

    return train_transform, val_transform


def flip_transform():

    train_transform = A.Compose([
                                A.OneOf([
                                            A.Flip(p=1.0),
                                            A.RandomRotate90(p=1.0),
                                            # A.ShiftScaleRotate(p=1.0),
                                            A.Rotate(p=1.0, limit=(-30, 30))
                                        ], p=0.5),
                                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ToTensorV2(),
                                ])

    val_transform = A.Compose([
                                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ToTensorV2(),
                                ])

    return train_transform, val_transform


def recommend_transform():

    train_transform = A.Compose([
                                A.Rotate(limit=(-30, 30), p=0.5),
                                A.RandomResizedCrop(512, 512, scale=(0.75, 1.0), p=0.5),
                                A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=0.5),
                                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ToTensorV2(),
                                ])

    val_transform = A.Compose([
                                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                              ToTensorV2(),
                              ])

    return train_transform, val_transform


def all_transform():
    train_transform = A.Compose([
                                A.OneOf([
                                            A.Flip(p=1.0),
                                            A.RandomRotate90(p=1.0),
                                            A.Rotate(p=1.0, limit=(-30, 30)),
                                            A.ShiftScaleRotate(p=1.0),
                                        ], p=0.5),
                                A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=0.5),
                                A.RandomResizedCrop(512, 512, scale=(0.75, 1.0), p=0.5),
                                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.15, p=0.5),
                                A.GaussNoise(p=0.3),
                                A.OneOf([
                                            A.Blur(p=1.0),
                                            A.GaussianBlur(p=1.0),
                                            A.MedianBlur(blur_limit=5, p=1.0),
                                            A.MotionBlur(p=1.0),
                                        ], p=0.1),
                                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ToTensorV2(),
                                ])

    val_transform = A.Compose([
                                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ToTensorV2(),
                                ])

    return train_transform, val_transform


def inference_transform():

    test_transform = A.Compose([
                              A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ToTensorV2(),
                            ])
                        
    return test_transform


def transfrom_entrypoint(transfrom_name):
    return _transfrom_entrypoints[transfrom_name]


def is_transfrom(transfrom_name):
    return transfrom_name in _transfrom_entrypoints


def get_transfrom(transfrom_name):
    if is_transfrom(transfrom_name):
        transfrom = transfrom_entrypoint(transfrom_name)
    else:
        raise RuntimeError('Unknown transfrom (%s)' % transfrom_name)
    return transfrom


_transfrom_entrypoints = {
    'base_transform': base_transform(),
    'rotate30_transform' : rotate30_transform(),
    'randomresizedcrop_transform' : randomresizedcrop_transform(),
    'clahe_transform' : clahe_transform(),
    'flip_transform': flip_transform(),
    'recommend_transform' : recommend_transform(),
    'all_transform' : all_transform(),
    'inference_transform': inference_transform(),
}