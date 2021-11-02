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


transform_entrypoints = {
    'train_transform' : TrainAugmentation,
    'valid_transform' : ValidAugmentation,
    'test_transform' : TestAugmentation,
}

def is_transform(transform_name):
    return transform_name in transform_entrypoints

def get_transform(transform_name):
    if is_transform(transform_name):
        transform = transform_entrypoints[transform_name]
    else:
        raise RuntimeError('Unknown transform (%s)' % transform_name)
    return transform
