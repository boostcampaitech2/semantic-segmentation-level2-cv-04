import ttach as tta

def get_tta(model):
    transforms = tta.Compose([
        tta.HorizontalFlip(),
        tta.VerticalFlip(),
        tta.Scale(scales=[0.5, 0.75, 1, 1.25, 1.5])
    ])

    tta_model = tta.SegmentationTTAWrapper(model, transforms)

    return tta_model

tta.Rotate90