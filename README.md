# ♻️ Semantic Segmentation for Recycling Trash

</br>

Boostcourse AI Competition from [https://stages.ai/](https://stages.ai/)

</br>

## 👨‍🌾 Team

- Level 2 CV Team 4 - 무럭무럭 감자밭 🥔
- 팀 구성원: 김세영, 박성진, 신승혁, 이상원, 이윤영, 이채윤, 조성욱

</br>

## 🏆 LB Score

- Public LB: 0.698 mAP (3등/19팀)
- Private LB: 0.685 mAP (3등/19팀)

</br>

## 🎈 Main Subject

- 바야흐로 **대량 생산, 대량 소비**의 시대. 우리는 많은 물건이 대량으로 생산되고 소비되는 시대를 삶에 따라 **쓰레기 대란, 매립지 부족**과 같은 사회 문제 발생
- 버려지는 쓰레기 중 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문에 분리수거는 사회적 환경 부담 문제를 줄일 수 있는 방법
- Deep Learning을 통해 쓰레기들을 자동으로 분류할 수 있는 모델 개발 

</br>

## ⚙ Development Environment
- GPU : Nvidia Tesla V100
- OS : Linux Ubuntu 18.04
- Runtime : Python 3.8.5
- Main Dependency : Yolov5, MMdetection, Detectron2, Pytorch 1.7.1, OpenCV 4.5.1

<br>

## 🔑 Project Summary

- 여러 종류의 쓰레기 사진들을 입력값으로 받아 쓰레기의 종류와 위치를 파악하는 Object Detection
- 다양한 API([mmdetection](https://github.com/open-mmlab/mmdetection) & [detectron2](https://github.com/facebookresearch/detectron2) & [yolov5](https://github.com/ultralytics/yolov5)) 활용    
- EDA: 주어진 데이터셋을 이해하기 위해 ipynb 파일로 시각화하여 학습데이터의 전체 & 클래스별 구성과 이미지들의 특징들을 파악
- CV Strategy: 각 클래스의 비율을 고려한 Training Dataset과 Validation Dataset을 8대2 비율로 분리
- Data Augmentation : Albumentation 라이브러리를 이용
    - Flip, RandomRotate90 : 가장 효과적인 Augmentation이였으며 이후 TTA에서도 사용되어 높은 성능향상
    - RandomResizedCrop : Flip과 마찬가지로 여러가지 크기와 잘린 이미지들이 들어올 수 있어서 해당 Augmentation 적용
    - RandomBrightnessContrast, HueSaturationValue : EDA 결과 여러가지 밝기와 색상의 입력이 들어올 수 있어서 해당 Augmentation을 적용
    - GaussNoise, Blur : 초점이 어긋난 사진이 있어 해당 Augmentation 적용
- [TTA(Test Time Augmentation)](https://inspaceai.github.io/2019/12/20/Test_Time_Augmentation_Review/) 적용
- Ensemble: [Weighted-Boxes-Fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion) WBF, IoU=0.6 으로 모델 앙상블 

### Dataset

- 전체 이미지 개수 : 9754장
- 10 class : General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
- 이미지 크기 : (1024, 1024)
- 학습데이터는 4883장, 평가데이터는 4871장으로 무작위 선정
    - 평가데이터: Public 50%, Private 50%

### Metrics

- mAP50 (Mean Average Precision)
    - Object Detection에서 사용하는 대표적인 성능 측정 방법
    - Ground Truth 박스와 Prediction 박스간 IoU(Intersection Over Union, Detector의 정확도를 평가하는 지표)가 50이 넘는 예측에 대해 True라고 판단

</br>

## 💁‍♀️ Composition

### Used Model
|Model|Neck|Head|Backbone|model_dir
|---|:---:|:---:|:---:|---|
|Swin|PA-FPN|Cascade-RCNN|Swin|/mmdetection|
|Swin-S|FPN|Cascade-RCNN|Swin|/mmdetection|
|Swin-B|FPN|Cascade-RCNN|Swin|/Swin-Transformer-Object-Detection|
|EfficientDet|-|-|Efficientnet|/efficientdet|
|YOLOv5x6|-|-|YOLOv5|/yolov5|

### Working Directory
```
├──dataset
|   ├──eda
|   ├──yolov5       # dataset by yolo format
|   └──json files   # dataset by coco format
├──output
|   ├──detectron
|   ├──mmdet
|   └──yolov5
├──detectron
├──mmdetection
├──Swin-Transformer-Object-Detection # swin-b
├──efficientdet
└──yolov5
```

각 폴더 별 자세한 사용 설명은 폴더 내 README.md 참고
