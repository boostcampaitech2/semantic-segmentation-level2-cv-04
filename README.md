# ♻️ Semantic Segmentation for Recycling Trash

</br>

Boostcourse AI Competition from [https://stages.ai/](https://stages.ai/)

</br>

## 👨‍🌾 Team

- Level 2 CV Team 4 - 무럭무럭 감자밭 🥔
- 팀 구성원: 김세영, 박성진, 신승혁, 이상원, 이윤영, 이채윤, 조성욱

</br>

## 🏆 LB Score

- Public LB: 0.780 mIoU (6등/20팀)
- Private LB: 0.760 mIoU (3등/20팀)

</br>

## 🎈 Main Subject

- 바야흐로 **대량 생산, 대량 소비**의 시대. 우리는 많은 물건이 대량으로 생산되고 소비되는 시대를 삶에 따라 **쓰레기 대란, 매립지 부족**과 같은 사회 문제 발생
- 버려지는 쓰레기 중 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문에 분리수거는 사회적 환경 부담 문제를 줄일 수 있는 방법
- Deep Learning을 통해 쓰레기들을 자동으로 추출할 수 있는 모델 개발 

</br>

## ⚙ Development Environment
- GPU : Nvidia Tesla V100
- OS : Linux Ubuntu 18.04
- Runtime : Python 3.8.5
- Main Dependency : MMsegmentation, segmentation_models.pytorch, Pytorch 1.7.1, OpenCV 4.5.1

<br>

## 🔑 Project Summary

- 여러 종류의 쓰레기 사진들을 입력값으로 받아 쓰레기의 종류와 위치를 파악하는 Semantic Segmentation
- 다양한 API ([mmsegmentation](https://github.com/open-mmlab/mmsegmentation) & [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) 활용
- Custom baseline 작성을 통한 모델 학습 및 추론
- EDA: 주어진 데이터셋을 이해하기 위해 ipynb 파일로 시각화하여 학습데이터의 전체 & 클래스별 구성과 이미지들의 특징들을 파악
- CV Strategy: 각 클래스의 비율을 고려한 Training Dataset과 Validation Dataset을 8대2 비율로 분리
- Data Augmentation : Albumentation 라이브러리를 이용
    - CLAHE
    - Flip, Rotate(90, 30)
    - Brightness/Contrast, HueSaturation
    - Crop (RandomResizedCrop)
    - Blur (Gaussian, Median, Motion)
    
- [TTA(Test Time Augmentation)](https://github.com/qubvel/ttach) API 활용
- Ensemble: Custom soft-voting 및 csv hard-voting 활용

### Dataset

- 전체 이미지 개수 : 4091장
- 11 class : Background, General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
- 이미지 크기 : (512, 512)
- 원활한 대회 운영 위해 output을 일괄적으로 256 x 256 으로 변경하여 score를 반영
- 학습데이터는 3272장, 평가데이터는 819장으로 무작위 선정
    - 평가데이터: Public 50%, Private 50%

### Metrics

- mIoU (Mean Intersection over Union)
    - Semantic Segmentation에서 사용하는 대표적인 성능 측정 방법
    - 각 클래스 별 Ground Truth와 Prediction 간 IoU 값들에 대한 평균값

</br>

## 💁‍♀️ Composition

### Used Model
|Model|Backbone|model_dir|LB Score@public|LB Score@private|
|---|:---:|:---:|:---:|---|
|UperNet|SwinL|/mmdetection|0.778|0.753
|PAN|SwinB|/torch_dev|0.752|0.737
|UperNet|ViT|/mmdetection|0.706|0.629

### Working Directory
```
├──input
├──output
├──mmsegmentation
└──torch_dev
```

각 폴더 별 자세한 사용 설명은 폴더 내 README.md 참고
