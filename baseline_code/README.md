## Install
```bash
$ pip install -r requirements.txt
````

## Train
- **[Config 예시](https://github.com/boostcampaitech2/semantic-segmentation-level2-cv-04/blob/dev_yy/baseline_code/config/base_test.yaml)**

- **Train with config file**
  ```bash
  $ python train.py --config './config/base_test.yaml' # config 경로
  ````
- **Train 결과**
  ```
  saved
    └──exp_name                       # config에서 설정한 exp_name
        ├──exp_name.pt                # best_model
        ├──exp_name{epochs-5}.pt      # 마지막 5개 model
        |   ~ exp_name{epochs}.pt
        ├──exp_name.yaml              # 실험에 사용한 yaml 파일    
        └──best_log.txt               # best validation score 갱신 시 mIoU와 class 별 IoU 기록
  ```

## Inference
- **Test with exp_name**
  ```bash
  $ python test.py --exp '0_Base_Test' # config에서 설정했던 실험 이름
  ````
- **Inference 결과**
  ```
  submission
    └──submission_exp_name.csv        # 대회 제출용 csv 파일
  ```
  
## 제출 모델

|Model|Backbone|LB score|Config|
|---|:---:|:---:|:---:|
|PANet|Swin-B|0.752|[Config](https://github.com/boostcampaitech2/semantic-segmentation-level2-cv-04/blob/dev_yy/baseline_code/config/pan_swin.yaml)|
