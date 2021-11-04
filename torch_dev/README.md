## Overview

pytorch, smp 기반 모델 훈련 파이프라인 입니다.

|Model|리더보드 mAP|Valid mAP|
|:---:|:---:|:---:|
|PAnet + Swin|0.752|0.716|
|PAnet + Resnest269e|0.716|0.645|

## Install

```bash
pip3 install -r requirements.txt
```

## Usage

- Train

  일단 custom 내에서 sample을 기반으로 각종 인자들을 수정합니다.  

	```
	python3 train.py --dir {custom 내 폴더이름}
	```

- Inference
	
	훈련이 끝난 뒤 output 폴더 내에 custom_name 인자를 따라 폴더가 생성됩니다.  
	다음 model 폴더 내에서 inference할 모델을 정합니다.  

	```
	python3 inference.py --dir {custom_name} --model {epoch00} 
	```
