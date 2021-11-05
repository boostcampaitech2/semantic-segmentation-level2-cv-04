# Install MMSegmentation

1. 가상환경 구축하기

```
MMSegmentation을 위한 가상환경을 따로 구축해서 사용했습니다. 
(Baseline_code와의 버전 충돌 방지)

conda create -n mmseg python=3.7 -y
conda activate mmseg
```

2. PyTorch, Torchvision, mmcv 설치

```
conda install pytorch=1.6.0 torchvision cudatoolkit=10.1 -c pytorch

pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
```

3. MMSegmentation 설치

```
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
python setup.py develop
```

MMSegmentation 설치가 완료되었다면, 필요한 파일을 가져와서 사용하시면 됩니다.
추가적으로 wandb, pandas 등 설치가 필요합니다.

---

# Change Dataset format

MMSegmentation 사용하려면 기존의 coco format 데이터를 MMseg에서 사용하기 위해 format을 먼저 바꿔야 합니다.

다음과 같이 dir을 구성해주세요
```
├──input
|   ├──data
|   └──mmseg                # dataset by mmseg format
|       ├──images           # .jpg
|       |   ├──training    
|       |   └──validation
|       ├──annotations      # .png
|       |   ├──training
|       |   └──validation
|       └──test
└──mmsegmentation
```

다음 파일들을 통해서 생성 가능합니다.
- "/input/data/make_json_image.py"
- "/input/data/make_json_mask.ipynb"

---

# Train 

MMSegmentation을 통한 학습은 다음 script와 같이 진행합니다.
```
python tools/train.py {model_config.py path} --work-dir {work_dir path} --seed 42 
```

# Inference

Inference.ipynb
