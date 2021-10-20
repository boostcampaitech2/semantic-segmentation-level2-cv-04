MMsegmentation 사용하려면 데이터 셋 format을 먼저 바꿔야 합니다.

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
- "/input/data/image_reroute.py"
- "/input/data/json-filename_edit.py"
- "/baseline_code/make_mask_image.ipynb"

train / inference / visualize code 구현되어 있습니다.