# Download Pretrain Model Weight

To use other repositories' pre-trained models, it is necessary to convert keys.
We provide a script swin2mmseg.py in the tools directory to convert the key of models from the official repo to MMSegmentation style.

```
python tools/model_converters/swin2mmseg.py ${PRETRAIN_PATH} ${STORE_PATH}
```

This script convert model from PRETRAIN_PATH and store the converted model in STORE_PATH.
(To supply zip format, need pytorch version >= 1.7.0, Use other environment to use)

---

Swin-B 
```
python tools/model_converters/swin2mmseg.py https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth pretrain/swin_base_patch4_window12_384_22k.pth
```

Swin-L
```
python tools/model_converters/swin2mmseg.py https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth pretrain/swin_large_patch4_window12_384_22k.pth
```

---

Deit-B
```
python tools/model_converters/vit2mmseg.py https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_deit-b16_ln_mln_512x512_160k_ade20k/upernet_deit-b16_ln_mln_512x512_160k_ade20k_20210623_153535-8a959c14.pth pretrain/deit_base_patch16_224-b5f2ef4d.pth
```

Beit-B
```
python tools/model_converters/vit2mmseg.py https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth pretrain/jx_vit_base_p16_224-80ecf9dd.pth
```

---

segformer 
```
python tools/model_converters/swin2mmseg.py https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_512x512_160k_ade20k/segformer_mit-b5_512x512_160k_ade20k_20210726_145235-94cedf59.pth pretrain/mit_b5.pth
```

---

SETR
```
python tools/model_converters/vit2mmseg.py https://download.openmmlab.com/mmsegmentation/v0.5/setr/setr_mla_512x512_160k_b8_ade20k/setr_mla_512x512_160k_b8_ade20k_20210619_191118-c6d21df0.pth pretrain/vit_large_patch16_384.pth
```

