# Downloab Pretrain Model Weight


pytorch >= 1.7.0 버전 이상이 필요합니다. 
다른 환경에서 설치해주세요! 
기존 환경에서 1.7.0 버전으로 변경시 mmcv가 1.6.0 버전에 맞춰져 있어서 오류 발생합니다. 

---

Swin-B : python tools/model_converters/swin2mmseg.py https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth pretrain/swin_base_patch4_window12_384_22k.pth

Swin-L : python tools/model_converters/swin2mmseg.py https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth pretrain/swin_large_patch4_window12_384_22k.pth

---

Upernet-Deit-B-ln-min : python tools/model_converters/vit2mmseg.py https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_deit-b16_ln_mln_512x512_160k_ade20k/upernet_deit-b16_ln_mln_512x512_160k_ade20k_20210623_153535-8a959c14.pth pretrain/deit_base_patch16_224-b5f2ef4d.pth

---

segformer : python tools/model_converters/swin2mmseg.py https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_512x512_160k_ade20k/segformer_mit-b5_512x512_160k_ade20k_20210726_145235-94cedf59.pth pretrain/mit_b5.pth

---

SETR : python tools/model_converters/vit2mmseg.py https://download.openmmlab.com/mmsegmentation/v0.5/setr/setr_mla_512x512_160k_b8_ade20k/setr_mla_512x512_160k_b8_ade20k_20210619_191118-c6d21df0.pth pretrain/vit_large_patch16_384.pth

