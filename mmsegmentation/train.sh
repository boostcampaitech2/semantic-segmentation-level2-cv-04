#!/bin/bash

# python tools/train.py configs/_custom_/models/hr48_512x512.py --seed 42 --work-dir ../output/mmseg/hr48_512x512 

# python tools/train.py configs/_custom_/models/hr48.py --seed 42 --work-dir ../output/mmseg/hr48

# python tools/train.py configs/_custom_/models/upernet_swin_b.py --seed 42 --work-dir ../output/mmseg/upernet_swin_b

# python tools/train.py configs/_custom_/models/deeplabv3plus_r101.py --seed 42 --work-dir ../output/mmseg/deeplabv3plus_r101

# python tools/train.py configs/_custom_/models/deeplabv3plus_r101.py --seed 42 --work-dir ../output/mmseg/deeplabv3plus_r101 --load-from ../output/mmseg/deeplabv3plus_r101/latest.pth --resume-from ../output/mmseg/deeplabv3plus_r101/latest.pth

# python tools/train.py configs/_custom_/models/upernet_swin_b.py --seed 42 --work-dir ../output/mmseg/upernet_swin_b_stratified

# python tools/train.py configs/_custom_/models/upernet_deit_b_ln_min.py --seed 42 --work-dir ../output/mmseg/upernet_deit_b_ln_min 

# python tools/train.py configs/_custom_/models/upernet_deit_b_ln_min.py --seed 42 --work-dir ../output/mmseg/upernet_deit_b_ln_min --load-from ../output/mmseg/upernet_deit_b_ln_min/latest.pth --resume-from ../output/mmseg/upernet_deit_b_ln_min/latest.pth

# python tools/train.py configs/_custom_/models/segformer_b5.py --seed 42 --work-dir ../output/mmseg/segformer_b5 --load-from ../output/mmseg/segformer_b5/latest.pth --resume-from ../output/mmseg/segformer_b5/latest.pth

# python tools/train.py configs/_custom_/models/ocrnet_hr48.py --seed 42 --work-dir ../output/mmseg/ocrnet_hr48

# python tools/train.py configs/_custom_/models/danet.py --seed 42 --work-dir ../output/mmseg/danet

# python tools/train.py configs/_custom_/models/upernet_swin_b.py --seed 42 --work-dir ../output/mmseg/upernet_swin_b_base

# python tools/train.py configs/_custom_/models/upernet_swin_b_flip.py --seed 42 --work-dir ../output/mmseg/upernet_swin_b_flip

# python tools/train.py configs/_custom_/models/upernet_swin_b_blur.py --seed 42 --work-dir ../output/mmseg/upernet_swin_b_blur

# python tools/train.py configs/_custom_/models/upernet_swin_b_bright.py --seed 42 --work-dir ../output/mmseg/upernet_swin_b_bright

# python tools/train.py configs/_custom_/models/upernet_swin_l.py --seed 42 --work-dir ../output/mmseg/upernet_swin_l

# python tools/train.py configs/_custom_/models/upernet_swin_b.py --seed 42 --work-dir ../output/mmseg/upernet_swin_b

# python tools/train.py configs/_custom_/models/dnl.py --seed 42 --work-dir ../output/mmseg/dnl

# python tools/train.py configs/_custom_/models/upernet_swin_b.py --seed 42 --work-dir ../output/mmseg/upernet_swin_b --load-from ../output/mmseg/upernet_swin_b/epoch_40.pth --resume-from ../output/mmseg/upernet_swin_b/epoch_40.pth

# python tools/train.py configs/_custom_/models/upernet_swin_l.py --seed 42 --work-dir ../output/mmseg/upernet_swin_l_augment --load-from ../output/mmseg/upernet_swin_l_augment/latest.pth --resume-from ../output/mmseg/upernet_swin_l_augment/latest.pth

python tools/train.py configs/_custom_/models/upernet_swin_l.py --seed 42 --work-dir ../output/mmseg/upernet_swin_l_all_data --no-validate
python tools/train.py configs/_custom_/models/upernet_swin_l.py --seed 4 --work-dir ../output/mmseg/upernet_swin_l_all_data --no-validate