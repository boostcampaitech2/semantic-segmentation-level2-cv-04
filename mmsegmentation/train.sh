#!/bin/bash

# python tools/train.py configs/_custom_/models/hr48_512x512.py --seed 42 --work-dir ../output/mmseg/hr48_512x512 

# python tools/train.py configs/_custom_/models/hr48.py --seed 42 --work-dir ../output/mmseg/hr48

# python tools/train.py configs/_custom_/models/upernet_swin_b.py --seed 42 --work-dir ../output/mmseg/upernet_swin_b

# python tools/train.py configs/_custom_/models/deeplabv3plus_r101.py --seed 42 --work-dir ../output/mmseg/deeplabv3plus_r101

# python tools/train.py configs/_custom_/models/deeplabv3plus_r101.py --seed 42 --work-dir ../output/mmseg/deeplabv3plus_r101 --load-from ../output/mmseg/deeplabv3plus_r101/latest.pth --resume-from ../output/mmseg/deeplabv3plus_r101/latest.pth

# python tools/train.py configs/_custom_/models/upernet_swin_b.py --seed 42 --work-dir ../output/mmseg/upernet_swin_b_stratified

#python tools/train.py configs/_custom_/models/upernet_deit_b_ln_min.py --seed 42 --work-dir ../output/mmseg/upernet_deit_b_ln_min 

python tools/train.py configs/_custom_/models/upernet_deit_b_ln_min.py --seed 42 --work-dir ../output/mmseg/upernet_deit_b_ln_min --load-from ../output/mmseg/upernet_deit_b_ln_min/latest.pth --resume-from ../output/mmseg/upernet_deit_b_ln_min/latest.pth
