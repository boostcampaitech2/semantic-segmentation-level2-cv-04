#!/bin/bash

# python tools/train.py configs/_custom_/models/hr48_512x512.py --seed 42 --work-dir ../output/mmseg/hr48_512x512 

# python tools/train.py configs/_custom_/models/hr48.py --seed 42 --work-dir ../output/mmseg/hr48

# python tools/train.py configs/_custom_/models/upernet_swin_b.py --seed 42 --work-dir ../output/mmseg/upernet_swin_b

# python tools/train.py configs/_custom_/models/deeplabv3plus_r101.py --seed 42 --work-dir ../output/mmseg/deeplabv3plus_r101

python tools/train.py configs/_custom_/models/ocrnet_hr48_512x_40k_voc12.py --seed 42 --work-dir ../output/mmseg/ocrnet_hr48