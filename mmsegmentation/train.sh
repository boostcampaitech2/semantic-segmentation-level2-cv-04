#!/bin/bash

# python tools/train.py configs/_custom_/models/hr48_512x512.py --seed 42 --work-dir ../output/mmseg/hr48_512x512 

# python tools/train.py configs/_custom_/models/hr48.py --seed 42 --work-dir ../output/mmseg/hr48_0

python tools/train.py configs/_custom_/models/upernet_swin_b.py --seed 42 --work-dir ../output/mmseg/upernet_swin_b_0