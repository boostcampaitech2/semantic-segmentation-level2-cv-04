#!/bin/bash

# python tools/test.py configs/_custom_/models/hr48.py ../output/mmseg/hr48_0/latest.pth --show-dir ../output/mmseg/test

# python tools/test.py configs/_custom_/models/upernet_swin_b.py ../output/mmseg/upernet_swin_b_stratified/latest.pth --show-dir ../output/mmseg/test

python tools/test.py configs/_custom_/models/segformer_b5.py ../output/mmseg/segformer_b5/latest.pth --show-dir ../output/mmseg/test