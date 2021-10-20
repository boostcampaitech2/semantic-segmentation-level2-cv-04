#!/bin/bash

conda activate mmseg

python tools/train.py configs/_custom_/models/hr48_512x512.py --seed 42 --work-dir ../output/mmseg/hr48_512x512 