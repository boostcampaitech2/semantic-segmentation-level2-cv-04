#!/bin/bash

conda activate mmseg

python tools/test.py configs/_custom_/models/hr48_512x512.py ../output/mmseg/hr48_512x512/latest.pth --show_dir ../output/mmseg/test
