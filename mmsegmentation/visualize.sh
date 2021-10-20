#!/bin/bash

python tools/test.py configs/_custom_/models/hr48_512x512.py ../output/mmseg/hr48_0/latest.pth --show-dir ../output/mmseg/test
