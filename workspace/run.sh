#!/bin/bash

python tools/ta_test_GT.py config/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth --method fgsm --show-dir adv_res/fr50_fgsm_15 --eps 15