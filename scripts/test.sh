#!/usr/bin/env bash
OMP_NUM_THREADS=12 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11 ./tools/dist_train.sh configs/adenomatous/mask_rcnn_r50_fpn_1x_adenomatous.py 12
python tools/test.py configs/gang/mask_rcnn_r50_fpn_1x_gang_left.py work_dirs/mask_rcnn_r50_fpn_1x_gang_left/latest.pth --eval bbox segm --options "classwise=True" --out work_dirs/mask_rcnn_r50_fpn_1x_gang_left/result.pkl
python tools/test.py configs/gang/mask_rcnn_r50_fpn_1x_gang_right.py work_dirs/mask_rcnn_r50_fpn_1x_gang_right/latest.pth --eval bbox segm --options "classwise=True" --out work_dirs/mask_rcnn_r50_fpn_1x_gang_right/result.pkl
python tools/test.py configs/gang/mask_rcnn_r50_fpn_1x_gang_right.py work_dirs/mask_rcnn_r50_fpn_1x_gang_right/latest.pth --format-only --options "jsonfile_prefix=work_dirs/mask_rcnn_r50_fpn_1x_polyp/"
python tools/eval_others.py --result work_dirs/faster_rcnn_r50_fpn_1x_erosive_9x/result.bbox.json --ann /data0/zzhang/annotation/erosive/test.json
#CUDA_VISIBLE_DEVICES=8,9,10,11 ./tools/dist_test.sh configs/polyp/solov2_r50_fpn_polyp_8gpu_3x.py work_dirs/solov2_r50_fpn_8gpu_3x_poly/latest.pth 4 --eval segm --out work_dirs/solov2_r50_fpn_8gpu_3x_poly/result.pkl