import json
import os
import shutil
from mmcv.utils import mkdir_or_exist
from tqdm import tqdm
import shutil
import os.path as osp
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import argparse
 
def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    parser.add_argument(
        '--data_root', type=str, default= 'work_dirs/faster_rcnn_r50_fpn_1x_uav/', help='CPU/CUDA device option')
    parser.add_argument(
        '--anno_json', type=str, default='./data/test.json',help='anno file')
    parser.add_argument(
        '--det_json', type=str, default='', help='det file')
    parser.add_argument(
        '--score', type=bool, default=False, help='det file')
    parser.add_argument(
        '--score-thr', type=float, default=0.1, help='bbox score threshold')
    args = parser.parse_args()
    return args

args = parse_args()
def json_load(file_name):
    with open(file_name,'r') as f:
        data = json.load(f)
        return data

data_root = args.data_root
anno_json = args.anno_json
det_json = data_root + 'result.bbox.json'
preds = json_load(det_json)
anno = COCO(anno_json)
predall = anno.loadRes(det_json)

result = {}

imgs = anno.imgs

for index, img in tqdm(imgs.items()):
    imgId = img["id"]
    file_name = anno.loadImgs(imgId)[0]['file_name']
    video_name = file_name.split("/")[0]
    frame = int(file_name.split("/")[2][:-4])

    annIds = predall.getAnnIds(imgId)
    anns = predall.loadAnns(annIds)

    score_max = args.score_thr
    res = []
    for ann in anns:
        if ann['score'] > score_max:
            score_max = ann['score']
            res = ann['bbox']
            if args.score:
                res.append(score_max)
    if video_name not in result.keys():
         result[video_name] = {}
    result[video_name][frame] = res


os.makedirs(data_root + 'results/', exist_ok=True)
os.popen('rm -r ' + data_root + 'results/' + '*')
for video_name, values in tqdm(result.items()):
    out_json = data_root + 'results/' + video_name + '_IR.txt'
    
    res_list = {}
    res_list['res'] = []
    for idx in range(1, 1+len(values)):
        res_list['res'].append(values[idx])
    with open(out_json, 'w') as out_file:
        json.dump(res_list, out_file)



