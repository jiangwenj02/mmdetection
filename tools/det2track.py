import json
import os
import shutil
from mmcv.utils import mkdir_or_exist
from tqdm import tqdm
import shutil
import os.path as osp
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def json_load(file_name):
    with open(file_name,'r') as f:
        data = json.load(f)
        return data

data_root = 'work_dirs/faster_rcnn_r50_fpn_1x_uav/'
anno_json = './data/train.json'
det_json = data_root + 'result.bbox.json'

preds = json_load(det_json)
anno = COCO(anno_json)
pred = anno.loadRes(det_json)

result = {}

# imgs = pred.imgs

# for img in imgs:
#     imgId = img[id]
#     annIds = pred.getAnnIds(imgId)

# preds = json_load(det_json)

for pred in tqdm(preds):
    image_id = pred["image_id"]
    bboxes = pred["bbox"]
    score = pred["score"]

    file_name = anno.getImgIds(imgIds=image_id)

    video_name = file_name.split("/")[0]
    frame = int(file_name.split("/")[1])

    if file_name not in result.keys():
        result['file_name'] = {}
    
    if frame in result['file_name'].keys():
        if score > 0.1:
            if len(result['file_name'][frame]):
                if score > result['file_name'][frame][-1]:
                    result['file_name'][frame] = bboxes.extend(score)
            else:
                result['file_name'][frame] = bboxes.extend(score)
        else:
            result['file_name'][frame] = []


for file_name, values in result.items():
    out_json = data_root + 'results/' + file_name + '_IR.txt'
    res_list = {}
    res_list['res'] = []
    for idx in range(len(values)):
        res_list.append(values[idx])
    with open(out_json, 'w') as out_file:
        json.dump(res_list, out_file)



