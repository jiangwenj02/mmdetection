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
predall = anno.loadRes(det_json)

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

    file_name = anno.loadImgs(image_id)[0]['file_name']
    # anns_id = predall.getAnnIds(image_id)
    # anns = predall.loadAnns(anns_id)

    video_name = file_name.split("/")[0]
    frame = int(file_name.split("/")[2][:-4])

    if video_name not in result.keys():
        result[video_name] = {}
    bboxes.append(score)
    if frame in result[video_name].keys():
        if score > 0.1:
            if len(result[video_name][frame]):
                if score > result[video_name][frame][-1]:                    
                    result[video_name][frame] = bboxes[:]
            else:
                result[video_name][frame] = bboxes[:]
    else:
        if score > 0.1:
            result[video_name][frame] = bboxes[:]
            print(result[video_name][frame])
        else:
            result[video_name][frame] = []


os.makedirs(data_root + 'results/', exist_ok=True)
for video_name, values in result.items():
    out_json = data_root + 'results/' + video_name + '_IR.txt'
    
    res_list = {}
    res_list['res'] = []
    for idx in range(1, 1+len(values)):
        res_list.append(values[idx][:-2])
    with open(out_json, 'w') as out_file:
        json.dump(res_list, out_file)



