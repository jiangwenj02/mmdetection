import json
import os
import shutil

def json_load(file_name):
    with open(file_name,'r') as f:
        data = json.load(f)
        return data

image_id = 0
annotation_id = 0

image_root = ''
dataset_root = ''
pre_json = '/data0/zzhang/annotation/erosive/trainall.json'
out_json = '/data0/zzhang/annotation/erosive/filter.json'
filter_filename = '/data0/zzhang/tmp/erosive_gt_det/erosive_gt_detfilename.txt'
det_json = 'work_dirs/faster_rcnn_r50_fpn_1x_erosive_9x/result.bbox.json'
preds = json_load(det_json)

merged_data = {
                "licenses": [{"name": "", "id": 0, "url": ""}],
                "info": {"contributor": "", "date_created": "", "description": "", "url": "", "version": "", "year": ""},
                "categories": [{"id": 1, "name": "erosive", "supercategory": ""}],
                "images": [],
                "annotations": []
}


with open(filter_filename, 'r') as f:
    filter_file = f.readlines()
    filter_file = [file.strip() for file in filter_file]

img_id_test = set()
imgs_num = 0
anno_num = 0
with open(pre_json) as json_file:
    data = json.load(json_file)
    merged_data["licenses"] = data["licenses"]
    merged_data["info"] = data["info"]
    merged_data["categories"] = data["categories"]

    id_list = set()
    img_id_map = {}
    for img in data["images"]:
        if img['file_name'] in filter_file:
            img_id_map[img['id']] = image_id
            img['id'] = image_id
            image_id = image_id + 1
            merged_data["images"].append(img)            

    for pred in preds:
        if pred['image_id'] in img_id_map.keys():
            anno = {}
            anno['id'] = annotation_id
            annotation_id = annotation_id + 1
            anno['category_id'] = pred['category_id']
            anno['image_id'] = img_id_map[pred['image_id']]
            anno['bbox'] = pred['bbox']  
            anno['iscrowd'] = 0
            anno['attributes'] = {"occluded": 'false'}
            merged_data['annotations'].append(anno)


print('images %d, annos %d'%(image_id, annotation_id))

with open(out_json, 'w') as out_file:
    json.dump(merged_data, out_file)
