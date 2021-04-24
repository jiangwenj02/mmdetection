import json
import os
import shutil
from tqdm import tqdm
def json_load(file_name):
    with open(file_name,'r') as f:
        data = json.load(f)
        return data



image_root = ''
dataset_root = ''
pre_json = '/data0/zzhang/annotation/ulcer/trainall.json'
out_jsons = ['/data0/zzhang/tmp/ulcer_gt_det/filterfilejson/filter' + str(i) + '.json' for i in range(12)]
filter_filenames = ['/data0/zzhang/tmp/ulcer_gt_det/filterfiletxt/filter_filename' + str(i) + '.txt' for i in range(12)]
det_json = 'work_dirs/faster_rcnn_r50_fpn_1x_ulcer_9x/result.bbox.json'
preds = json_load(det_json)

for i in range(len(filter_filenames)):
    merged_data = {
                "licenses": [{"name": "", "id": 0, "url": ""}],
                "info": {"contributor": "", "date_created": "", "description": "", "url": "", "version": "", "year": ""},
                "categories": [],
                "images": [],
                "annotations": []
                }
    filter_filename = filter_filenames[i]
    out_json = out_jsons[i]

    with open(filter_filename, 'r') as f:
        filter_file = f.readlines()
        filter_file = [file.strip() for file in filter_file]
    print('the number of filter image: %d' % (len(filter_file)))
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
        file_names = []
        for img in tqdm(data["images"]):
            file_names.append(img['file_name'])
            if img['file_name'] in filter_file:
                image_id = len( merged_data["images"]) + 1
                img_id_map[img['id']] = image_id
                img['id'] = image_id
                merged_data["images"].append(img)            

        file_names = set(file_names)
        filter_file = set(filter_file)

        for pred in tqdm(preds):
            if pred['image_id'] in img_id_map.keys():
                anno = {}
                anno['id'] = len(merged_data['annotations']) + 1
                anno['category_id'] = pred['category_id']
                anno['image_id'] = img_id_map[pred['image_id']]
                anno['bbox'] = pred['bbox']  
                anno['iscrowd'] = 0
                anno['area'] = pred['bbox'][-2] * pred['bbox'][-1]
                anno['attributes'] = {"occluded": 'false'}
                merged_data['annotations'].append(anno)


    print('images %d, annos %d'%(len(merged_data["images"]), len(merged_data['annotations'])))

    with open(out_json, 'w') as out_file:
        json.dump(merged_data, out_file)
