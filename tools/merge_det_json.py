import json
import os
import shutil
from mmcv.utils import mkdir_or_exist
from tqdm import tqdm
import shutil
import os.path as osp
def json_load(file_name):
    with open(file_name,'r') as f:
        data = json.load(f)
        return data

image_root = ''
dataset_root = ''
pre_json = '/data3/zzhang/annotation/ulcer/trainall.json'
out_jsons = ['/data3/zzhang/tmp/ulcer_gt_det/filterfilejson/filter' + str(i) + '.json' for i in range(12)]
before_images = '/data2/dataset/gastric_object_detection/ulcer/'
filter_filenames = ['/data3/zzhang/tmp/ulcer_gt_det/filterfiletxt/filter_filename' + str(i) + '.txt' for i in range(12)]
filter_images_dir = ['/data3/zzhang/tmp/ulcer_gt_det/filterimages/filter' + str(i) for i in range(12)]
det_json = '/data3/zzhang/mmdetectionwork_dirs/faster_rcnn_r50_fpn_1x_ulcer_9x/result.bbox.json'
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
        mkdir_or_exist(filter_images_dir[i])
        for file in filter_file:
            shutil.copy(osp.join(before_images, file), osp.join(filter_images_dir[i], file))
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
            
            if img['file_name'] in filter_file:
                file_names.append(img['file_name'])
                image_id = len(merged_data["images"]) + 1
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

    print(len(file_names))
    print('images %d, annos %d'%(len(merged_data["images"]), len(merged_data['annotations'])))

    with open(out_json, 'w') as out_file:
        json.dump(merged_data, out_file)
