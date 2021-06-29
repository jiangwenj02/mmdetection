import json
import os
import shutil
import os.path as osp
import mmcv
from tqdm import tqdm

image_id = 0
annotation_id = 0

image_summary = 'data/erosiveulcer/fp.txt'
image_root = ''
dataset_root = ''
out_json = 'data/erosiveulcer/test2.json'

with open(image_summary, 'r') as f:
    image_filenames = f.readlines()


merged_data = {
                "licenses": [{"name": "", "id": 0, "url": ""}],
                "info": {"contributor": "", "date_created": "", "description": "", "url": "", "version": "", "year": ""},
                "categories": [{"id": 1, "name": "others", "supercategory": ""}],
                "images": [],
                "annotations": []
}
img_id_test = set()
imgs_num = 0
anno_num = 0
for idx, image in tqdm(enumerate(image_filenames)):
        image = image.strip()
        img = {}            
        img['id'] = len(merged_data["images"]) + 1
        img['file_name'] = image.split('/')[-1]
        img_ori = mmcv.imread(image)
        img['height'] = img_ori.shape[0]
        img['width'] = img_ori.shape[1]
        img['license'] = 0
        img['flickr_url'] = ''
        img['coco_url'] = ''
        img['date_captured'] = 0
        merged_data["images"].append(img)

print('images %d, annos %d'%(len(merged_data["images"]), len(merged_data["annotations"])))

with open(out_json, 'w') as out_file:
    json.dump(merged_data, out_file)
