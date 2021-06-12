import json
import os
import shutil

image_id = 0
annotation_id = 0

adenomatous_json_dir = 'data/erosiveulcer_fine/sub_train/'
adenomatous_json_dir = ['/data3/zzhang/annotation/erosiveulcer_fine/train.json', '/data3/zzhang/annotation/erosiveulcer_fine/filt_fp.json']
if type(adenomatous_json_dir) is type(''):
    files = os.listdir(adenomatous_json_dir)
    json_dir = [os.path.join(adenomatous_json_dir, f) for f in files]
image_root = ''
dataset_root = ''
out_json = '/data3/zzhang/annotation/erosiveulcer_fine/trainfp.json'

merged_data = {
                "licenses": [{"name": "", "id": 0, "url": ""}],
                "info": {"contributor": "", "date_created": "", "description": "", "url": "", "version": "", "year": ""},
                "categories": [{"id": 1, "name": "erosive", "supercategory": ""}],
                "images": [],
                "annotations": []
}
img_id_test = set()
imgs_num = 0
anno_num = 0
filename_to_id = {}
category_to_id = {}
for idx, f in enumerate(adenomatous_json_dir):
    if f == '.DS_Store' or not('.json' in f):
        continue
    
    with open(f) as json_file:

        data = json.load(json_file)
        for img_idx in range(len(data['images'])):
            data['images'][img_idx]['file_name'] = data['images'][img_idx]['file_name'].split('/')[-1]
            # if data['images'][img_idx]['file_name'] == '3a6f2f9f-bbbe-4373-a82f-363cde7508b1.jpg':
            #     print(f)
        
        if idx == 0:
            merged_data["licenses"] = data["licenses"]
            merged_data["info"] = data["info"]
            merged_data["categories"] = data["categories"]
            for item in merged_data["categories"]:
                category_to_id[item['name']] = item['id']
            merged_data['annotations'] = data["annotations"]
            merged_data['images'] = data["images"]

            for img in merged_data['images']:
                filename_to_id[img['file_name']] = img['id']

        else:
            oldid_to_filename = {}
            for img in data["images"]:
                if img['file_name'] in filename_to_id:
                    oldid_to_filename[img['id']] = img['file_name']
                else:
                    oldid_to_filename[img['id']] = img['file_name']
                    img['id'] = image_id
                    merged_data["images"].append(img)                    
                    filename_to_id[img['file_name']] = image_id
                    image_id = image_id + 1

            oldcatid_to_new_id = {}
            for item in data["categories"]:
                if item['name'] in category_to_id:
                    oldcatid_to_new_id[item['id']] = category_to_id[item['name']]
                else:                    
                    category_to_id[item['name']] = len(category_to_id) + 1
                    oldcatid_to_new_id[item['id']] = len(category_to_id)
                    item['id'] = len(category_to_id)
                    merged_data['categories'].append(item)
            print(oldcatid_to_new_id)
            for anno in data["annotations"]:
                anno['id'] = len(merged_data['annotations']) + 1
                anno['category_id'] = oldcatid_to_new_id[anno['category_id']]
                anno['image_id'] = filename_to_id[oldid_to_filename[anno['image_id']]]                  
                merged_data['annotations'].append(anno)
        print(f, len(data['images']), len(merged_data["images"]))
        image_id = len(merged_data["images"]) + 1
        annotation_id = len(merged_data["annotations"]) + 1
print('images %d, annos %d'%(len(merged_data["images"]), len(merged_data["annotations"])))

with open(out_json, 'w') as out_file:
    json.dump(merged_data, out_file)
