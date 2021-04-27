import json
import os
import shutil

image_id = 0
annotation_id = 0

adenomatous_json_dir = 'data/erosive2/annotations/trainall.json'
image_root = ''
out_json = ['data/erosive2/annotations/fine_train.json', 'data/erosive2/annotations/fine_test.json']

merged_data1 = {
                "licenses": [{"name": "", "id": 0, "url": ""}],
                "info": {"contributor": "", "date_created": "", "description": "", "url": "", "version": "", "year": ""},
                "categories": [{"id": 1, "name": "erosive", "supercategory": ""}],
                "images": [],
                "annotations": []
}
merged_data2 = {
                "licenses": [{"name": "", "id": 0, "url": ""}],
                "info": {"contributor": "", "date_created": "", "description": "", "url": "", "version": "", "year": ""},
                "categories": [{"id": 1, "name": "erosive", "supercategory": ""}],
                "images": [],
                "annotations": []
}
merged_com_data = [merged_data1, merged_data2]
img_id_test = set()
imgs_num = 0
anno_num = 0
filename_com_to_id = [{}, {}]
category_to_id = {}


with open(adenomatous_json_dir) as json_file:
    data = json.load(json_file)
    # for img_idx in range(len(data['images'])):
    #     data['images'][img_idx]['file_name'] = data['images'][img_idx]['file_name'].split('/')[-1]
        # if data['images'][img_idx]['file_name'] == '3a6f2f9f-bbbe-4373-a82f-363cde7508b1.jpg':
        #     print(f)
    idx = len(data["images"])
    for merged_data in  merged_com_data:
        merged_data["licenses"] = data["licenses"]
        merged_data["info"] = data["info"]
        merged_data["categories"] = data["categories"]
        for item in merged_data["categories"]:
            category_to_id[item['name']] = item['id']
        merged_data['annotations'] = []
        merged_data['images'] = []

    
    oldid_com_to_filename = [{}, {}]
    split_factor = 0.616
    for idx, img in enumerate(data["images"]):
        com_idx = int(idx > split_factor * len(data["images"]))
        merged_data = merged_com_data[com_idx]
        oldid_to_filename = oldid_com_to_filename[com_idx]
        filename_to_id = filename_com_to_id[com_idx]
        oldid_to_filename[img['id']] = img['file_name']
        image_id = len(merged_data["images"]) + 1
        img['id'] = image_id
        # img['file_name'] = img['file_name'].split('/')[-1]
        merged_data["images"].append(img)                    
        filename_to_id[img['file_name']] = image_id

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
        for idx, oldid_to_filename in enumerate(oldid_com_to_filename):
            if anno['image_id'] in oldid_to_filename:
                break
        
        merged_data = merged_com_data[idx]
        filename_to_id = filename_com_to_id[idx]
        oldid_to_filename = oldid_com_to_filename[idx]

        anno['id'] = len(merged_data['annotations']) + 1
        anno['category_id'] = oldcatid_to_new_id[anno['category_id']]
        
        anno['image_id'] = filename_to_id[oldid_to_filename[anno['image_id']]]                  
        merged_data['annotations'].append(anno)
    print(len(data['images']), len(merged_data["images"]))
    image_id = len(merged_data["images"]) + 1
    annotation_id = len(merged_data["annotations"]) + 1
print('images %d, annos %d'%(len(merged_data["images"]), len(merged_data["annotations"])))

for idx, merged_data in enumerate(merged_com_data):
    with open(out_json[idx], 'w') as out_file:
        json.dump(merged_data, out_file)
