import json
import os
import shutil

image_id = 1
annotation_id = 0

adenomatous_json_dir = 'data/erosive/annotations/test'
inflammatory_json_dir = '/Users/xinzisun/Documents/new_polyp_annotation/Inflammatory/train'
hyperplastic_json_dir = '/Users/xinzisun/Documents/new_polyp_annotation/Hyperplastic/train'
image_root = ''
dataset_root = ''
out_json = 'data/erosive/annotations/test.json'

# merged_data = {
#                 "licenses": [{"name": "", "id": 0, "url": ""}],
#                 "info": {"contributor": "", "date_created": "2021-2", "description": "", "url": "", "version": 1, "year": "2020"},
#                 "categories": [{"id": 1, "name": "Adenomatous", "supercategory": ""}, {"id": 2, "name": "Inflammatory", "supercategory": ""}, {"id": 3, "name": "Hyperplastic", "supercategory": ""}],
#                 "images": [],
#                 "annotations": []
# }

merged_data = {
                "licenses": [{"name": "", "id": 0, "url": ""}],
                "info": {"contributor": "", "date_created": "2021-03", "description": "", "url": "", "version": 1, "year": "2021"},
                "categories": [{"id": 1, "name": "erosive", "supercategory": ""}],
                "images": [],
                "annotations": []
}

# merged_data = {
#                 "licenses": [{"name": "", "id": 0, "url": ""}],
#                 "info": {"contributor": "", "date_created": "2021-2", "description": "", "url": "", "version": 1, "year": "2020"},
#                 "categories": [{"id": 1, "name": "Polyp", "supercategory": ""}, {"id": 2, "name": "Instrument", "supercategory": ""}],
#                 "images": [],
#                 "annotations": []
# }

img_id_test = set()
imgs_num = 0
anno_num = 0
for f in os.listdir(adenomatous_json_dir):
    if f == '.DS_Store':
        continue
    json_dir = os.path.join(adenomatous_json_dir, f)
    with open(json_dir) as json_file:
        data = json.load(json_file)
        id_list = set()
        for img in data["images"]:
            img['id'] += image_id
            img['file_name'] = 'colon_inflammatory/' + '/'.join(img['file_name'].split('/')[2:])
            merged_data["images"].append(img)
            imgs_num += 1

        for anno in data["annotations"]:
            anno['id'] += annotation_id
            anno['category_id'] = 1
            anno['image_id'] += image_id    
            
            merged_data['annotations'].append(anno)

            anno_num += 1
        image_id += len(data["images"])
        annotation_id += len(data["annotations"])
print('images %d, annos %d'%(imgs_num, anno_num))
# for f in os.listdir(inflammatory_json_dir):
#     if f == '.DS_Store':
#         continue
#     json_dir = os.path.join(inflammatory_json_dir, f)
#     with open(json_dir) as json_file:
#         data = json.load(json_file)
#         for img in data['images']:
#             img['id'] += image_id
#             img['file_name'] = 'colon_inflammatory/' + '/'.join(img['file_name'].split('/')[2:])
#             merged_data["images"].append(img)
#         for anno in data["annotations"]:
#             anno['id'] += annotation_id
#             anno['category_id'] = 2
#             anno['image_id'] += image_id
#             merged_data['annotations'].append(anno)
            
#         image_id += len(data['images'])
#         annotation_id += len(data["annotations"])

# for f in os.listdir(hyperplastic_json_dir):
#     if f == '.DS_Store':
#         continue
#     json_dir = os.path.join(hyperplastic_json_dir, f)
#     with open(json_dir) as json_file:
#         data = json.load(json_file)
#         for img in data["images"]:
#             img['id'] += image_id
#             img['file_name'] = 'colon_hyperplastic/' + '/'.join(img['file_name'].split('/')[2:])
#             merged_data["images"].append(img)
#         for anno in data["annotations"]:
#             anno['id'] += annotation_id
#             anno['category_id'] = 2
#             anno['image_id'] += image_id
#             merged_data['annotations'].append(anno)

#         image_id += len(data["images"])
#         annotation_id += len(data["annotations"])

with open(out_json, 'w') as out_file:
    json.dump(merged_data, out_file)
