import json
import os
import shutil

image_id = 0
annotation_id = 0

adenomatous_json_dir = 'data/antiUAVimages/'
image_root = ''
dataset_root = ''
out_json = 'data/train.json'

merged_data = {
                "licenses": [{"name": "", "id": 0, "url": ""}],
                "info": {"contributor": "", "date_created": "", "description": "", "url": "", "version": "", "year": ""},
                "categories": [{"id": 1, "name": "uav", "supercategory": ""}],
                "images": [],
                "annotations": []
}
img_id_test = set()
imgs_num = 0
anno_num = 0
for idx, f in enumerate(os.listdir(adenomatous_json_dir)):
    import pdb
    pdb.set_trace()
    json_dir = os.path.join(adenomatous_json_dir, f, 'IR_label.json')
    with open(json_dir) as json_file:

        data = json.load(json_file)
        import pdb
        pdb.set_trace()
        # if idx == 0:
        #     merged_data["licenses"] = data["licenses"]
        #     merged_data["info"] = data["info"]
        #     merged_data["categories"] = data["categories"]

        id_list = set()
        for img in data["images"]:
            img['id'] += image_id
            img['file_name'] = img['file_name'].split('/')[-1]
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

with open(out_json, 'w') as out_file:
    json.dump(merged_data, out_file)
