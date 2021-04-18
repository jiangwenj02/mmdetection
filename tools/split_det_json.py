import json
import os
import shutil
from tqdm import tqdm
def json_load(file_name):
    with open(file_name,'r') as f:
        data = json.load(f)
        return data

image_id = 0
annotation_id = 0

image_root = ''
dataset_root = ''
pre_json = '/data0/zzhang/annotation/erosive/trainall.json'
filter_filename = '/data0/zzhang/tmp/erosive_gt_det/filter_filename.txt'
outputfile = ['/data0/zzhang/tmp/erosive_gt_det/filter_filename1.txt', '/data0/zzhang/tmp/erosive_gt_det/filter_filename2.txt']
f_list = [open(file, 'w') for file in outputfile]

with open(filter_filename, 'r') as f:
    filter_file = f.readlines()
    filter_file = [file.strip() for file in filter_file]
print('the number of filter image: %d' % (len(filter_file)))
img_id_test = set()
imgs_num = 0
anno_num = 0
with open(pre_json) as json_file:
    data = json.load(json_file)
    
    id_list = set()
    img_id_map = {}
    file_names = []
    count = 0 
    for img in tqdm(data["images"]):
        file_names.append(img['file_name'])
        if img['file_name'] not in filter_file: 
            f_list[count % len(outputfile)].write(img['file_name'] + '\n')
            count = count + 1

for f in f_list:
    f.close()
            
    