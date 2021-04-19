import json
import os
import shutil

image_id = 0
annotation_id = 0

adenomatous_json_dir = 'data/erosive/'
image_root = ''
dataset_root = ''
json_file = 'data/erosive2/annotations/instances_default.json'
txt_file = 'data/erosive2/des.txt'
with open(json_file) as json_f:
    data = json.load(json_f)

des = {}
with open(txt_file, 'r', encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split(' ')
        print(line)
        des[int(line[0]) + 1] = line[1]

print(des)

remove1_txt_f = open('data/erosive2/shidao.txt', 'w')
remove2_txt_f = open('data/erosive2/no.txt', 'w')
remove_all_txt_f = open('data/erosive2/fine.txt', 'w')
remove_txt_f = open('data/erosive2/remove.txt', 'w')

for img in data["images"]:
    if '未标注' in des[img['id']]:
        remove2_txt_f.write(img['file_name'] + '\n')
        remove_txt_f.write(img['file_name'] + '\n')
    if '食管' in des[img['id']]:
        remove1_txt_f.write(img['file_name'] + '\n')
        remove_txt_f.write(img['file_name'] + '\n')
    remove_all_txt_f.write(img['file_name'].split('/')[-1] + '\n')

remove1_txt_f.close()
remove2_txt_f.close()
remove_txt_f.close()
remove_all_txt_f.close()
    