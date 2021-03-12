import json
import glob

# ANNOT_PATH = '/home/zxy/Datasets/VOC/annotations/'
ANNOT_PATH = 'voc/annotations/'
OUT_PATH = ANNOT_PATH
# INPUT_FILES = ['pascal_train2012.json', 'pascal_val2012.json',
#                'pascal_train2007.json', 'pascal_val2007.json']
dirs = './data/erosive/annotations/train/'
INPUT_FILES = glob.glob('*.json')
OUTPUT_FILE = 'train.json'
KEYS = ['images', 'type', 'annotations', 'categories']
MERGE_KEYS = ['images', 'annotations']

out = {}
tot_anns = 0
for i, file_name in enumerate(INPUT_FILES):
  data = json.load(open(ANNOT_PATH + file_name, 'r'))
  print('keys', data.keys())
  if i == 0:
    for key in KEYS:
      out[key] = data[key]
      print(file_name, key, len(data[key]))
  else:
    out['images'] += data['images']
    for j in range(len(data['annotations'])):
      data['annotations'][j]['id'] += tot_anns
    out['annotations'] += data['annotations']
    print(file_name, 'images', len(data['images']))
    print(file_name, 'annotations', len(data['annotations']))
  tot_anns = len(out['annotations'])
print('tot', len(out['annotations']))
json.dump(out, open(OUT_PATH + OUTPUT_FILE, 'w'))
