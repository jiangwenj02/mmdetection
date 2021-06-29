import os
import json
from os import listdir, getcwd
from os.path import join
from pycocotools.coco import COCO
from tqdm import tqdm
classes = ['erosive', 'ulcer']

#box form[x,y,w,h]
def convert(size,box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[2] /2)*dw
    y = (box[1] + box[3] /2)*dh
    w = box[2]*dw
    h = box[3]*dh
    return (x,y,w,h)

def convert_annotation():
    # coco_instance = COCO('/data3/zzhang/annotation/erosiveulcer_fine/trainfp.json')
    coco_instance = COCO('data/erosiveulcer/trainfp.json')
    # coco_instance = COCO('E:/Users/jiangwenj02/Downloads/coco/annotations/instances_val2017.json')
    coco_imgs = coco_instance.imgs
    # sumfile = open('/data3/zzhang/annotation/erosiveulcer_fine/train.txt', 'w')
    sumfile = open('data/erosiveulcer/trainfp.txt', 'w')
    # sumfile = open('E:/Users/jiangwenj02/Downloads/coco/annotations/test.txt', 'w')
    for key in tqdm(coco_imgs):
        annIds = coco_instance.getAnnIds(imgIds= coco_imgs[key]['id'])
        file_name = coco_imgs[key]['file_name']
        # sumfile.write('/data3/zzhang/annotation/erosiveulcer_fine/train/images/' + file_name  + '\n')
        sumfile.write('data/erosiveulcer/images/' + file_name  + '\n')
        width = coco_imgs[key]['width']
        height = coco_imgs[key]['height']
        anns = coco_instance.loadAnns(annIds)
        # outfile = open('/data3/zzhang/annotation/erosiveulcer_fine/train/labels/%s.txt'%(file_name[:-4]), 'w')
        outfile = open('data/erosiveulcer/labels/%s.txt'%(file_name[:-4]), 'w')
        # outfile = open('E:/Users/jiangwenj02/Downloads/coco/labels_verify/%s.txt'%(file_name[:-4]), 'w')
        for item2 in anns:
            category_id = item2['category_id']
            class_id = category_id - 1
            box = item2['bbox']
            bb = convert((width,height),box)
            outfile.write(str(class_id)+" "+" ".join([str(a) for a in bb]) + '\n')
        outfile.close()
    sumfile.close()
            
if __name__ == '__main__':
    convert_annotation()    
