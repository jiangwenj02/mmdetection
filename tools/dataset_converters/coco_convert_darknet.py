import os
import json
from os import listdir, getcwd
from os.path import join
from pycocotools.coco import COCO

classes = ['erosive', 'ulcer']

#box form[x,y,w,h]
def convert(size,box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = box[0]*dw
    y = box[1]*dh
    w = box[2]*dw
    h = box[3]*dh
    return (x,y,w,h)

def convert_annotation():
    coco_instance = COCO('/data3/zzhang/annotation/erosiveulcer_fine/train.json')
    coco_imgs = coco_instance.imgs
    sumfile = open('/data3/zzhang/annotation/erosiveulcer_fine/train.txt', 'w')
    for key in coco_imgs:
        annIds = coco_instance.getAnnIds(imgIds= coco_imgs[key]['id'])
        file_name = coco_imgs[key]['file_name']
        sumfile.write(file_name  + '\n')
        width = coco_imgs[key]['width']
        height = coco_imgs[key]['height']
        anns = coco_instance.loadAnns(annIds)
        outfile = open('/data3/zzhang/annotation/erosiveulcer_fine/darknetlabel/%s.txt'%(file_name[:-4]), 'w')
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


        
    
