import os
import argparse
import os.path as osp
import numpy as np
import PIL.Image
import PIL.ImageDraw
from tqdm import tqdm
from pycocotools.coco import COCO
import cv2 as cv
import json
import pycocotools.mask as mask_util

def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert Coco bbox annotations to instance segmentation by grabcut')
    parser.add_argument(
        '--ann',
        default='/data3/publicData/PASCALVOC/VOCdevkit/voc12_trainval.json')
    parser.add_argument(
        '--out',
        default='/data3/publicData/PASCALVOC/VOCdevkit/voc12_trainval_grab.json')
    parser.add_argument(
        '--imgdir',
        default='/data3/publicData/PASCALVOC/VOCdevkit/')
    args = parser.parse_args()
    return args
 
def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)

POOL_SIZE = 4 # multiprocessing para
MINI_AREA = 9 #custom cv2.grabcut break threshold
ITER_NUM = 5  # iterative parameter of grabcut
RECT_SHRINK = 3  # custom parameter for big bounding box 
IOU_THRESHOLD = 0.15  # cue C2 custom parameter in SDI paper section3.1

def json_load(file_name):
    with open(file_name,'r') as f:
        data = json.load(f)
        return data

def grabcut(args):
    """Generate the annotation information according to the json_file
    Args:
        json_file (string): jsonfile of coco format annotations
        export_dir (string): output path
    """

    coco = COCO(args.ann)
    data = json_load(args.ann)
    img_ids = coco.getImgIds()
    img_infos = []
    cats = coco.getCatIds()
    for i in tqdm(img_ids):
        info = coco.loadImgs([i])[0]
        anns_ids = coco.getAnnIds(imgIds=i,  iscrowd=None)
        anns = coco.loadAnns(anns_ids)
        # info['filename'] = info['file_name']
        img = cv.imread(osp.join(args.imgdir, info['file_name']))
        img_infos.append(info)      
        for ann_id, ann in zip(anns_ids, anns):
            mask = np.zeros((info['height'], info['width'])).astype(np.uint8)      
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            rect=ann['bbox']
            xmin, ymin, box_w, box_h = rect[0], rect[1], rect[2], rect[3]
            ymax, xmax = ymin + box_h, xmin + box_w

            if box_w * box_h < MINI_AREA:
                img_mask = mask[ymin:ymax, xmin:xmax] = 1
            # for big box that area == img.area(one object bbox is just the whole image)
            elif box_w * box_h == info['width'] * info['height']:
                rect = [RECT_SHRINK, RECT_SHRINK, box_w - RECT_SHRINK * 2, box_h - RECT_SHRINK * 2]
                cv.grabCut(img, mask, rect, bgdModel,fgdModel, ITER_NUM, cv.GC_INIT_WITH_RECT)
                # astype('uint8') keep the image pixel in range[0,255]
                img_mask =  np.where((mask == 0) | (mask == 2), 0, 1).astype('uint8')
            # for normal bbox:
            else:
                cv.grabCut(img, mask, rect, bgdModel,fgdModel, ITER_NUM, cv.GC_INIT_WITH_RECT)
                img_mask = np.where((mask == 0) | (mask == 2), 0, 1).astype('uint8')
                # if the grabcut output is just background(it happens in my dataset)
                if np.sum(img_mask) == 0:
                    img_mask = np.where((mask == 0), 0, 1).astype('uint8')
                # couting IOU
                # if the grabcut output too small region, it need reset to bbox mask
                box_mask = np.zeros((img.shape[0], img.shape[1]))
                box_mask[ymin:ymax, xmin:xmax] = 1
                sum_area = box_mask + img_mask
                intersection = np.where((sum_area==2), 1, 0).astype('uint8')
                union = np.where((sum_area==0), 0, 1).astype('uint8')
                IOU = np.sum(intersection) / np.sum(union)
                if IOU <= IOU_THRESHOLD:
                    img_mask = box_mask
            
            for idx, data_ann in enumerate(data['annotations']):

                if data_ann['id'] == ann_id:                    
                    data_ann['segmentation'] = mask_util.encode(np.array(img_mask[:, :, np.newaxis], order='F', dtype='uint8'))
                    data_ann['area'] = mask_util.area(data_ann['segmentation'])
                    if data_ann['iscrowd'] is None:
                        data_ann['iscrowd'] = False
                    data['annotations'][idx] = data_ann
                    
                    break
        if i > 5:
            break
    
    with open(args.out, 'w') as out_file:
        json.dump(data, out_file)
 
def main():
    args = parse_args()
    grabcut(args)
    # cooc_to_segmentation('/data0/zzhang/new_polyp_annotation_01_03/test.json', './test_anno') 
    #cooc_to_segmentation('E:/Users/jiangwenj02/Downloads/new_polyp_annotation_01_03/test.json', './test_anno')


if __name__ == '__main__':
    main()
