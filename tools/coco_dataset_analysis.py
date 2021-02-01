import copy
import os
from argparse import ArgumentParser
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from pycocotools.coco import COCO
import json
import pandas as pd
def analyze_dataset(json_file):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['font.family']='sans-serif'
    plt.rcParams['figure.figsize'] = (10.0, 5.0)

    # 这里打开你的训练集的标注，格式是COCO数据集的格式
    with open(json_file) as f:
        ann=json.load(f)
        
    category_dic=dict([(i['id'],i['name']) for i in ann['categories']])
    counts_label=dict([(i['name'],0) for i in ann['categories']])

    for i in ann['annotations']:
        counts_label[category_dic[i['category_id']]]+=1
        
    box_w = []
    box_h = []
    box_wh = []
    categorys_wh = [[] for j in range(204)]
    for a in ann['annotations']:
        if a['category_id'] != 0:
            box_w.append(round(a['bbox'][2],2))
            box_h.append(round(a['bbox'][3],2))
            #因为anchor_ratio是指H/W，这里就统计H/W
            hw = a['bbox'][3]/a['bbox'][2]
            if hw > 1:
                hw = round(hw, 0)
            else:
                hw = round(hw, 1)
            box_wh.append(hw)
            categorys_wh[a['category_id']-1].append(hw)
            
    # 所有标签的长宽高比例
    box_wh_unique = list(set(box_wh))
    box_wh_unique.sort()
    box_wh_count=[box_wh.count(i) for i in box_wh_unique]

    # 绘图
    wh_df = pd.DataFrame(box_wh_count,index=box_wh_unique,columns=['宽高比数量'])
    wh_df.plot(kind='bar',color="#55aacc")
    plt.show()


def main():
    parser = ArgumentParser(description='COCO Dataset Analysis Tool')
    parser.add_argument(
        '--ann',
        default='data/left.json',
        help='annotation file path')
    args = parser.parse_args()
    analyze_dataset(args.ann)


if __name__ == '__main__':
    main()
