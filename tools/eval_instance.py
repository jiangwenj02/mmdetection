from argparse import ArgumentParser
import json
import numpy as np
#from Metric import Metric
from metric_polyp_multiclass import Metric
from pycocotools.coco import COCO
import numpy as np
import pprint
from mmdet.datasets.builder import build_dataset
import mmcv
from mmcv import Config
from mmcv.image import tensor2imgs
from tqdm import tqdm
import os

def json_load(file_name):
    with open(file_name,'r') as f:
        data = json.load(f)
        return data

def analyze_results(anno, result, cfg, visualize=False, visualization_folder='./work_dirs/mask_rcnn_r50_fpn_1x_polyp/', testset=None, image_pre = ''):
    #threshold_list = np.arange(0, 1, 0.01).tolist()
    threshold_list = [0.25]
    coco = COCO(anno)
    img_ids = coco.getImgIds()
    num_cls = cfg.num_clsses
    pred = json_load(result)

    pred_lists = {}
    target_lists = {}
    filename_lists = {}
    for i in img_ids:
        anns_ids = coco.getAnnIds(imgIds=i,  iscrowd=None)
        anns = coco.loadAnns(anns_ids)
        image_info = coco.loadImgs(i)
        filename_lists[i] = image_info[0]['file_name']
        target_lists[i] = []
        for ann in anns:
            ann['bbox'][2] += ann['bbox'][0]
            ann['bbox'][3] += ann['bbox'][1]
            info = ann['bbox']
            info.append(ann['category_id'])
            target_lists[i].append(info)
        pred_lists[i] = []

    for item in pred:
        info = [item['score']]
        item['bbox'][2] += item['bbox'][0]
        item['bbox'][3] += item['bbox'][1]
        info.extend(item['bbox'])
        info.append(item['category_id'])
        pred_lists[item['image_id']].append(info)

    # for cls in range(1, cfg.num_clsses+1):
    #     out = '\n====================class {}====================='.format(cls)
    #     print(out)
    precision_list = []
    recall_list = []
    F1_list = []
    F2_list = []
    output_list = []
    best_f1 = 0
    best_f2 = 0
    best_binary_f1 = 0
    best_binary_f2 = 0

    best_f1_cls = [0 for i in range(num_cls)]
    best_f2_cls = [0 for i in range(num_cls)]
    best_f1_cls_string = ['' for i in range(num_cls)]
    best_f2_cls_string = ['' for i in range(num_cls)]
    best_f1_string = ''
    best_f2_string = ''
    best_f1_binary_string = ''
    best_f2_binary_string = ''
    for thresh in threshold_list:
        eval = Metric(visualize=visualize, visualization_root=visualization_folder+"/{:.3}/".format(thresh), classes=testset.CLASSES)
        for key in tqdm(pred_lists.keys()):            
            pred_list = pred_lists[key]
            target_list = target_lists[key]
            pred_list = [p for p in pred_list if p[0] >= thresh]
            filtered_p = [p[1:] for p in pred_list]
            filterd_target = [p for p in target_list]
            if len(filterd_target) == 0 and len(filtered_p) > 0:
                print(filename_lists[key])
            else:
                continue
            image= None
            if visualize:
                image = mmcv.imread(os.path.join(image_pre, filename_lists[key]))
                # item = testset.__getitem__(key)
                #img_tensor = item['img'].data.unsqueeze(0)
                #img_metas = item['img_metas'].data

                # img_tensor = item['img'][0].unsqueeze(0)
                # img_metas = item['img_metas'][0].data
                # img = tensor2imgs(img_tensor, **img_metas['img_norm_cfg'])[0]
                # h, w, _ = img_metas['img_shape']
                # ori_h, ori_w = img_metas['ori_shape'][:-1]
                # image = img[:h, :w, :]
                # image = mmcv.imresize(image, (ori_w, ori_h))
                #image = image.astype(np.uint8)[:,:,(2,1,0)].copy()
            eval.eval_add_result(filterd_target, filtered_p,image=image, image_name=filename_lists[key])
            #break
            res = eval.get_result()



def retrieve_data_cfg(config_path, skip_type):
    cfg = Config.fromfile(config_path)
    train_data_cfg = cfg.data.test
    if train_data_cfg.type == 'RepeatDataset':
        train_data_cfg = train_data_cfg.dataset
    train_data_cfg['pipeline'] = [
        x for x in train_data_cfg.pipeline if x['type'] not in skip_type
    ]

    return cfg

def main():
    parser = ArgumentParser(description='COCO Error Analysis Tool')
    parser.add_argument('--result', default='work_dirs/mask_rcnn_r50_fpn_1x_adenomatous/result.bbox.json', help='result file (json format) path')
    parser.add_argument('--out_dir', default = './work_dirs/', help='dir to save analyze result images')
    parser.add_argument(
        '--ann',
        default='/data0/zzhang/annotations/adenomatous/test.json',
        #default='data/adenomatous/test.json',
        help='annotation file path')
    parser.add_argument(
        '--visualize',
        type=bool,
        default=False,
        help='annotation file path')
    parser.add_argument(
        '--num_clsses', type=int, default=2)
    parser.add_argument('--config', default=None, help='train config file path')
    parser.add_argument(
        '--skip-type',
        type=str,
        nargs='+',
        default=['DefaultFormatBundle', 'Normalize', 'Collect'],
        help='skip some useless pipeline')
    args = parser.parse_args()
    if args.config is not None:
        cfg = retrieve_data_cfg(args.config, args.skip_type)
        cfg.data.test.test_mode = True
        dataset = build_dataset(cfg.data.test)
    analyze_results(args.ann, args.result, args, args.visualize, args.out_dir, dataset, cfg.data.test.img_prefix)

if __name__ == '__main__':
    main()