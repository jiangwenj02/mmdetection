from argparse import ArgumentParser
import json
import numpy as np
#from Metric import Metric
from metric_polyp_multiclass_our import Metric
from metric_polyp_multiclass import MetricMulticlass
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
    threshold_list = np.arange(0, 1, 0.01).tolist()
    if visualize:
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
        # eval = Metric(mode='iou', iou_thresh=0.01,visualize=visualize, visualization_root=visualization_folder+"/{:.3}/".format(thresh), classes=testset.CLASSES)
        eval = Metric(mode='center', iou_thresh=0.01,visualize=visualize, visualization_root=visualization_folder+"/{:.3}/".format(thresh), classes=testset.CLASSES)
        # eval = Metric(mode='siou', iou_thresh=0.8, visualize=visualize, visualization_root=visualization_folder+"/{:.3}/".format(thresh), classes=testset.CLASSES)
        # eval = MetricMulticlass(classes=testset.CLASSES)
        #eval_none = Metric(mode='iou', iou_thresh=0.1,visualize=visualize, visualization_root=visualization_folder+"/none/", classes=testset.CLASSES)
        for key in tqdm(pred_lists.keys()):
            pred_list = pred_lists[key]
            target_list = target_lists[key]
            pred_list = [p for p in pred_list if p[0] >= thresh and p[-1] <= num_cls]
            filtered_p = [p[1:] + p[0:1] for p in pred_list] # concat
            print(filtered_p)
            filterd_target = [p for p in target_list]
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
            # if len(pred_list) > 0 and len(target_list) == 0:
            #     eval_none.eval_add_result(filterd_target, filtered_p, image=image, image_name=filename_lists[key])
            # else:
            # print(filterd_target)
            # print(filtered_p)
            # import pdb
            # pdb.set_trace()
            eval.eval_add_result(filterd_target, filtered_p, image=image, image_name=filename_lists[key])
            #break
        # with open(visualization_folder + '/filter_filename.txt', 'w') as f:
        #     print('the number of filter image: %d' % (len(eval.filter_filename)))
        #     for filename in eval.filter_filename:
        #         f.write(filename + '\n')
        #     f.close()
        res = eval.get_result()
        F1 = res['overall']['F1']
        F2 = res['overall']['F2']
        precision_list.append(res['overall']['precision'])
        recall_list.append(res['overall']['recall'])
        F1_list.append(F1)
        F2_list.append(F2)
        
        out = "precision: {:.4f}  recall:  {:.4f} F1: {:.4f} F2: {:.4f} thresh: {:.4f} TP: {:3} FP: {:3} FN: {:3} FP+FN: {:3}" \
            .format(res['overall']['precision'], res['overall']['recall'], F1, F2, thresh, len(eval.TPs), len(eval.FPs), len(eval.FNs), len(eval.FPs)+len(eval.FNs))
        output_list.append(out)
        for i in range(num_cls):
            if res[i+1]['F1'] > best_f1_cls[i]:
                best_f1_cls[i] = res[i+1]['F1']
                best_f1_cls_string[i] = res[i+1]
                best_f1_cls_string[i]['thresh'] = thresh
                best_f1_cls_string[i]['confusion_matrix'] = res['confusion_matrix']
            if res[i+1]['F2'] > best_f2_cls[i]:
                best_f2_cls_string[i] = res[i+1]
                best_f2_cls[i] = res[i+1]['F2']
                best_f2_cls_string[i]['thresh'] = thresh
                best_f2_cls_string[i]['confusion_matrix'] = res['confusion_matrix']
        if F1 > best_f1:
            best_f1 = F1
            best_f1_string = res['overall']
            best_f1_string['thresh'] = thresh
            best_f1_string['confusion_matrix'] = res['confusion_matrix']
            best_f1_string_all = res
        if F2 > best_f2:
            best_f2 = F2
            best_f2_string = res['overall']
            best_f2_string['thresh'] = thresh
            best_f2_string['confusion_matrix'] = res['confusion_matrix']
        if res['binary']['F1'] > best_binary_f1:
            best_binary_f1 = res['binary']['F1']
            best_f1_binary_string = res['binary']
            best_f1_binary_string['thresh'] = thresh
            best_f1_binary_string['confusion_matrix'] = res['confusion_matrix']
        if res['binary']['F2'] > best_binary_f2:
            best_binary_f2 = res['binary']['F2']
            best_f2_binary_string = res['binary']
            best_f2_binary_string['thresh'] = thresh
            best_f2_binary_string['confusion_matrix'] = res['confusion_matrix']
            
    
    out = '\n====================overall====================='
    print(out)
    print(best_f1_string_all['overall'])
    # pprint.pprint(best_f1_string)
    # pprint.pprint(best_f2_string)
    for i in range(num_cls):
        out = '\n====================class {}====================='.format(i)
        pprint.pprint(out)
        print(best_f1_string_all[i+1])
        # pprint.pprint(best_f1_cls_string[i])
        # pprint.pprint(best_f2_cls_string[i])
    out = '\n====================binary====================='
    pprint.pprint(out)
    print(best_f1_string_all['binary'])
    # pprint.pprint(best_f1_binary_string)
    # pprint.pprint(best_f2_binary_string)
    # pprint.pprint(best_f1_string)
    # pprint.pprint(best_f2_string)

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
    else:
        dataset = None
    analyze_results(args.ann, args.result, args, args.visualize, args.out_dir, dataset, cfg.data.test.img_prefix)

if __name__ == '__main__':
    main()