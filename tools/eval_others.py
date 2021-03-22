from argparse import ArgumentParser
import json
import numpy as np
#from Metric import Metric
from metric_polyp_multiclass import Metric
from pycocotools.coco import COCO
import numpy as np
import pprint
def json_load(file_name):
    with open(file_name,'r') as f:
        data = json.load(f)
        return data

def analyze_results(anno, result, cfg, visualize=False, visualization_folder='./work_dirs/mask_rcnn_r50_fpn_1x_polyp/'):
    threshold_list = np.arange(0, 1, 0.01).tolist()
    coco = COCO(anno)
    img_ids = coco.getImgIds()
    num_cls = cfg.num_clsses
    pred = json_load(result)

    pred_lists = {}
    target_lists = {}
    for i in img_ids:
        anns_ids = coco.getAnnIds(imgIds=i,  iscrowd=None)
        anns = coco.loadAnns(anns_ids)
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
        eval = Metric(visualize=visualize, visualization_root=visualization_folder+"/{:.3}/".format(thresh))
        for key in pred_lists.keys():
            pred_list = pred_lists[key]
            target_list = target_lists[key]
            pred_list = [p for p in pred_list if p[0] >= thresh]
            filtered_p = [p[1:] for p in pred_list]
            filterd_target = [p for p in target_list]
            image= None
            if visualize:
                image, target = testset.pull_image(i)
                image = image.astype(np.uint8)[:,:,(2,1,0)].copy()
            eval.eval_add_result(filterd_target, filtered_p,image=image, image_name=i)
            #break
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
    args = parser.parse_args()
    analyze_results(args.ann, args.result, args, args.visualize, args.out_dir)

if __name__ == '__main__':
    main()