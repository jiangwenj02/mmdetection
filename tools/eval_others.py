from argparse import ArgumentParser
import json
import numpy as np
#from Metric import Metric
from metric_polyp_multiclass import Metric
from pycocotools.coco import COCO
import numpy as np

def json_load(file_name):
    with open(file_name,'r') as f:
        data = json.load(f)
        return data

def analyze_results(anno, result, cfg, visualize=False, visualization_folder='./work_dirs/mask_rcnn_r50_fpn_1x_polyp/'):
    threshold_list = np.arange(0, 1, 0.01).tolist()
    coco = COCO(anno)
    img_ids = coco.getImgIds()

    pred = json_load(result)

    pred_lists = {}
    target_lists = {}
    for i in img_ids:
        anns_ids = coco.getAnnIds(imgIds=i,  iscrowd=None)
        anns = coco.loadAnns(anns_ids)
        target_lists[i] = []
        for ann in anns:
            info = ann['bbox']
            info.append(ann['category_id'])
            target_lists[i].append(info)
        pred_lists[i] = []

    for item in pred:
        info = [item['score']]
        info.extend(item['bbox'])
        info.append(item['category_id'])
        pred_lists[item['image_id']].append(info)

    for cls in range(1, cfg.num_clsses+1):
        out = '\n====================class {}====================='.format(cls)
        print(out)
        precision_list = []
        recall_list = []
        F1_list = []
        F2_list = []
        output_list = []
        best_f1 = 0
        best_f2 = 0
        best_f1_string = ''
        best_f2_string = ''
        for thresh in threshold_list:
            eval = Metric(visualize=visualize, visualization_root=visualization_folder+"/{:.3}/".format(thresh))
            for key in pred_lists.keys():
                pred_list = pred_lists[key]
                target_list = target_lists[key]
                pred_list = [p for p in pred_list if p[0] >= thresh]
                filtered_p = [p[1:] for p in pred_list if p[5] == cls]
                filterd_target = [p for p in target_list if p[4] == cls]
                image= None
                if visualize:
                    image, target = testset.pull_image(i)
                    image = image.astype(np.uint8)[:,:,(2,1,0)].copy()
                eval.eval_add_result(filterd_target, filtered_p,image=image, image_name=i)
            res = eval.get_result()

            F1 = res['overall']['F1']
            F2 = res['overall']['F2']
            precision_list.append(precision)
            recall_list.append(recall)
            F1_list.append(F1)
            F2_list.append(F2)
            
            out = "precision: {:.4f}  recall:  {:.4f} F1: {:.4f} F2: {:.4f} thresh: {:.4f} TP: {:3} FP: {:3} FN: {:3} FP+FN: {:3}" \
                .format(precision, recall, F1, F2, thresh, len(eval.TPs), len(eval.FPs), len(eval.FNs), len(eval.FPs)+len(eval.FNs))
            output_list.append(out)
            if F1 > best_f1:
                best_f1 = F1
                best_f1_string = out
            if F2 > best_f2:
                best_f2 = F2
                best_f2_string = out
        print(best_f1_string)
        print(best_f2_string)

def main():
    parser = ArgumentParser(description='COCO Error Analysis Tool')
    parser.add_argument('--result', default='work_dirs/mask_rcnn_r50_fpn_1x_adenomatous/result.bbox.json', help='result file (json format) path')
    parser.add_argument('--out_dir', default = './work_dirs/', help='dir to save analyze result images')
    parser.add_argument(
        '--ann',
        default='/data0/zzhang/adenomatous/test.json',
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