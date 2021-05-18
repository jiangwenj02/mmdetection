import numpy as np
from mmdet.apis import init_detector, inference_detector
import mmcv
from pycocotools.coco import COCO
import os
import cv2
import json
import pickle
from metric_polyp import Metric
from metric_polyp_multiclass import MetricMulticlass
from img_crop import crop_img
from extra_nms import *
import datetime
NMS_ALL = True
show_time = False
def convert_result(bbox_result):
    json_result = dict()
    json_result['results'] = []
    for label in range(len(bbox_result)):
        bboxes = bbox_result[label]
        for i in range(bboxes.shape[0]):
            data = dict()
            data['bbox'] = xyxy2xywh(bboxes[i])
            data['score'] = float(bboxes[i][4])
            data['label'] = int(label+1)
            data['location'] = (label+1,i)
            json_result['results'].append(data) 
    return json_result

def test_data(with_gt=False):
    # Specify the path to model config and checkpoint file
    model_name = 'reppoints_moment_r50_fpn_1x_coco'
    score_thr = 0.3

    config_file = 'configs/erosive/'+model_name+'.py'
    checkpoint_file = '/data1/qilei_chen/DATA/erosive/work_dirs/' + \
        model_name+'/epoch_83.pth'

    # build the model from a config file and a checkpoint file
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # test images and show the results
    set_name = 'test'  # ['train','test']
    anns_file = '/data1/qilei_chen/DATA/erosive/annotations/'+set_name+'.json'
    coco_instance = COCO(anns_file)
    coco_imgs = coco_instance.imgs

    count_images_with_anns = 0
    count_images_without_anns = 0
    count_anns = 0
    img_file_name_list = {}
    count_zero_ann = 0

    for key in coco_imgs:
        # print(key)
        # print(coco_imgs[key]['id'])
        annIds = coco_instance.getAnnIds(imgIds=coco_imgs[key]['id'])
        anns = coco_instance.loadAnns(annIds)
        if not len(anns) == 0:
            count_images_with_anns += 1
            count_anns += len(anns)
        else:
            count_images_without_anns += 1
        img_file_name = coco_imgs[key]["file_name"]
        if not img_file_name in img_file_name_list:
            img_file_name_list[img_file_name] = []
            img_file_name_list[img_file_name].append(anns)
        else:
            # print(img_file_name)
            # print(img_file_name_list[img_file_name])
            if len(img_file_name_list[img_file_name][0]) == 0:
                count_zero_ann += 1
            # print(anns)

        img_file_name = coco_imgs[key]["file_name"]
        img_dir = os.path.join(
            "/data1/qilei_chen/DATA/erosive/images", img_file_name)
        img = mmcv.imread(img_dir)

        result = inference_detector(model, img)
        if with_gt:
            for ann in anns:
                [x, y, w, h] = ann['bbox']
                cv2.rectangle(img, (int(x), int(y)),
                              (int(x+w), int(y+h)), (0, 255, 0), 2)

        model.show_result(img, result, score_thr=score_thr, bbox_color=(255, 0, 0),
                          text_color=(255, 0, 0), font_size=5,
                          out_file='/data1/qilei_chen/DATA/erosive/work_dirs/'+model_name+'/'+set_name+'_result_'+str(score_thr)+'/'+img_file_name)

    print(count_images_with_anns)
    print(count_anns)
    print(count_images_without_anns)
    print(len(img_file_name_list))
    print(count_zero_ann)


colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
classes = ('ulcer', 'erosive')
classes = ('Adenomatous','non-Adenomatous')
classes = ('erosive',)

def inference_and_save_result(model, coco_instance, img_folder_dir,
                              result_save_dir, imshow=False, score_thr=0.3):
    coco_imgs = coco_instance.imgs
    results = dict()
    for key in coco_imgs:
        img_file_name = coco_imgs[key]["file_name"]
        img_dir = os.path.join(img_folder_dir, img_file_name)
        img = mmcv.imread(img_dir)

        import datetime
        time0 = datetime.datetime.now()
        result = inference_detector(model, img)
        time1 = datetime.datetime.now()
        #print("--------inference_detector process--------")
        # print((time1-time0).microseconds/1000)

        if imshow:
            annIds = coco_instance.getAnnIds(imgIds=coco_imgs[key]['id'])
            anns = coco_instance.loadAnns(annIds)
            for ann in anns:
                [x, y, w, h] = ann['bbox']
                # print(ann['category_id'])
                cv2.putText(img, classes[ann['category_id']-1], (int(x), int(
                    y)), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[ann['category_id']-1], 2, cv2.LINE_AA)
                cv2.rectangle(img, (int(x), int(y)), (int(x+w),
                                                      int(y+h)), colors[ann['category_id']-1], 2)
            out_file = result_save_dir+'_result_' + \
                str(score_thr)+'/'+img_file_name
            model.show_result(img, result, score_thr=score_thr, bbox_color=colors[2],
                              text_color=colors[2], font_size=10,
                              out_file=out_file)

        results[coco_imgs[key]['id']] = dict()
        results[coco_imgs[key]['id']]['file_name'] = img_file_name
        results[coco_imgs[key]['id']]['result'] = result
        #print(result)

    with open(result_save_dir, 'wb') as fp:
        pickle.dump(results, fp)


def draw_result(show_result, coco_instance, img_folder_dir,
                result_save_dir, imshow=False, score_thr=0.3):
    coco_imgs = coco_instance.imgs
    results = pickle.load(open(result_save_dir, 'rb'))
    ids_with_ann = set(_['image_id'] for _ in coco_instance.anns.values())
    #for key in coco_imgs:
    for key in ids_with_ann:
        img_file_name = coco_imgs[key]["file_name"]
        img_dir = os.path.join(img_folder_dir, img_file_name)
        img = mmcv.imread(img_dir)
        annIds = coco_instance.getAnnIds(imgIds=coco_imgs[key]['id'])
        anns = coco_instance.loadAnns(annIds)
        for ann in anns:
            [x, y, w, h] = ann['bbox']
            cv2.putText(img, classes[ann['category_id']-1], (int(x), int(
                y)), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[ann['category_id']-1], 2, cv2.LINE_AA)
            cv2.rectangle(img, (int(x), int(y)), (int(x+w),
                                                  int(y+h)), colors[ann['category_id']-1], 2)
        out_file = result_save_dir+'_result_'+str(score_thr)+'_withoutempty/'+img_file_name
        show_result(img, results[coco_imgs[key]['id']]['result'], score_thr=score_thr, bbox_color=colors[2],
                    text_color=colors[2], font_size=10,
                    out_file=out_file)


def generate_result(model_name, work_dir, model_epoch,
                    coco_instance,data_set_name="ulcer", set_name='test', imshow=False, score_thr=0.3):
    # Specify the path to model config and checkpoint file
    #model_name = 'reppoints_moment_r50_fpn_1x_coco'

    config_file = 'configs/'+data_set_name+'/'+model_name.replace("_fine","")+'.py'
    checkpoint_file = os.path.join(work_dir, model_name, model_epoch)

    # build the model from a config file and a checkpoint file

    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    if os.path.exists(checkpoint_file+"_"+set_name+".pkl"):
        draw_result(model.show_result, coco_instance, '/data1/qilei_chen/DATA/'+data_set_name+'/images',
                    checkpoint_file+"_"+set_name+".pkl", imshow=imshow, score_thr=score_thr)
    else:
        inference_and_save_result(model, coco_instance, '/data1/qilei_chen/DATA/'+data_set_name+'/images',
                                  checkpoint_file+"_"+set_name+".pkl", imshow=imshow, score_thr=score_thr)

    return checkpoint_file+"_"+set_name+".pkl"


def xyxy2xywh(box):
    return [box[0], box[1], box[2]-box[0], box[3]-box[1]]


def xyxycenter(box):
    return ((box[0]+box[2])/2, (box[1]+box[3])/2)


def xywh2xyxy(box):
    return [box[0], box[1], box[2]+box[0], box[3]+box[1]]


def xywhcenter(box):
    return (box[0]+box[2]/2, box[1]+box[3]/2)


def center_in_xyxyrule(Point, Bbox):
    if Point[0] >= Bbox[0] and Point[1] >= Bbox[1] and Point[0] <= Bbox[0]+Bbox[2] and Point[1] <= Bbox[1]+Bbox[3]:
        return True
    return False


def center_in_xywhrule(Point, Bbox):
    if Point[0] >= Bbox[0] and Point[1] >= Bbox[1] and Point[0] <= Bbox[2] and Point[1] <= Bbox[3]:
        return True
    return False


def filt_boxes(boxes_with_scores, categories,thres):
    filted_boxes = []
    if NMS_ALL:
        json_result = convert_result(boxes_with_scores)

        nms_result(json_result)

        all_nms_locs = []

        for jr in json_result["results"]:
            all_nms_locs.append(jr['location'])

    for category in categories:
        for box,i in zip(boxes_with_scores[category-1],range(boxes_with_scores[category-1].shape[0])):
            if box[4] >= thres:
                box[4] = category #replace score with category
                if NMS_ALL:
                    if (category,i) in all_nms_locs:
                        filted_boxes.append(box)
                else:
                    filted_boxes.append(box)
    return filted_boxes


def anns2gtboxes(gtanns,categories):
    gtboxes = []
    for ann in gtanns:
        if ann['category_id'] in categories:
            xyxy = xywh2xyxy(ann['bbox'])
            xyxy.append( ann['category_id'])
            gtboxes.append(xyxy)
    return gtboxes


def peval(result_dir, coco_instance, thresh=0.3, with_empty_images=True):
    categories = [1]
    #print(categories)
    fp = open(result_dir, 'rb')
    results = pickle.load(fp)
    eval = Metric()

    for img_id in results:
        filed_boxes = filt_boxes(results[img_id]['result'],categories, thresh)
        gtannIds = coco_instance.getAnnIds(imgIds=img_id)
        gtanns = coco_instance.loadAnns(gtannIds)
        if len(gtanns) == 0 and (not with_empty_images):
            continue
        gtboxes = anns2gtboxes(gtanns,categories)
        eval.eval_add_result(gtboxes, filed_boxes,dup_TP=True)

    precision, recall = eval.get_result()
    F1 = 2 * (precision * recall) / max((precision + recall), 1e-5)
    F2 = 5 * (precision * recall) / max((4 * precision + recall), 1e-5)
    out = "precision: {:.4f}  recall:  {:.4f} F1: {:.4f} F2: {:.4f} thresh: {:.4f} TP: {:3} FP: {:3} FN: {:3} FP+FN: {:3}" \
        .format(precision, recall, F1, F2, thresh, len(eval.TPs), len(eval.FPs), len(eval.FNs), len(eval.FPs)+len(eval.FNs))
    print(out)

def peval_m(result_dir, coco_instance, thresh=0.3, with_empty_images=True,categories = [1,2]):
    
    fp = open(result_dir, 'rb')
    results = pickle.load(fp)
    eval_m = MetricMulticlass()

    category = eval_m.classes

    for img_id in results:
        filed_boxes = filt_boxes(results[img_id]['result'],categories, thresh)
        gtannIds = coco_instance.getAnnIds(imgIds=img_id)
        gtanns = coco_instance.loadAnns(gtannIds)
        if len(gtanns) == 0 and (not with_empty_images):
            continue
        gtboxes = anns2gtboxes(gtanns,categories)

        eval_m.eval_add_result(gtboxes, filed_boxes)

    evaluation = eval_m.get_result()
    for key in evaluation:
        if key in ['overall', 'binary']:
            print('\n==================== {} ====================='.format(key))
        elif key == 'confusion_matrix':
            continue
        else:
            print('\n==================== {} ====================='.format(
                category[key - 1]))
        print("Precision: {:.4f}  Recall: {:.4f}  F1: {:.4f}  F2: {:.4f}  "
            "TP: {:3}  FP: {:3}  FN: {:3}  FP+FN: {:3}"
            .format(evaluation[key]['precision'],
                    evaluation[key]['recall'],
                    evaluation[key]['F1'],
                    evaluation[key]['F2'],
                    evaluation[key]['TP'],
                    evaluation[key]['FP'],
                    evaluation[key]['FN'],
                    evaluation[key]['FN'] + evaluation[key]['FP']))
    template = "{:^20}"
    out_t = ''
    out = []
    for i in range(len(category) + 1):
        out_t = out_t + template
    out.append(out_t.format('\n  gt class -->', *[i for i in category]))
    total_proposal = 0
    for i in range(1, len(category) + 1):
        cm = []
        for j in range(1, len(category) + 1):
            cm.append(evaluation['confusion_matrix'][i][j])
            total_proposal += evaluation['confusion_matrix'][i][j]
        out.append(out_t.format(category[i - 1], *cm))
    print('\n'.join(out))
    print('\nTotal proposals: {}, Accuracy: {:.4f}'.format(total_proposal,
                                                        (evaluation['confusion_matrix'][1][1] +
                                                            evaluation['confusion_matrix'][2][2]) / total_proposal))

def getResult(imgid, json_results):
    results = []
    for result in json_results:
        if imgid == result["image_id"]:
            box = xywh2xyxy(result["bbox"])
            box.append(result['score'])
            results.append(box)

    return results


def peval_yolof(result_dir, coco_instance, thresh=0.00, with_empty_images=True):
    print("-------"+str(thresh)+"-------")
    fp = open(result_dir, 'rb')
    results = json.load(fp)
    eval = Metric()
    coco_imgs = coco_instance.imgs
    for img_id in coco_imgs:
        filed_boxes = filt_boxes(getResult(img_id, results), thresh)
        gtannIds = coco_instance.getAnnIds(imgIds=img_id)
        gtanns = coco_instance.loadAnns(gtannIds)
        gtboxes = anns2gtboxes(gtanns)
        if len(gtboxes) == 0 and (not with_empty_images):
            continue
        eval.eval_add_result(gtboxes, filed_boxes)

    precision, recall = eval.get_result()
    F1 = 2 * (precision * recall) / max((precision + recall), 1e-5)
    F2 = 5 * (precision * recall) / max((4 * precision + recall), 1e-5)
    out = "precision: {:.4f}  recall:  {:.4f} F1: {:.4f} F2: {:.4f} thresh: {:.4f} TP: {:3} FP: {:3} FN: {:3} FP+FN: {:3}" \
        .format(precision, recall, F1, F2, thresh, len(eval.TPs), len(eval.FPs), len(eval.FNs), len(eval.FPs)+len(eval.FNs))
    print(out)


def eval_yolof(coco_instance):

    results_file_dir = "/data1/qilei_chen/DATA/erosive/work_dirs_yolof/R_50_C5_1x/inference/coco_instances_results.json"
    for thresh in np.linspace(0, 1, 10, endpoint=False):
        peval_yolof(results_file_dir, coco_instance,
                    thresh=thresh, with_empty_images=False)


def getResultbyName(file_name, json_results):
    results = []
    for result in json_results:
        if file_name == result["image_id"]:
            box = xywh2xyxy(result["bbox"])
            box.append(result['score'])
            results.append(box)

    return results


def peval_yolov5(result_dir, coco_instance, thresh=0.3, with_empty_images=True):
    print("-------"+str(thresh)+"-------")
    fp = open(result_dir, 'rb')
    results = json.load(fp)
    eval = Metric()
    coco_imgs = coco_instance.imgs
    for img_id in coco_imgs:
        file_name = coco_imgs[img_id]['file_name']
        filed_boxes = filt_boxes(getResultbyName(
            file_name[:-4], results), thresh)
        gtannIds = coco_instance.getAnnIds(imgIds=img_id)
        gtanns = coco_instance.loadAnns(gtannIds)
        gtboxes = anns2gtboxes(gtanns)
        if len(gtboxes) == 0 and (not with_empty_images):
            continue
        eval.eval_add_result(gtboxes, filed_boxes)

    precision, recall = eval.get_result()
    F1 = 2 * (precision * recall) / max((precision + recall), 1e-5)
    F2 = 5 * (precision * recall) / max((4 * precision + recall), 1e-5)
    out = "precision: {:.4f}  recall:  {:.4f} F1: {:.4f} F2: {:.4f} thresh: {:.4f} TP: {:3} FP: {:3} FN: {:3} FP+FN: {:3}" \
        .format(precision, recall, F1, F2, thresh, len(eval.TPs), len(eval.FPs), len(eval.FNs), len(eval.FPs)+len(eval.FNs))
    print(out)


def eval_yolov5(coco_instance):
    results_file_dir = "/data1/qilei_chen/DEVELOPMENTS/yolov5/runs/test/exp5/best_predictions.json"
    for thresh in np.linspace(0, 1, 10, endpoint=False):
        peval_yolov5(results_file_dir, coco_instance,
                     thresh=thresh, with_empty_images=False)



def test_images(model_name = 'cascade_rcnn_r50_fpn_1x_coco_fine',model_epoch = 'epoch_9.pth'):
    # test images and show the results
    # test_data()

    sets = ['train', 'test']
    set_name = sets[1]
    #anns_file = '/data1/qilei_chen/DATA/polyp_xinzi/annotations/'+set_name+'.json'
    #anns_file = '/data1/qilei_chen/DATA/erosive/annotations/'+set_name+'4.19.json'
    anns_file = '/data1/qilei_chen/DATA/erosive/annotations/fine_'+set_name+'.json'
    coco_instance = COCO(anns_file)

    
    
    '''
    model_name = 'ssd384_coco'
    model_epoch = 'epoch_17.pth'
    '''
    '''
    model_name = 'faster_rcnn_r50_fpn_1x_coco_384'
    model_epoch = 'epoch_14.pth'
    '''

    #model_name = 'reppoints_moment_r50_fpn_1x_coco_4_19'
    #model_epoch = 'epoch_61.pth'

    #work_dir = '/data1/qilei_chen/DATA/polyp_xinzi/work_dirs/'
    work_dir = '/data1/qilei_chen/DATA/erosive/work_dirs/'

    print("----------------")
    print(model_name)
    print("----------------")

    results_file_dir = os.path.join(
        work_dir, model_name, model_epoch+"_"+set_name+".pkl")
    #results_file_dir = generate_result(
    #    model_name, work_dir, model_epoch, coco_instance,data_set_name = 'erosive', set_name = set_name, imshow=True)
    for thresh in range(0,100,5):
        thresh = float(thresh)/100
        print('------------threshold:'+str(thresh)+'--------------')
        peval(results_file_dir, coco_instance,
              thresh=thresh, with_empty_images=False)

    # eval_yolof(coco_instance)
    # eval_yolov5(coco_instance)



def test_video():

    #video_dir = "/data0/dataset/Xiangya_Gastric_data/2021_gastric_video_annotation/20191111-1120/20191120080002-00.23.16.084-00.27.17.158-seg2.avi"
    
    source_list = open("video_list.txt")

    #video_dir = "/data1/qilei_chen/DATA/20191120080002-00.23.16.084-00.27.17.158-seg2.avi"
    model_name = "cascade_rcnn_r50_fpn_1x_coco"
    categories = ["ulcer","erosive"]
    category = categories[1]
    print("-----------------")
    print(model_name)
    print(category)
    print("-----------------")
    if category==categories[0]:
        #for ulcer
        model_shresh={"faster_rcnn_r50_fpn_1x_coco":0.4,"cascade_rcnn_r50_fpn_1x_coco":0.3}
    else:
        #for erosive
        model_shresh={"faster_rcnn_r50_fpn_1x_coco":0.3,"cascade_rcnn_r50_fpn_1x_coco":0.3}
    
    config_file = 'configs/'+category+'/'+model_name+'.py'
    anno_date = "_4_19"
    #anno_date = ""
    checkpoint_file = '/data1/qilei_chen/DATA/'+category+'/work_dirs/'+model_name+anno_date+"/epoch_9.pth"
    
    score_thr = model_shresh[model_name]
    # build the model from a config file and a checkpoint file

    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    line = source_list.readline()
    while line:
        
        file_name = os.path.basename(line[:-1])
        video_dir = os.path.join("/data1/qilei_chen/DATA",file_name)
        if not os.path.exists(video_dir):
            command = "cp /data0/dataset/Xiangya_Gastric_data/"+line[:-1]+" /data1/qilei_chen/DATA/"
            os.system(command)
        
        src_cap = cv2.VideoCapture(video_dir)

        fps = src_cap.get(cv2.CAP_PROP_FPS)
        frame_size = (int(src_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(src_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if not os.path.exists('/data1/qilei_chen/DATA/'+category+'/video_test_results/'+model_name+anno_date):
            os.makedirs('/data1/qilei_chen/DATA/'+category+'/video_test_results/'+model_name+anno_date)
        
        save_dir = os.path.join('/data1/qilei_chen/DATA/'+category+'/video_test_results/',model_name+anno_date, os.path.basename(video_dir))
        if not os.path.exists(save_dir+".pkl"):
            print(file_name)
            dst_writer = cv2.VideoWriter(save_dir, cv2.VideoWriter_fourcc("P", "I", "M", "1"), fps, frame_size)
            
            positive_records = open(save_dir+".txt","w")

            success, frame = src_cap.read()
            count = 0

            results = dict()

            while success:
                
                ##preprocess todo
                frame = crop_img(frame)

                result = inference_detector(model, frame)
                results[count] =convert_result(result)
                frame = model.show_result(frame, result, score_thr=score_thr, bbox_color=colors[2],
                                    text_color=colors[2], font_size=10)
                
                cv2.putText(frame,str(count),(30,30),cv2.FONT_HERSHEY_SIMPLEX, 1,colors[2],1,cv2.LINE_AA)
                #cv2.imwrite('/data1/qilei_chen/DATA/'+category+'/video_test_results/test.jpg',frame)
                
                box_count=0
                #print(len(result[0]))
                for box in result[0]:
                    if box[4]>=score_thr:
                        box_count+=1

                #print(box_count)
                dst_writer.write(cv2.resize(frame,frame_size))
                #print(str(count)+" "+str(box_count)+" "+str(box_count!=0)+"\n")
                positive_records.write(str(count)+" "+str(box_count)+" "+str(box_count!=0)+"\n")   

                count +=1

                success, frame = src_cap.read()
            with open(save_dir+".pkl", 'wb') as outfile:
                pickle.dump(results, outfile)
            #with open(save_dir+".json", 'w') as outfile:
            #    json.dump(results, outfile)        
        
        line = source_list.readline()

def test_video_batch(batch_size = 8):

    #video_dir = "/data0/dataset/Xiangya_Gastric_data/2021_gastric_video_annotation/20191111-1120/20191120080002-00.23.16.084-00.27.17.158-seg2.avi"
    
    source_list = open("video_list.txt")

    #video_dir = "/data1/qilei_chen/DATA/20191120080002-00.23.16.084-00.27.17.158-seg2.avi"
    model_name = "faster_rcnn_r50_fpn_1x_coco"
    categories = ["ulcer","erosive"]
    category = categories[0]
    print("-----------------")
    print(model_name)
    print(category)
    print("-----------------")
    if category==categories[0]:
        #for ulcer
        model_shresh={"faster_rcnn_r50_fpn_1x_coco":0.5,"cascade_rcnn_r50_fpn_1x_coco":0.3}
    else:
        #for erosive
        model_shresh={"faster_rcnn_r50_fpn_1x_coco":0.5,"cascade_rcnn_r50_fpn_1x_coco":0.3}
    
    config_file = 'configs/'+category+'/'+model_name+'.py'
    #anno_date = "_4_19"
    anno_date = ""
    #anno_date = "_fine"
    checkpoint_file = '/data1/qilei_chen/DATA/'+category+'/work_dirs/'+model_name+anno_date+"/epoch_9.pth"
    
    score_thr = model_shresh[model_name]
    # build the model from a config file and a checkpoint file

    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    line = source_list.readline()
    while line:
        
        file_name = os.path.basename(line[:-1])
        video_dir = os.path.join("/data1/qilei_chen/DATA",file_name)
        if not os.path.exists(video_dir):
            command = "cp /data0/dataset/Xiangya_Gastric_data/"+line[:-1]+" /data1/qilei_chen/DATA/"
            os.system(command)
        
        src_cap = cv2.VideoCapture(video_dir)

        fps = src_cap.get(cv2.CAP_PROP_FPS)
        frame_size = (int(src_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(src_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if not os.path.exists('/data1/qilei_chen/DATA/'+category+'/video_test_results/'+model_name+anno_date):
            os.makedirs('/data1/qilei_chen/DATA/'+category+'/video_test_results/'+model_name+anno_date)
        
        save_dir = os.path.join('/data1/qilei_chen/DATA/'+category+'/video_test_results/',model_name+anno_date, os.path.basename(video_dir))
        if not os.path.exists(save_dir+".pkl"):
            print(file_name)
            dst_writer = cv2.VideoWriter(save_dir, cv2.VideoWriter_fourcc("P", "I", "M", "1"), fps, frame_size)
            
            positive_records = open(save_dir+".txt","w")

            success, frame = src_cap.read()
            count = 0

            results = dict()

            frame_batch = []
            
            while success:
                
                ##preprocess todo
                frame = crop_img(frame)
                
                frame_batch.append(frame)

                if len(frame_batch)==batch_size:
                    start_time=datetime.datetime.now()
                    result = inference_detector(model, frame_batch)
                    end_time=datetime.datetime.now()
                    if show_time:
                        print("inference batch time:")
                        print((end_time-start_time).microseconds)
                        print("--------------------------")
                    for i in range(batch_size):
                        results[count] =convert_result(result[i])
                        
                        frame = model.show_result(frame_batch[i], result[i], score_thr=score_thr, bbox_color=colors[2],
                                            text_color=colors[2], font_size=10)
                        
                        cv2.putText(frame_batch[i],str(count),(30,30),cv2.FONT_HERSHEY_SIMPLEX, 1,colors[2],1,cv2.LINE_AA)
                        #cv2.imwrite('/data1/qilei_chen/DATA/'+category+'/video_test_results/test.jpg',frame)
                        
                        box_count=0
                        #print(len(result[0]))
                        for box in result[i][0]:
                            if box[4]>=score_thr:
                                box_count+=1

                        #print(box_count)
                        dst_writer.write(cv2.resize(frame,frame_size))
                        #print(str(count)+" "+str(box_count)+" "+str(box_count!=0)+"\n")
                        positive_records.write(str(count)+" "+str(box_count)+" "+str(box_count!=0)+"\n")   
                        
                        count +=1
                    
                    frame_batch = []
                success, frame = src_cap.read()
            #dst_writer.close()
            with open(save_dir+".pkl", 'wb') as outfile:
                pickle.dump(results, outfile)
            #with open(save_dir+".json", 'w') as outfile:
            #    json.dump(results, outfile)        
        
        line = source_list.readline()

if __name__ == "__main__":
    '''
    test_images(model_name = 'cascade_rcnn_r50_fpn_1x_coco_fine',model_epoch = 'epoch_9.pth')
    
    test_images(model_name = 'reppoints_moment_r50_fpn_1x_coco_fine',model_epoch = 'epoch_32.pth')
    test_images(model_name = 'retinanet_r50_fpn_1x_coco_fine',model_epoch = 'epoch_22.pth')
    
    test_images(model_name = 'faster_rcnn_mobilev2_fpn_1x_coco_fine',model_epoch = 'epoch_22.pth')
    test_images(model_name = 'faster_rcnn_r50_fpn_1x_coco_fine',model_epoch = 'epoch_9.pth')
    '''
    test_video_batch()