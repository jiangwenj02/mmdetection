from pycocotools.coco import COCO
from tqdm import tqdm 
import glob

def test_data(anns_file):
    # Specify the path to model config and checkpoint file
    # model_name = 'reppoints_moment_r50_fpn_1x_coco'
    # score_thr=0.25
    # config_file = 'configs/erosive/faster_rcnn_r50_fpn_1x_erosive.py'
    # checkpoint_file = '/data0/zzhang/mmdetection/erosive/work_dirs/'+model_name+'/epoch_83.pth'
    # build the model from a config file and a checkpoint file
    #model = init_detector(config_file, checkpoint_file, device='cuda:0')
    # test images and show the results
    set_name = 'test' #['train','test']
    #anns_file = '/data1/qilei_chen/DATA/erosive/annotations/'+set_name+'.json'
    coco_instance = COCO(anns_file)
    coco_imgs = coco_instance.imgs
    print(len(coco_imgs))
    count_images_with_anns = 0
    count_images_without_anns = 0
    count_anns = 0
    img_file_name_list = {}
    img_file_name_dict = {}
    count_zero_ann = 0
    for key in tqdm(coco_imgs):
        annIds = coco_instance.getAnnIds(imgIds= coco_imgs[key]['id'])
        anns = coco_instance.loadAnns(annIds)

        for ann in anns:
            if len(ann['segmentation']) > 0 and len(ann['bbox']) == 0:
                import pdb
                pdb.set_trace()
                print(coco_imgs[key]["file_name"])
        if not len(anns)==0:
            count_images_with_anns+=1
            count_anns+=len(anns)
        else:
            count_images_without_anns+=1
        img_file_name = coco_imgs[key]["file_name"]
        if not img_file_name in img_file_name_dict:
            img_file_name_dict[img_file_name] = 0
        else:
            img_file_name_dict[img_file_name] += 1
            print(img_file_name)
        if not img_file_name in img_file_name_list:
            img_file_name_list[img_file_name] = []
            img_file_name_list[img_file_name].append(anns)
        else:
            print(img_file_name)
            print(img_file_name_list[img_file_name])
            if len(img_file_name_list[img_file_name][0])==0:
                print(img_file_name)
                count_zero_ann+=1
            print(anns)
        '''
        img_file_name = coco_imgs[key]["file_name"]
        img_dir = os.path.join("/data1/qilei_chen/DATA/erosive/images",img_file_name)
        img = mmcv.imread(img_dir)
        result = inference_detector(model, img)
        for ann in anns:
            [x,y,w,h] = ann['bbox']
            cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (0,255,0), 2)
        model.show_result(img, result,score_thr=score_thr,bbox_color =(255,0,0),
                        text_color = (255,0,0),font_size=5,
                        out_file='/data1/qilei_chen/DATA/erosive/work_dirs/'+model_name+'/'+set_name+'_result_'+str(score_thr)+'/'+img_file_name)
        '''
    with open('non_gt.filename', 'w') as f:
        for key, value in img_file_name_list.items():            
            if len(value[0])==0:
                f.write(key + '\n')
                count_zero_ann+=1
        f.close()
    
    print(anns_file)
    print('the number of images: ', len(img_file_name_dict))
    print('the number of images with annotation: ', count_images_with_anns)
    print('the number of annotations: ', count_anns)
    print('the number of images without annotation: ', count_images_without_anns)
    print(len(img_file_name_list))
    print(count_zero_ann)


anns_dir = '/data0/zzhang/annotation/erosive2/test.json'
anns_dir = 'data/erosive2/annotations/test/fine_test.json'
files = glob.glob(anns_dir)
for ann_file in files:
    test_data(ann_file)