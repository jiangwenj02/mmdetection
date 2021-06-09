'''
Descripttion: 
version: 
Author: LiQiang
Date: 2021-01-21 11:45:22
LastEditTime: 2021-01-21 13:05:07
'''
import argparse
 
import cv2
import torch
import os 
import os.path as osp
import glob
from mmdet.apis import inference_detector, init_detector
from mmcv.utils import mkdir_or_exist
from tqdm import tqdm
def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--video_in_dir', type=str, default='',help='test video path')
    parser.add_argument(
    '--video_out_dir', type=str, default='', help='output video path')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument(
        '--show', type=bool, default=False, help='bbox score threshold')
    args = parser.parse_args()
    return args

def list_files(path, ends):
    files = []
    list_dir = os.walk(path)
    for maindir, subdir, all_file in list_dir:
        
        for filename in all_file:
            apath = os.path.join(maindir, filename)
            if apath.endswith(ends):
                files.append(apath)
    return files

def detectvideo(model, video_in, video_out, args):
    cap = cv2.VideoCapture(video_in)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #获取视频帧率
    #设置写入视频的编码格式
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    ####重要
    videoWriter = cv2.VideoWriter(video_out, fourcc, fps_video, (frame_width, frame_height))
    count=0
    print('Press "Esc", "q" or "Q" to exit.')
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(range(length))
    for i in pbar:
        torch.cuda.empty_cache()
        ret_val, img = cap.read()
        if ret_val:
            if count<0:
                count+=1 
                print('Write {} in result Successfully!'.format(count))
                continue
            #############################
            result = inference_detector(model, img)
            # ch = cv2.waitKey(1)
            # if ch == 27 or ch == ord('q') or ch == ord('Q'):
            #     break
            frame=model.show_result(
                img, result, score_thr=args.score_thr, wait_time=1, show=False,thickness=1)
            if args.show:
                cv2.imshow('frame',frame)
            if len(frame)>=1 or frame:
                #写入视频
                videoWriter.write(frame)
                count+=1
            #############################
            """
            # if count%24==0:  #快些看到效果
            #     result = inference_detector(model, img)
            #     ch = cv2.waitKey(1)
            #     if ch == 27 or ch == ord('q') or ch == ord('Q'):
            #         break
            #     frame=model.show_result(
            #         img, result, score_thr=args.score_thr, wait_time=1, show=False,thickness=1,font_scale=1)
            #     cv2.imshow('frame',frame)
            #     if len(frame)>=1 or frame:
            #         #写入视频
            #         videoWriter.write(frame)
            #         count+=1
            #         print('Write {} in result Successfully!'.format(count))
            # else:
            #     count+=1
            """
        else:
            print('fail！！')
            break
        pbar.set_description("Processing video %s, frame : %d" % (video_in.replace(args.video_in_dir, ''), i))
    cap.release()
    videoWriter.release()
 
def main():
    args = parse_args()
    
    device = torch.device(args.device)
 
    model = init_detector(args.config, args.checkpoint, device=device)
    
    input_videos = list_files(args.video_in_dir, '.mp4')
    print(input_videos)
    for video in input_videos:        
        video_out = video.replace(args.video_in_dir, args.video_out_dir)
        dir_name = osp.abspath(osp.dirname(video_out))
        mkdir_or_exist(dir_name)
        detectvideo(model, video, video_out, args)
    
 
if __name__ == '__main__':
    main()