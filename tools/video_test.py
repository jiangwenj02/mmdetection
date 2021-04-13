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
from mmdet.apis import inference_detector, init_detector
 
 
def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--file', type=str, default='',help='test video path')
    parser.add_argument(
    '--out', type=str, help='output video path')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()
    return args
 
 
def main():
    args = parse_args()
 
    if not args.file:
        print("no input test file")
        exit(0)
    
    device = torch.device(args.device)
 
    model = init_detector(args.config, args.checkpoint, device=device)
 
    cap = cv2.VideoCapture(args.file)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #获取视频帧率
    #设置写入视频的编码格式
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    ####重要
    videoWriter = cv2.VideoWriter(args.out, fourcc, fps_video, (frame_width, frame_height))
    count=0
    print('Press "Esc", "q" or "Q" to exit.')
    while True:
        torch.cuda.empty_cache()
        ret_val, img = cap.read()
        if ret_val:
            if count<0:
                count+=1 
                print('Write {} in result Successfully!'.format(count))
                continue
            #############################
            result = inference_detector(model, img)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord('q') or ch == ord('Q'):
                break
            frame=model.show_result(
                img, result, score_thr=args.score_thr, wait_time=1, show=False,thickness=1,font_scale=1)
            cv2.imshow('frame',frame)
            if len(frame)>=1 or frame:
                #写入视频
                videoWriter.write(frame)
                count+=1
                print('Write {} in result Successfully!'.format(count))
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
    cap.release()
    videoWriter.release()
    cv2.destroyAllWindows()
 
if __name__ == '__main__':
    main()