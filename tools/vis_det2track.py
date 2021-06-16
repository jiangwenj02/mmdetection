"""
baseline for 1st Anti-UAV
https://anti-uav.github.io/
Qiang Wang
2020.02.16
"""
from __future__ import absolute_import
import os
import glob
import json
import cv2
import numpy as np
from tqdm import tqdm
from moviepy.editor import ImageSequenceClip

def iou(bbox1, bbox2):
    """
    Calculates the intersection-over-union of two bounding boxes.
    Args:
        bbox1 (numpy.array, list of floats): bounding box in format x,y,w,h.
        bbox2 (numpy.array, list of floats): bounding box in format x,y,w,h.
    Returns:
        int: intersection-over-onion of bbox1, bbox2
    """
    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, w1_1, h1_1) = bbox1
    (x0_2, y0_2, w1_2, h1_2) = bbox2
    x1_1 = x0_1 + w1_1
    x1_2 = x0_2 + w1_2
    y1_1 = y0_1 + h1_1
    y1_2 = y0_2 + h1_2
    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union


def not_exist(pred):
    return (len(pred) == 1 and pred[0] == 0) or len(pred) == 0


def eval(out_res, label_res):
    measure_per_frame = []
    for _pred, _gt, _exist in zip(out_res, label_res['gt_rect'], label_res['exist']):
        measure_per_frame.append(not_exist(_pred) if not _exist else iou(_pred, _gt) if len(_pred) > 1 else 0)
    return np.mean(measure_per_frame)


def main(mode='IR', visulization=False):
    assert mode in ['IR', 'RGB'], 'Only Support IR or RGB to evalute'

    # setup experiments
    video_paths = glob.glob(os.path.join('/data3/publicData/Anti_UAV_new_test/', '*'))
    save_path = 'work_dirs/faster_rcnn_r50_fpn_1x_uav/results_video/'
    os.popen('rm -r ' + save_path + '*')
    video_num = len(video_paths)
    overall_performance = []

    # run tracking experiments and report performance
    for video_id, video_path in tqdm(enumerate(video_paths, start=1)):
        video_name = os.path.basename(video_path)
        video_file = os.path.join(video_path, '%s.mp4'%mode)
        label_file = os.path.join(video_path, '%s_label.json'%mode)
        with open(label_file, 'r') as f:
            label_res = json.load(f)

        res_file = os.path.join('work_dirs/faster_rcnn_r50_fpn_1x_uav/results/', '%s_IR.txt'%video_name)
        save_file = os.path.join(save_path, '%s_IR.mp4'%video_name)
        with open(res_file, 'r') as f:
            res = json.load(f)
            res = res['res']

        capture = cv2.VideoCapture(video_file)
        frame_sum = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_id = 0
        frame_list = []
        while True:
            ret, frame = capture.read()
            if not ret:
                capture.release()
                break
            out = res[frame_id]
            if visulization:
                _gt = label_res['gt_rect'][frame_id]
                _exist = label_res['exist'][frame_id]
                if _exist:
                    cv2.rectangle(frame, (int(_gt[0]), int(_gt[1])), (int(_gt[0] + _gt[2]), int(_gt[1] + _gt[3])),
                                  (0, 255, 0))
                cv2.putText(frame, 'exist' if _exist else 'not exist',
                            (frame.shape[1] // 2 - 20, 30), 1, 2, (0, 255, 0) if _exist else (0, 0, 255), 2)
                if len(out) > 0:
                    cv2.putText(frame, '%.3f'%(out[-1]),
                                (frame.shape[1] // 2 + 40, 30), 1, 2, (0, 255, 255), 2)
                    cv2.rectangle(frame, (int(out[0]), int(out[1])), (int(out[0] + out[2]), int(out[1] + out[3])),
                                (0, 255, 255))
                else:
                    cv2.putText(frame, 'not exist',
                            (frame.shape[1] // 2 + 40, 30), 1, 2, (0, 255, 255), 2)
                # cv2.imshow(video_name, frame)
                # cv2.waitKey(1)
                frame_list.append(frame)
            frame_id += 1
        if visulization:
            cv2.destroyAllWindows()

        visual_clip = ImageSequenceClip(frame_list, fps=fps) #put frames together using moviepy
        visual_clip.write_videofile(save_file, threads=8, logger=None) #export the video
        mixed_measure = eval(res, label_res)
        overall_performance.append(mixed_measure)
        print('[%03d/%03d] %20s %5s Fixed Measure: %.03f' % (video_id, video_num, video_name, mode, mixed_measure))

    print('[Overall] %5s Mixed Measure: %.03f\n' % (mode, np.mean(overall_performance)))


if __name__ == '__main__':
    main(mode='IR', visulization=True)
