import os
import numpy as np
import cv2

class Metric(object):
    def __init__(self, mode='center', iou_thresh=0.05, visualize=False, visualization_root='demo/',
                 image_classification=False):

        self.TPs = []
        self.FNs = []
        self.FPs = []
        self.TNs = []
        assert mode == 'center' or mode == 'iou', '({}) mode is not supported'
        self.mode = mode
        self.iou_thresh = iou_thresh
        self.image_classification = image_classification

        # in BGR order
        self.FP_color = (255, 0, 0)
        self.Detection_color = (0, 255, 0)
        self.GT_color = (0, 0, 255)
        self.visualize = visualize
        self.total_gt = 0.0
        if visualize:
            #  create image folder for saving detection result
            self.detection_folder = visualization_root + 'ALL/'
            self.false_positive_folder = visualization_root + 'FP/'
            self.false_negative_folder = visualization_root + 'FN/'
            os.makedirs(self.detection_folder, exist_ok=True)
            os.makedirs(self.false_positive_folder, exist_ok=True)
            os.makedirs(self.false_negative_folder, exist_ok=True)
            os.popen('rm -r ' + self.detection_folder + '*')
            os.popen('rm -r ' + self.false_positive_folder + '*')
            os.popen('rm -r ' + self.false_negative_folder + '*')

    def eval_add_result_image_classification(self, ground_truth: list, pred_points: list, image: np.ndarray = None,
                                             image_name=None):

        self.total_gt += 1
        hasTP = False
        for index_gt_box, gt_box in enumerate(ground_truth):
            gt = gt_box
            not_matched = []
            for j in pred_points:
                if self.mode == 'center':
                    ctx = j[0] + (j[2] - j[0]) * 0.5
                    cty = j[1] + (j[3] - j[1]) * 0.5
                    bbox_matched = gt[0] < ctx < gt[2] and gt[1] < cty < gt[3]

                elif self.mode == 'iou':
                    query_area = (j[2] - j[0]) * (j[3] - j[1])
                    gt_area = (gt[2] - gt[0]) * (gt[3] - gt[1])
                    iw = (min(j[2], gt[2]) - max(j[0], gt[0]))
                    ih = (min(j[3], gt[3]) - max(j[1], gt[1]))
                    iw = max(0, iw)
                    ih = max(0, ih)
                    ua = query_area + gt_area - (iw * ih)
                    overlaps = (iw * ih) / float(ua)
                    bbox_matched = overlaps > self.iou_thresh

                if bbox_matched:
                    if not hasTP:
                        hasTP = True
                else:
                    not_matched.append(j)
            pred_points = not_matched
        # 实际 1
        if len(ground_truth) > 0:
            # 预测 1
            if hasTP:
                self.TPs.append(1)
            # 预测 0
            else:
                self.FNs.append(1)
        # 实际 0
        else:
            # 预测 1
            if len(pred_points) > 1:
                self.FPs.append(1)
            # 预测 0
            else:
                self.TNs.append(1)

    def eval_add_result(self, ground_truth: list, pred_points: list, image: np.ndarray = None, image_name=None):
        if self.image_classification:
            return self.eval_add_result_image_classification(ground_truth, pred_points, image, image_name)

        if self.visualize:
            FPimage = image.copy()
            FNimage = image.copy()
            Detectionimage = image.copy()

            for pt in pred_points:
                pt1 = tuple([int(pt[0]), int(pt[1])])
                pt2 = tuple([int(pt[2]), int(pt[3])])
                cv2.rectangle(Detectionimage, pt1, pt2, self.Detection_color, 2)

        missing = False
        self.total_gt += len(ground_truth)
        for index_gt_box, gt_box in enumerate(ground_truth):
            hasTP = False
            gt = gt_box

            not_matched = []
            for j in pred_points:
                if self.mode == 'center':
                    ctx = j[0] + (j[2] - j[0]) * 0.5
                    cty = j[1] + (j[3] - j[1]) * 0.5
                    bbox_matched = gt[0] < ctx < gt[2] and gt[1] < cty < gt[3]

                elif self.mode == 'iou':
                    query_area = (j[2] - j[0]) * (j[3] - j[1])
                    gt_area = (gt[2] - gt[0]) * (gt[3] - gt[1])
                    iw = (min(j[2], gt[2]) - max(j[0], gt[0]))
                    ih = (min(j[3], gt[3]) - max(j[1], gt[1]))
                    iw = max(0, iw)
                    ih = max(0, ih)
                    ua = query_area + gt_area - (iw * ih)
                    overlaps = (iw * ih) / float(ua)
                    bbox_matched = overlaps > self.iou_thresh

                if bbox_matched:
                    if not hasTP:
                        self.TPs.append(j)
                        hasTP = True
                else:
                    not_matched.append(j)
            pred_points = not_matched

            if not hasTP:
                self.FNs.append(gt)

                if self.visualize:
                    # Draw False negative rect
                    missing = True
                    pt1 = tuple([int(gt[0]), int(gt[1])])
                    pt2 = tuple([int(gt[2]), int(gt[3])])
                    cv2.rectangle(FNimage, pt1, pt2, self.GT_color, 2)

            if self.visualize:
                # Draw groundturth on detection and FP images
                pt1 = tuple([int(gt[0]), int(gt[1])])
                pt2 = tuple([int(gt[2]), int(gt[3])])
                cv2.rectangle(Detectionimage, pt1, pt2, self.GT_color, 2)
                cv2.rectangle(FPimage, pt1, pt2, self.GT_color, 2)

        if self.visualize:
            if missing:
                cv2.imwrite(self.false_negative_folder + str(image_name) + '.jpg', FNimage)
            cv2.imwrite(self.detection_folder + str(image_name) + '.jpg', Detectionimage)
        if len(pred_points) > 0 and self.visualize:
            # Draw false positive rect
            for fp in pred_points:
                pt1 = tuple([int(fp[0]), int(fp[1])])
                pt2 = tuple([int(fp[2]), int(fp[3])])
                cv2.rectangle(FPimage, pt1, pt2, self.FP_color, 2)
            cv2.imwrite(self.false_positive_folder + str(image_name) + '.jpg', FPimage)
        #  add FP here
        self.FPs += pred_points

    def get_specificity_sensitivity(self):
        TP = len(self.TPs) * 1.0
        FP = len(self.FPs) * 1.0
        TN = len(self.TNs) * 1.0
        FN = len(self.FNs) * 1.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
        return specificity, sensitivity

    def get_precision_recall(self):
        TP = len(self.TPs) * 1.0
        FP = len(self.FPs) * 1.0
        FN = len(self.FNs) * 1.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        return precision, recall

    def get_result(self):
        if self.image_classification:
            return self.get_specificity_sensitivity()
        else:
            return self.get_precision_recall()

    def reset(self):
        self.TPs = []
        self.FNs = []
        self.FPs = []
        self.TNs = []
        self.total_gt = 0.0

