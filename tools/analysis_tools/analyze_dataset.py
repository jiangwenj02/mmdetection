import copy
import os
from argparse import ArgumentParser
import xml.etree.ElementTree as ET
import numpy as np
from pycocotools.coco import COCO
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mmdet.datasets.builder import build_dataset
from tqdm import tqdm
import mmcv
from mmcv import Config, DictAction

def iou(box, clusters):
    """
    计算一个ground truth边界盒和k个先验框(Anchor)的交并比(IOU)值。
    参数box: 元组或者数据，代表ground truth的长宽。
    参数clusters: 形如(k,2)的numpy数组，其中k是聚类Anchor框的个数
    返回：ground truth和每个Anchor框的交并比。
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")
    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
    iou_ = intersection / (box_area + cluster_area - intersection)
    return iou_


def avg_iou(boxes, clusters):
    """
    计算一个ground truth和k个Anchor的交并比的均值。
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])

class kMean_parse:
    def __init__(self, ration_n_clusters, size_n_clusters, data):
        self.ration_n_clusters = ration_n_clusters
        self.size_n_clusters = size_n_clusters
        self.ratio_km = KMeans(n_clusters=self.ration_n_clusters,init="k-means++",n_init=10,max_iter=3000000,tol=1e-3,random_state=0)
        self.size_km = KMeans(n_clusters=self.size_n_clusters,init="k-means++",n_init=10,max_iter=3000000,tol=1e-3,random_state=0)
        self.data = data
 
 
    def parse_data (self):
        self.one_data = (self.data[:,1] / self.data[:,0]).reshape(-1, 1)
        # print(self.data.shape, self.one_data.shape)
        self.y_k = self.ratio_km.fit_predict(self.one_data)
        print('ratio: ', sorted(self.ratio_km.cluster_centers_))

        self.one_data = np.sqrt((self.data[:,1] ** 2 + self.data[:,0] ** 2).reshape(-1, 1))
        self.y_k = self.size_km.fit_predict(self.one_data)
        print('size: ', sorted(np.sqrt(self.size_km.cluster_centers_ ** 2 / 2)))
 
    def plot_data (self):
 
        cValue = ['orange','r','y','green','b','gray','black','purple','brown','tan']
 
        for i in range(self.n_clusters):
            plt.scatter(self.data[self.y_k == i, 0], self.data[self.y_k == i, 1], s=50, c=cValue[i%len(cValue)], marker="o",
                        label="cluster "+str(i))
 
       # draw the centers
        plt.scatter(self.km.cluster_centers_[:, 0], self.km.cluster_centers_[:, 1], s=250, marker="*", c="red", label="cluster center")
        plt.legend()
        plt.grid()
        plt.show()

def Iou_Kmeans(boxes, k, dist=np.median):
    """
    利用IOU值进行K-means聚类
    参数boxes: 形状为(r, 2)的ground truth框，其中r是ground truth的个数
    参数k: Anchor的个数
    参数dist: 距离函数
    返回值：形状为(k, 2)的k个Anchor框
    """
    # 即是上面提到的r
    rows = boxes.shape[0]
    # 距离数组，计算每个ground truth和k个Anchor的距离
    distances = np.empty((rows, k))
    # 上一次每个ground truth"距离"最近的Anchor索引
    last_clusters = np.zeros((rows,))
    # 设置随机数种子
    np.random.seed()

    # 初始化聚类中心，k个簇，从r个ground truth随机选k个
    clusters = boxes[np.random.choice(rows, k, replace=False)]
    # 开始聚类
    while True:
        # 计算每个ground truth和k个Anchor的距离，用1-IOU(box,anchor)来计算
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)
        # 对每个ground truth，选取距离最小的那个Anchor，并存下索引
        nearest_clusters = np.argmin(distances, axis=1)
        # 如果当前每个ground truth"距离"最近的Anchor索引和上一次一样，聚类结束
        if (last_clusters == nearest_clusters).all():
            break
        # 更新簇中心为簇里面所有的ground truth框的均值
        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)
        # 更新每个ground truth"距离"最近的Anchor索引
        last_clusters = nearest_clusters

    return clusters


def id2name(coco):
    classes = dict()
    classes_id = []
    for cls in coco.dataset['categories']:
        classes[cls['id']] = cls['name']

    for key in classes.keys():
        classes_id.append(key)
    return classes, classes_id

def retrieve_data_cfg(config_path, skip_type, cfg_options):
    cfg = Config.fromfile(config_path)
    if cfg_options is not None:
        cfg.merge_from_dict(cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    train_data_cfg = cfg.data.train
    if train_data_cfg.type == 'RepeatDataset':
        train_data_cfg = train_data_cfg.dataset
    train_data_cfg['pipeline'] = [
        x for x in train_data_cfg.pipeline if x['type'] not in skip_type
    ]

    return cfg

def load_dataset(cfg):

    bboxes = np.zeros((0,2))

    dataset = build_dataset(cfg.data.train)
    for i in tqdm(range(len(dataset))):
        item = dataset.__getitem__(i)
        gt_bboxes_wh = item['gt_bboxes'][:, 2:] - item['gt_bboxes'][:, :2] 
        bboxes = np.concatenate((bboxes, gt_bboxes_wh), axis=0)

    return bboxes

def main():
    parser = ArgumentParser(description='COCO Dataset Analysis Tool')
    parser.add_argument(
        '--config',
        default='configs/faster_rcnn/faster_rcnn_r50_caffe_c4_1x_coco.py',
        help='config file path')
    parser.add_argument(
        '--ratio_clusters',
        default=3,
        help='config file path')
    parser.add_argument(
        '--size_clusters',
        default=5,
        help='config file path')
    parser.add_argument(
        '--skip-type',
        type=str,
        nargs='+',
        default=['DefaultFormatBundle', 'Normalize', 'Collect'],
        help='skip some useless pipeline')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    cfg = retrieve_data_cfg(args.config, args.skip_type, args.cfg_options)
    data = load_dataset(cfg)

    kmean_parse = kMean_parse(args.ratio_clusters, args.size_clusters, data)
    kmean_parse.parse_data()
    # kmean_parse.plot_data()

    # print('ratio : ', out)
    # print('size', size)
    # anchor = np.array(out) * Inputdim
    # print("Boxes: {} ".format(anchor))
    # print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))

    # final_anchors = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
    # print("Before Sort Ratios:\n {}".format(final_anchors))
    # print("After Sort Ratios:\n {}".format(sorted(final_anchors)))


if __name__ == '__main__':
    main()
