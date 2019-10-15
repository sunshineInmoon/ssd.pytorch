from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch


class PriorBox(object):
    """Compute priorbox coordinates in center-offset(中心偏离) form for each source
    feature map.
    """
    # 以cfg = voc 为例
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim'] #‘min_dim’: 300
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios']) # 6
        self.variance = cfg['variance'] or [0.1] #'variance': [0.1, 0.2]
        self.feature_maps = cfg['feature_maps'] #'feature_maps': [38, 19, 10, 5, 3, 1]
        self.min_sizes = cfg['min_sizes'] #'min_sizes': [30, 60, 111, 162, 213, 264]
        self.max_sizes = cfg['max_sizes'] #'max_sizes': [60, 111, 162, 213, 264, 315]
        self.steps = cfg['steps'] #'steps': [8, 16, 32, 64, 100, 300]
        self.aspect_ratios = cfg['aspect_ratios'] #纵横比 'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.clip = cfg['clip'] #'clip': True
        self.version = cfg['name'] # VOC
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps): #'feature_maps': [38, 19, 10, 5, 3, 1] 论文中特征图的大小
            for i, j in product(range(f), repeat=2): # A_n_2 组合，获取每个像素的坐标
                f_k = self.image_size / self.steps[k]
                # unit center x,y 对像素坐标进行归一化，
                # 加上0.5，cx，cy是小数了
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k]/self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]
        # back to torch land
        # 对于不同分辨率的feature map我们设计不同大小的先验框，同时我们需要注意第二个循环，是对feature map上每个像素点的循环
        # 因此总的先验框数量为4*38*38+6*19*19+6*10*10+6*5*5+4*3*3+4*1*1=8732
        output = torch.Tensor(mean).view(-1, 4) 
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
