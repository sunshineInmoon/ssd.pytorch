# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import coco as cfg
from ..box_utils import match, log_sum_exp

# criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
#                             False, args.cuda)
class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive(过多) number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu #use_gpu=True 是否用GPU
        self.num_classes = num_classes #num_classes=21 类别数量
        self.threshold = overlap_thresh #overlap_thresh=0.5 IOU阈值
        self.background_label = bkg_label #背景label=0
        self.encode_target = encode_target #encode_target=False
        self.use_prior_for_matching = prior_for_matching #True
        self.do_neg_mining = neg_mining #难样本挖掘 True
        self.negpos_ratio = neg_pos #negative ：positive = 3：1
        self.neg_overlap = neg_overlap # 0.5
        self.variance = cfg['variance'] #'variance': [0.1, 0.2]

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        # loc_data Shape: [batch,num_priors,4l]
        loc_data, conf_data, priors = predictions 
        num = loc_data.size(0) # batch_size N
        priors = priors[:loc_data.size(1), :] #priors的维度8732*4
        num_priors = (priors.size(0)) # 8732
        num_classes = self.num_classes #21

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4) #loc_target 是个差值，是ground和default boxes的差值，我们拟合这个差值
        conf_t = torch.LongTensor(num, num_priors) #conf_target 为每个default分配一个label
        for idx in range(num): #遍历N张图片，但是一张图片里可能有多个目标，所以下面targets索引时才是下面那个样子
            truths = targets[idx][:, :-1].data #获取ground truth boxes坐标 
            labels = targets[idx][:, -1].data #获取对应label
            defaults = priors.data #获取先验框
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        pos = conf_t > 0 #非背景Prior(default boxes)的数量即正样本, 将conf_t>0的位置标记为1
        num_pos = pos.sum(dim=1, keepdim=True) #正样本的数量

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        # pos Shape: [batch,num_priors]
        # pos.unsqueeze(pos.dim()) Shape: [batch,num_priors,1]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4) #之所以将pos维度转成和loc_data相同，为了将其做成掩码
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        # conf_data Shape: [batch,num_priors,num_classes]
        # batch_conf Shape: [batch*num_priors, num_classes]
        batch_conf = conf_data.view(-1, self.num_classes)
        # conf_t 为每一个Prior分配了一个label Shape: [batch, num_priors]
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1)) #????

        # Hard Negative Mining #难负样本挖掘
        loss_c[pos] = 0  # filter out pos boxes for now 正样本不考虑
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c
