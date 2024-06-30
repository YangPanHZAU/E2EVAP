import torch
from torch import nn


def sigmoid(x):
    y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
    return y

def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
            pred (batch x c x h x w)
            gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)

class DMLoss(nn.Module):
    def __init__(self, type='l1'):
        type_list = {'l1': torch.nn.functional.l1_loss, 'smooth_l1': torch.nn.functional.smooth_l1_loss}
        self.crit = type_list[type]
        super(DMLoss, self).__init__()

    def interpolation(self, poly, time=10):
        ori_points_num = poly.size(1)
        poly_roll =torch.roll(poly, shifts=1, dims=1)
        poly_ = poly.unsqueeze(3).repeat(1, 1, 1, time)
        poly_roll = poly_roll.unsqueeze(3).repeat(1, 1, 1, time)
        step = torch.arange(0, time, dtype=torch.float32).cuda() / time
        poly_interpolation = poly_ * step + poly_roll * (1. - step)
        poly_interpolation = poly_interpolation.permute(0, 1, 3, 2).reshape(poly_interpolation.size(0), ori_points_num * time, 2)
        return poly_interpolation

    def compute_distance(self, pred_poly, gt_poly):
        pred_poly_expand = pred_poly.unsqueeze(1)
        gt_poly_expand = gt_poly.unsqueeze(2)
        gt_poly_expand = gt_poly_expand.expand(gt_poly_expand.size(0), gt_poly_expand.size(1),
                                               pred_poly_expand.size(2), gt_poly_expand.size(3))
        pred_poly_expand = pred_poly_expand.expand(pred_poly_expand.size(0), gt_poly_expand.size(1),
                                                   pred_poly_expand.size(2), pred_poly_expand.size(3))
        distance = torch.sum((pred_poly_expand - gt_poly_expand) ** 2, dim=3)
        return distance
    
    def lossPred2NearestGt(self, ini_pred_poly, pred_poly, gt_poly):
        gt_poly_interpolation = self.interpolation(gt_poly)
        distance_pred_gtInterpolation = self.compute_distance(ini_pred_poly, gt_poly_interpolation)
        index_gt = torch.min(distance_pred_gtInterpolation, dim=1)[1]
        index_0 = torch.arange(index_gt.size(0))
        index_0 = index_0.unsqueeze(1).expand(index_gt.size(0), index_gt.size(1))
        loss_predto_nearestgt = self.crit(pred_poly,gt_poly_interpolation[index_0, index_gt, :])
        return loss_predto_nearestgt

    def lossGt2NearestPred(self, ini_pred_poly, pred_poly, gt_poly):
        distance_pred_gt = self.compute_distance(ini_pred_poly, gt_poly)
        index_pred = torch.min(distance_pred_gt, dim=2)[1]
        index_0 = torch.arange(index_pred.size(0))
        index_0 = index_0.unsqueeze(1).expand(index_pred.size(0), index_pred.size(1))
        loss_gtto_nearestpred = self.crit(pred_poly[index_0, index_pred, :], gt_poly,reduction='none')
        return loss_gtto_nearestpred

    def setloss(self, ini_pred_poly, pred_poly, gt_poly, keyPointsMask):
        keyPointsMask = keyPointsMask.unsqueeze(2).expand(keyPointsMask.size(0), keyPointsMask.size(1), 2)
        lossPred2NearestGt = self.lossPred2NearestGt(ini_pred_poly, pred_poly, gt_poly)
        lossGt2NearestPred = self.lossGt2NearestPred(ini_pred_poly, pred_poly, gt_poly)

        loss_set2set = torch.sum(lossGt2NearestPred * keyPointsMask) / (torch.sum(keyPointsMask) + 1) + lossPred2NearestGt
        return loss_set2set / 2.

    def forward(self, ini_pred_poly, pred_polys_, gt_polys, keyPointsMask):
        return self.setloss(ini_pred_poly, pred_polys_, gt_polys, keyPointsMask)
    
import torch
import torch.nn as nn
def dice_loss(pred,
              target,
              eps=1e-3,
              naive_dice=False):
    """Calculate dice loss, there are two forms of dice loss is supported:

        - the one proposed in `V-Net: Fully Convolutional Neural
            Networks for Volumetric Medical Image Segmentation
            <https://arxiv.org/abs/1606.04797>`_.
        - the dice loss in which the power of the number in the
            denominator is the first power instead of the second
            power.

    Args:
        pred (torch.Tensor): The prediction, has a shape (n, *)
        target (torch.Tensor): The learning label of the prediction,
            shape (n, *), same shape of pred.
        weight (torch.Tensor, optional): The weight of loss for each
            prediction, has a shape (n,). Defaults to None.
        eps (float): Avoid dividing by zero. Default: 1e-3.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        naive_dice (bool, optional): If false, use the dice
                loss defined in the V-Net paper, otherwise, use the
                naive dice loss in which the power of the number in the
                denominator is the first power instead of the second
                power.Defaults to False.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """

    input = pred.flatten(1).cuda()
    target = target.flatten(1).float().cuda()

    a = torch.sum(input * target, 1)
    if naive_dice:
        b = torch.sum(input, 1)
        c = torch.sum(target, 1)
        d = (2 * a + eps) / (b + c + eps)
    else:
        b = torch.sum(input * input, 1) + eps
        c = torch.sum(target * target, 1) + eps
        d = (2 * a) / (b + c)

    loss = 1 - d
    return loss

def topology_loss(pred,
                  target,
                  weight=[0.5,0.5],
                  eps=1e-5,
                  naive_dice=False):
    #pred,[bs,max_ct_num,h,w] target,[bs,1,h,w]
    bs,max_ct_num,h,w=pred.shape[:]
    target_resize = nn.functional.interpolate(target, (h,w), mode='nearest')
    pos_pred_sum = torch.sum(pred,dim=1)#[b,h,w]
    pos_pred_max = torch.max(pred,dim=1)[0]#[b,h,w]
    topo_score = (pos_pred_max.flatten(1)+eps)/(pos_pred_sum.flatten(1)+eps)
    topo_loss = 1 - topo_score
    loss_dice = dice_loss(torch.unsqueeze(pos_pred_max,1),target_resize)
    loss = topo_loss.mean()*weight[0]+loss_dice.mean()*weight[1]
    return loss

class TopologyBceLoss(nn.Module):

    def __init__(self,use_sigmoid=True):
        super(TopologyBceLoss, self).__init__()
        self.use_sigmoid=use_sigmoid
        self.loss = nn.BCELoss()

    def forward(self,
                pred,
                target):
        if self.use_sigmoid:
            pred=pred.sigmoid()
        bs,c,h,w=pred.shape[:]
        pos_pred_max = torch.max(pred,dim=1)[0]#[b,h,w]
        target_resize = nn.functional.interpolate(target, (h,w), mode='nearest')
        loss = self.loss(torch.unsqueeze(pos_pred_max,1).float(),target_resize.float())

        return loss

class TopologyDiceLoss(nn.Module):

    def __init__(self,use_sigmoid=True,eps=1e-3):
        super(TopologyDiceLoss, self).__init__()
        self.use_sigmoid=use_sigmoid
        self.eps = eps

    def forward(self,
                pred,
                target):
        if self.use_sigmoid:
            pred=pred.sigmoid()
        bs,c,h,w=pred.shape[:]
        pos_pred_max = torch.max(pred,dim=1)[0]#[b,h,w]
        target_resize = nn.functional.interpolate(target, (h,w), mode='nearest')
        loss = dice_loss(torch.unsqueeze(pos_pred_max,1).float(),target_resize.float(),eps=self.eps).mean()
        return loss

class TopologyBceDiceLoss(nn.Module):

    def __init__(self,use_sigmoid=True,eps=1e-3,weight=[0.5,0.5]):
        super(TopologyBceDiceLoss, self).__init__()
        self.use_sigmoid=use_sigmoid
        self.eps = eps
        self.dice_loss=TopologyDiceLoss(use_sigmoid=self.use_sigmoid,eps=self.eps)
        self.bce_loss=TopologyBceLoss(use_sigmoid=self.use_sigmoid)
        self.weight=weight

    def forward(self,
                pred,
                target):
        loss=self.weight[0]*self.dice_loss(pred,target)+self.weight[1]*self.bce_loss(pred,target)
        return loss

class TopologyLoss(nn.Module):

    def __init__(self,
                 loss_weight=1.0,
                 loss_topo_dice_weoght=[0.5,0.5],
                 eps=1e-3):
        super(TopologyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.loss_topo_dice_weoght = loss_topo_dice_weoght
        self.eps = eps

    def forward(self,
                pred,
                target):
        loss = self.loss_weight * topology_loss(
            pred,
            target,
            self.loss_topo_dice_weoght,
            eps=self.eps)

        return loss
class DiceLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 naive_dice=False,
                 eps=1e-3):
        super(DiceLoss, self).__init__()
        self.use_sigmoid=use_sigmoid
        self.eps = eps
        self.naive_dice =naive_dice
    def forward(self,
            pred,
            target):
        if self.use_sigmoid:
            pred = pred.sigmoid()
        bs,c,h,w=pred.shape[:]
        if (h,w) !=(target.size(2),target.size(3)):
            target = nn.functional.interpolate(target.float(), (h,w), mode='bilinear')
        loss =  dice_loss(
            pred,
            target,
            eps=self.eps).mean()
        return loss