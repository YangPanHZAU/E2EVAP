import torch.nn as nn
import torch

def nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = gather_feat(feat, ind)
    return feat

def topk(scores, K=100):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)
    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def decode_ct_hm(ct_hm, wh, reg=None, K=100, stride=10.):
    batch, cat, height, width = ct_hm.size()
    ct_hm = nms(ct_hm)
    scores, inds, clses, ys, xs = topk(ct_hm, K=K)#[bs,100],[bs,100],[bs,100],[bs,100],[bs,100]
    wh = transpose_and_gather_feat(wh, inds)
    wh = wh.view(batch, K, -1, 2)

    if reg is not None:
        reg = transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1)
        ys = ys.view(batch, K, 1)

    clses = clses.view(batch, K, 1).float()#[bs,100,1]
    scores = scores.view(batch, K, 1)#[bs,100,1]
    ct = torch.cat([xs, ys], dim=2)#[bs,100,2]
    poly = ct.unsqueeze(2).expand(batch, K, wh.size(2), 2) + wh * stride#[bs,100,128,2]
    detection = torch.cat([ct, scores, clses], dim=2)#[bs,100,4]
    return poly, detection

def clip_to_image(poly, h, w):
    poly[..., :2] = torch.clamp(poly[..., :2], min=0)
    poly[..., 0] = torch.clamp(poly[..., 0], max=w-1)
    poly[..., 1] = torch.clamp(poly[..., 1], max=h-1)
    return poly

def get_gcn_feature(cnn_feature, img_poly, ind, h, w):
    img_poly = img_poly.clone()#[113,129,2]
    img_poly[..., 0] = img_poly[..., 0] / (w / 2.) - 1 #why?,x/64 -1,y/64 -1
    img_poly[..., 1] = img_poly[..., 1] / (h / 2.) - 1
    batch_size = cnn_feature.size(0)
    gcn_feature = torch.zeros([img_poly.size(0), cnn_feature.size(1), img_poly.size(1)]).to(img_poly.device)
    for i in range(batch_size):#gcn_feature [113, 64, 129]
        poly = img_poly[ind == i].unsqueeze(0)#[1,8,129,2]# batch,object num,ct+xy,xy
        feature = torch.nn.functional.grid_sample(cnn_feature[i:i+1], poly)[0].permute(1, 0, 2)#
        gcn_feature[ind == i] = feature
    return gcn_feature#[all_obj_num,64,129]
#grid sample cnn_feature[i:i+1] [1,64,128,128],poly[1,8,129,2],->[64,8,129],->[8,64,129]
#每个目标的点对应的64个特征向量
# import torch
# hm_pred, wh_pred = torch.randn(4,1,128,128), torch.randn(4,256,128,128)
# poly_init, detection = decode_ct_hm(torch.sigmoid(hm_pred), wh_pred,
#                                             K=100, stride=4)
# valid = detection[0, :, 2] >= 0.01#[100]
# poly_init, detection = poly_init[0][valid], detection[0][valid]
# init_polys = clip_to_image(poly_init, 128, 128)
# output={}
# output.update({'poly_init': init_polys * 4})