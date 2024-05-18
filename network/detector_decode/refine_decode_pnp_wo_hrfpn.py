import sys
sys.path.append("./")
import torch
from network.detector_decode.utils import decode_ct_hm, clip_to_image, get_gcn_feature
from torch import nn
from network.detector_decode.PnP import pnp
BN_MOMENTUM = 0.1
class PnP_contour_feature(nn.Module):
    def __init__(self, c_in=64,c_out=64, num_point=128):
        super(PnP_contour_feature, self).__init__()
        self.num_point = num_point
        self.relu=nn.ReLU(inplace=True)
    def forward_train(self,contour,cnn_feature,ct_num):#contour [112,128,2],cnn_feature[bs,c_in,h,w],ct_num [num1,num2,...,num8]
        if len(contour) == 0:
            mask_feat=torch.zeros(contour.size(0),1,cnn_feature.size(2),cnn_feature.size(3))
            return mask_feat,cnn_feature
        bs,c_in,h,w=cnn_feature.shape[:]
        #contour = clip_to_image(contour, h,w)
        mask=pnp(contour,h,w)#[poly_num,h,w]
        max_channel=torch.max(ct_num)
        mask_batch=torch.zeros((bs,max_channel,h,w)).to(cnn_feature.device)#[bs,max(len),128,128]
        mask_batch[0,:ct_num[0],:,:]=mask[:ct_num[0],:,:]
        #mask_batch[bs-1,:ct_num[-1],:,:]=mask[-ct_num[-1]:,:,:]
        for i in range(1,bs):
            mask_batch[i,:ct_num[i],:,:]=mask[ct_num[i-1]:ct_num[i-1]+ct_num[i],:,:]
        max_mask_feat=torch.max(mask_batch,dim=1,keepdim=True)[0]
        cnn_feature=self.relu(max_mask_feat*cnn_feature+cnn_feature)
        return mask_batch,cnn_feature
    def forward_test(self,contour,cnn_feature):#contour [112,128,2],cnn_feature[bs,c_in,h,w],ct_num max box num /patch
        if len(contour) == 0:
            mask_feat=torch.zeros(contour.size(0),1,cnn_feature.size(2),cnn_feature.size(3))
            return mask_feat,cnn_feature
        bs,c_in,h,w=cnn_feature.shape[:]
        contour = clip_to_image(contour, h,w)
        mask=pnp(contour,h,w)#[poly_num,h,w]
        poly_num=contour.shape[0]
        mask_batch=torch.unsqueeze(mask,dim=0)#[1,poly_num,h,w]
        max_mask_feat=torch.max(mask_batch,dim=1,keepdim=True)[0]
        cnn_feature=self.relu(max_mask_feat*cnn_feature+cnn_feature)
        return mask_batch,cnn_feature
    def forward(self,contour,cnn_feature,ct_num=None):
        if not (ct_num is None):
            mask_batch,cnn_feature_pnp=self.forward_train(contour,cnn_feature,ct_num)
        else:
            mask_batch,cnn_feature_pnp=self.forward_test(contour,cnn_feature)
        return mask_batch,cnn_feature_pnp
#mask_batch [bs,max(ct_num),h/4,w/4]
#cnn_feature [bs,c_out,h/2,w/2]
class Refine(torch.nn.Module):
    def __init__(self, c_in=64, num_point=128, stride=4.):
        super(Refine, self).__init__()
        self.num_point = num_point
        self.stride = stride
        self.trans_feature = torch.nn.Sequential(torch.nn.Conv2d(c_in, 256, kernel_size=3,
                                                                 padding=1, bias=True),
                                                 torch.nn.ReLU(inplace=True),
                                                 torch.nn.Conv2d(256, 64, kernel_size=1,
                                                                 stride=1, padding=0, bias=True))
        self.trans_poly = torch.nn.Linear(in_features=((num_point + 1) * 64),
                                          out_features=num_point * 4, bias=False)
        self.trans_fuse = torch.nn.Linear(in_features=num_point * 4,
                                          out_features=num_point * 2, bias=True)

    def global_deform(self, points_features, init_polys):
        poly_num = init_polys.size(0)
        points_features = self.trans_poly(points_features)#[poly_num,64*129]->[poly_num,128*4]
        offsets = self.trans_fuse(points_features).view(poly_num, self.num_point, 2)#[poly_num,128,2]
        coarse_polys = offsets * self.stride + init_polys.detach()
        return coarse_polys

    def forward(self, feature, ct_polys, init_polys, ct_img_idx, ignore=False):
        if ignore or len(init_polys) == 0:
            return init_polys
        h, w = feature.size(2), feature.size(3)#128,128
        poly_num = ct_polys.size(0)
    
        feature = self.trans_feature(feature)
#feature [8,64,128,128],ct_polys [113,2]->[113,1,2],init_polys [113,128,2],points[113,129,2]
        ct_polys = ct_polys.unsqueeze(1).expand(init_polys.size(0), 1, init_polys.size(2))
        points = torch.cat([ct_polys, init_polys], dim=1)
        feature_points = get_gcn_feature(feature, points, ct_img_idx, h, w).view(poly_num, -1)
        coarse_polys = self.global_deform(feature_points, init_polys)
        return coarse_polys
class Refine_wtih_pnp(nn.Module):
    def __init__(self, c_in=64, num_point=128, stride=2.):
        super(Refine_wtih_pnp, self).__init__()
        self.num_point = num_point
        self.stride = stride
        self.trans_feature = torch.nn.Sequential(torch.nn.Conv2d(c_in, 256, kernel_size=3,
                                                                 padding=1, bias=True),
                                                 torch.nn.ReLU(inplace=True),
                                                 torch.nn.Conv2d(256, 64, kernel_size=1,
                                                                 stride=1, padding=0, bias=True))
        self.trans_poly = torch.nn.Linear(in_features=((num_point + 1) * 64),
                                          out_features=num_point * 4, bias=False)
        self.trans_fuse = torch.nn.Linear(in_features=num_point * 4,
                                          out_features=num_point * 2, bias=True)
        self.PnP_contour_feature = PnP_contour_feature(c_in,c_out=c_in,num_point=num_point)
    def global_deform(self, points_features, init_polys):
        poly_num = init_polys.size(0)
        points_features = self.trans_poly(points_features)#[poly_num,128*129]->[poly_num,128*4]
        offsets = self.trans_fuse(points_features).view(poly_num, self.num_point, 2)#[poly_num,128,2]
        coarse_polys = offsets * self.stride + (init_polys).detach()#[0,64]->scale [0,128]
        return coarse_polys

    def forward(self, feature, ct_polys, init_polys, ct_img_idx,ct_num, ignore=False):
        if ignore or len(init_polys) == 0:
            mask_feat=torch.zeros(init_polys.size(0),1,feature.size(2),feature.size(3))
            return init_polys,mask_feat,feature
        h, w = feature.size(2), feature.size(3)#128,128,[bs,64,128,128]
        poly_num = ct_polys.size(0)
        feature = self.trans_feature(feature)#[bs,64,128,128]\
        mask_batch,feature=self.PnP_contour_feature(init_polys,feature,ct_num)
        #bs,max(ct_num),h/4,w/4,feature [bs,c_out,128,128]
#feature [8,64,128,128],ct_polys [113,2]->[113,1,2],init_polys [113,128,2],points[113,129,2]
        ct_polys = ct_polys.unsqueeze(1).expand(init_polys.size(0), 1, init_polys.size(2))
        points = torch.cat([ct_polys, init_polys], dim=1)
        #up_h, up_w = up_feature.size(2), up_feature.size(3)#256,256
        feature_points = get_gcn_feature(feature, points, ct_img_idx, h, w).view(poly_num, -1)##[all_obj_num,64,129]
        coarse_polys = self.global_deform(feature_points, init_polys)
        return coarse_polys,mask_batch,feature
class Decode(torch.nn.Module):
    def __init__(self, c_in=64, num_point=128, init_stride=10., coarse_stride=4., down_sample=4., min_ct_score=0.05):
        super(Decode, self).__init__()
        self.stride = init_stride
        self.down_sample = down_sample
        self.min_ct_score = min_ct_score
        self.refine=Refine_wtih_pnp(c_in,num_point,coarse_stride)
        self.coarse_pnp=PnP_contour_feature(c_in=c_in, c_out=c_in,num_point=num_point)

    def train_decode(self, data_input, output, cnn_feature):
        meta = data_input['meta']
        wh_pred = output['wh']#[8,256,128,128]
        ct_01 = data_input['ct_01'].bool()#[8,23]
        ct_ind = data_input['ct_ind'][ct_01]#[113]
        ct_img_idx = data_input['ct_img_idx'][ct_01]#[113] batch every object,all number is 113
        _, _, height, width = data_input['ct_hm'].size()#128,128
        ct_x, ct_y = ct_ind % width, ct_ind // width#[113],[113]
#line 54 is the solve ct_x,and tc_y
        if ct_x.size(0) == 0:
            ct_offset = wh_pred[ct_img_idx, :, ct_y, ct_x].view(ct_x.size(0), 1, 2)
        else:
            ct_offset = wh_pred[ct_img_idx, :, ct_y, ct_x].view(ct_x.size(0), -1, 2)
#ct_offset [113, 128, 2],every object the xy offset compare center
        ct_x, ct_y = ct_x[:, None].to(torch.float32), ct_y[:, None].to(torch.float32)
        ct = torch.cat([ct_x, ct_y], dim=1)
#ct [113,2]
        init_polys = ct_offset * self.stride + ct.unsqueeze(1).expand(ct_offset.size(0),
                                                                      ct_offset.size(1), ct_offset.size(2))
        coarse_polys,mask_batch_init,feature = self.refine(cnn_feature, ct, init_polys, ct_img_idx.clone(),meta['ct_num'])
        mask_batch_coarse,feature=self.coarse_pnp(coarse_polys,feature,meta['ct_num'])#contour,cnn_feature,ct_num=None
#init polys is [113,128,2],coarse_polys [poly_num,128,2],up_feature1 [bs,64,128,128]
        #mask_batch_coarse,multi_scale_deform_feat=self.decoder_multi_scale_deform(cnn_feature,up_feature1,coarse_polys,meta['ct_num'])
        output.update({'poly_init': init_polys * self.down_sample})#[poly_num,128,2]
        output.update({'poly_coarse': coarse_polys * self.down_sample})#[poly_num,128,2]
        output.update({'mask_batch_init': mask_batch_init})#[bs,max(ct_num),128,128]
        output.update({'mask_batch_coarse': mask_batch_coarse})#[bs,max(ct_num),128,128]
        return feature#[bs,64,128,128]

    def test_decode(self, cnn_feature, output, K=100, min_ct_score=0.05, ignore_gloabal_deform=False):
        hm_pred, wh_pred = output['ct_hm'], output['wh']
        #ct_hm [bs,class_num,128,128] wh [bs,256,128,128]
        poly_init, detection = decode_ct_hm(torch.sigmoid(hm_pred), wh_pred,
                                            K=K, stride=self.stride)
        #poly_init [bs,100,128,2] detection [ct, scores, clses] [bs,100,4],[bs,100,2],[bs,100,1],[bs,100,1]
        valid = detection[0, :, 2] >= min_ct_score
        poly_init, detection = poly_init[0][valid], detection[0][valid]

        init_polys = clip_to_image(poly_init, cnn_feature.size(2), cnn_feature.size(3))
        output.update({'poly_init': init_polys * self.down_sample})#init_polys [100,128,2]
        img_id = torch.zeros((len(poly_init), ), dtype=torch.int64)
        #poly_coarse = self.refine(cnn_feature, detection[:, :2], poly_init, img_id, ignore=ignore_gloabal_deform)
        coarse_polys,mask_batch_init,feature = self.refine(cnn_feature, detection[:, :2], poly_init, img_id,ct_num=None, ignore=ignore_gloabal_deform)
        mask_batch_coarse,feature=self.coarse_pnp(coarse_polys,feature,ct_num=None)
        coarse_polys = clip_to_image(coarse_polys, cnn_feature.size(2), cnn_feature.size(3))
        output.update({'poly_coarse': coarse_polys * (self.down_sample)})
        output.update({'detection': detection})
        #output.update({'multi_scale_deform_feat': multi_scale_deform_feat})
        return feature

    def forward(self, data_input, cnn_feature, output=None, is_training=True, ignore_gloabal_deform=False):
        if is_training:
            feature=self.train_decode(data_input, output, cnn_feature)
        else:
            feature=self.test_decode(cnn_feature, output, min_ct_score=self.min_ct_score,
                             ignore_gloabal_deform=ignore_gloabal_deform)
        return feature
# import torch
# a=torch.randn(1,64,64,64)
# # b=torch.max(a,dim=1,keepdim=True)[0]
# # print(b.shape)
# model=DeformConv_up(64,128)
# c=model(a)
# print(c.shape)