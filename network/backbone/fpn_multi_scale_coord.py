import torch
import torch.nn as nn
from mmcv.cnn import (ConvModule,normal_init)
import torch
from torch import nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
class MultiscaleCoordConvHead(nn.Module):
    def __init__(self,
                 in_channels=256,
                 feat_channels=256,
                 out_channels=256,
                 out_act_cfg=dict(type='ReLU'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg= dict(type='GN', num_groups=32, requires_grad=True),
                 conv_cfg=None,
                 fuse_by_cat=False,
                 feature_level_num=4,
                 **kwargs):
        super(MultiscaleCoordConvHead, self).__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.out_channels = out_channels
        self.out_act_cfg=out_act_cfg
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.fuse_by_cat = fuse_by_cat
        self.feature_level_num = feature_level_num
        if self.fuse_by_cat:
            pred_in_channels=(feat_channels+2)*self.feature_level_num
            self.conv_pred = ConvModule(
                pred_in_channels,
                self.out_channels,
                1,
                padding=0,
                conv_cfg=self.conv_cfg,
                act_cfg=self.out_act_cfg,
                norm_cfg=self.norm_cfg)  # 256->256 1*1 conv
        else:
            pred_in_channels=feat_channels+2
            self.conv_pred = ConvModule(
                pred_in_channels,
                self.out_channels,
                1,
                padding=0,
                conv_cfg=self.conv_cfg,
                act_cfg=self.out_act_cfg,
                norm_cfg=self.norm_cfg)  # 256->256 1*1 conv
        self.loc_convs=ConvModule(
                    self.out_channels,
                    64,
                    1,
                    norm_cfg=self.norm_cfg)
        self.seg_convs=ConvModule(
                    self.out_channels,
                    64,
                    1,
                    norm_cfg=self.norm_cfg)
        self._init_layers()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        for conv in [self.p2_conv,self.p3_conv,self.p4_conv,self.p5_conv,self.p6_conv]:
            for m in conv.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
        normal_init(self.init_kernels, mean=0, std=0.01)

    def Double_conv(self, kernel_size,in_channels=256):
        return nn.Sequential(
            ConvModule(in_channels,
                       self.feat_channels,
                       kernel_size,
                       1,
                       padding=int(kernel_size//2),
                       conv_cfg=self.conv_cfg,
                       norm_cfg=self.norm_cfg,
                       act_cfg=self.act_cfg,
                       inplace=False),
            ConvModule(self.feat_channels,
                       self.feat_channels,
                       kernel_size,
                       1,
                       padding=int(kernel_size // 2),
                       conv_cfg=self.conv_cfg,
                       norm_cfg=self.norm_cfg,
                       act_cfg=self.act_cfg,
                       inplace=False)
        )
    def _init_layers(self):
        self.coordconv_block_num=4
        #self.kernel_size_list=[7,7,3,3]
        self.kernel_size_list=[3,3,3,3]
        assert len(self.kernel_size_list)==self.coordconv_block_num
        p2_conv =[]
        p3_conv = []
        p4_conv = []
        p5_conv = []
        in_channels=self.in_channels
        for i in range(self.coordconv_block_num):
            p2_conv.append(self.Double_conv(self.kernel_size_list[i]))
        for i in range(self.coordconv_block_num):
            p3_conv.append(self.Double_conv(self.kernel_size_list[i]))
            if i == self.coordconv_block_num - 1:
                p3_conv.append(nn.Upsample(
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=False)
                )
        for i in range(self.coordconv_block_num):
            in_channels = self.in_channels
            if i >= 3:
                in_channels = self.feat_channels + 2
            p4_conv.append(self.Double_conv(self.kernel_size_list[i], in_channels))
            if i >= self.coordconv_block_num - 2:
                p4_conv.append(nn.Upsample(
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=False)
                )
        for i in range(self.coordconv_block_num):
            in_channels = self.in_channels
            if i >= 2:
                in_channels = self.feat_channels + 2
            p5_conv.append(self.Double_conv(self.kernel_size_list[i], in_channels))
            if i >= self.coordconv_block_num - 3:
                p5_conv.append(nn.Upsample(
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=False)
                )
        self.p2_conv = nn.ModuleList(p2_conv)
        self.p3_conv = nn.ModuleList(p3_conv)
        self.p4_conv = nn.ModuleList(p4_conv)
        self.p5_conv = nn.ModuleList(p5_conv)
    def generate_coord(self, input_feat_shape,device):
        x_range = torch.linspace(
            -1, 1, input_feat_shape[-1], device=device)
        y_range = torch.linspace(
            -1, 1, input_feat_shape[-2], device=device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([input_feat_shape[0], 1, -1, -1])
        x = x.expand([input_feat_shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        return coord_feat
    def MultiScaleCoordConv(self,img):
        p2,p3,p4,p5=img
        for conv in self.p2_conv:
            p2 = conv(p2)#[bs,feat_channels,h,w]
        p2=torch.cat([p2,self.generate_coord(p2.shape,p2.device)],1)#[bs,feat_channels+2,h,w],[1,258,128,128]
        #p3_cood_idx=[3,4]
        # p4_cood_idx = [2,4,5]
        # p5_cood_idx = [1,3,5,6]
        # p6_cood_idx = [0,2,4,6,7]
        p3_cood_idx = [3]
        p4_cood_idx = [2,4]
        p5_cood_idx = [1,3,5]
        for idx,conv in enumerate(self.p3_conv):
            p3=conv(p3)
            if idx in p3_cood_idx:
                p3=torch.cat([p3,self.generate_coord(p3.shape,p3.device)],1)
        for idx,conv in enumerate(self.p4_conv):
            p4=conv(p4)
            if idx in p4_cood_idx:
                p4=torch.cat([p4,self.generate_coord(p4.shape,p4.device)],1)
        for idx,conv in enumerate(self.p5_conv):
            p5=conv(p5)
            if idx in p5_cood_idx:
                p5=torch.cat([p5,self.generate_coord(p5.shape,p5.device)],1)
        MultiScale_feats=[p2,p3,p4,p5]#[bs,258,128,128]
        #[bs,258,128,128],[bs,260,128,128]
        if self.fuse_by_cat:
            feature_all_level = self.conv_pred(torch.cat(MultiScale_feats, dim=1))
        else:
            feature_all_level = self.conv_pred(sum(MultiScale_feats))#[bs,c+2,h,w]->[bs,c,h,w]
        loc_feats = self.loc_convs(feature_all_level)
        semantic_feats = self.seg_convs(feature_all_level)
        return semantic_feats,loc_feats# the dim of each stage is [bs,feat_channels,h,w]
    #[bs,feat_channels,h,w],[B,N,H,W],[B,N,class_num]
    def forward(self,img):
        semantic_feats,loc_feats = self.MultiScaleCoordConv(img)
        return semantic_feats,loc_feats
class Center_ResNet_FPN(nn.Module):
    def __init__(self,encoder,depth,heads,head_conv,**kwargs):
        super(Center_ResNet_FPN, self).__init__()
        #encoder，resnet50
        #{'ct_hm': 1, 'wh': commen.points_per_poly * 2}
        #head_conv 256
        self.encoder=resnet_fpn_backbone(encoder,pretrained=True,trainable_layers=depth)
        self.mult_scale_coordconv=MultiscaleCoordConvHead(in_channels=256,feat_channels=256,out_channels=256)
        self.depth=depth
        self.heads=heads
        for head in sorted(self.heads):
          num_output = self.heads[head]
          if head_conv > 0:
            fc = nn.Sequential(
                nn.Conv2d(64, head_conv,
                  kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, num_output,
                  kernel_size=1, stride=1, padding=0))
          else:
            fc = nn.Conv2d(
              in_channels=256,
              out_channels=num_output,
              kernel_size=1,
              stride=1,
              padding=0
          )
          self.__setattr__(head, fc)
    def forward(self,x):
        multi_feat = self.encoder(x)
        multi_feat_list=[v for k, v in multi_feat.items()][:self.depth]
        semantic_feats,loc_feats=self.mult_scale_coordconv(multi_feat_list)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(loc_feats)
        return ret,semantic_feats
# if __name__ == '__main__':
#     a=[torch.randn(1,256,128,128),torch.randn(1,256,64,64),torch.randn(1,256,32,32),torch.randn(1,256,16,16)]
#     #('0', torch.Size([1, 256, 128, 128])), ('1', torch.Size([1, 256, 64, 64])), ('2', torch.Size([1, 256, 32, 32])), (
#     #'#3', torch.Size([1, 256, 16, 16]))
#     model=MultiscaleCoordConvHead()
#     semantic_feats,loc_feats=model(a)
#     print(semantic_feats.shape)
#     print(loc_feats.shape)


