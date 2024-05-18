import sys
sys.path.append('./')
import torch.nn as nn
from network.backbone.dla import DLASeg
from network.detector_decode.refine_decode_pnp_wo_hrfpn import Decode
from network.evolve.evolve import Evolution_with_edge_att_no_share
import torch
from network.msra_resnet import get_center_resnet
def patch_first_conv(model, in_channels):
    """Change first convolution layer input channels.
    In case:
        in_channels == 1 or in_channels == 2 -> reuse original weights
        in_channels > 3 -> make random kaiming normal initialization
    """

    # get first conv
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            break

    # change input channels for first conv
    module.in_channels = in_channels
    weight = module.weight.detach()
    reset = False

    if in_channels == 1:
        weight = weight.sum(1, keepdim=True)
    elif in_channels == 2:
        weight = weight[:, :2] * (3.0 / 2.0)
    else:
        reset = True
        weight = torch.Tensor(
            module.out_channels,
            module.in_channels // module.groups,
            *module.kernel_size
        )

    module.weight = nn.parameter.Parameter(weight)
    if reset:
        module.reset_parameters()
    return model
class Network(nn.Module):
    def __init__(self, cfg=None):
        super(Network, self).__init__()
        in_channel=cfg.model.in_channel
        encoder=cfg.model.encoder#'resnet50','dla34'
        #num_layers = cfg.model.dla_layer
        head_conv = cfg.model.head_conv
        #decode_c_in =cfg.model.decode_c_in
        down_ratio = cfg.commen.down_ratio
        heads = cfg.model.heads
        self.test_stage = cfg.test.test_stage
        if 'resnet' in encoder:
            dla = get_center_resnet(encoder, heads,head_conv,pretrained=True)
        if 'dla' in encoder:
            dla = DLASeg(encoder, heads,
                              pretrained=True,
                              down_ratio=down_ratio,
                              final_kernel=1,
                              last_level=5,
                              head_conv=head_conv)

        if in_channel!=3:
            self.dla=patch_first_conv(model=dla, in_channels=in_channel)
        else:
            self.dla=dla
        #resnet_encoder='resnet50's
        self.train_decoder = Decode(num_point=cfg.commen.points_per_poly, init_stride=cfg.model.init_stride,
                                        coarse_stride=cfg.model.coarse_stride, down_sample=cfg.commen.down_ratio,
                                        min_ct_score=cfg.test.ct_score)            
        self.gcn = Evolution_with_edge_att_no_share(evole_ietr_num=cfg.model.evolve_iters, evolve_stride=cfg.model.evolve_stride,
                             ro=cfg.model.evolve_down_ratio)

    def forward(self, x, batch=None):
        output, cnn_feature = self.dla(x)
        #ouput size:[]ct_hm [bs,class_num,128,128] wh [bs,256,128,128]
        #cnn_feature size:[]
        #rs50 [8, 256, 128, 128]-》[8, 64, 128, 128]
        if 'test' not in batch['meta']:
            cnn_feature=self.train_decoder(batch, cnn_feature, output, is_training=True)
        else:
            with torch.no_grad():
                if self.test_stage == 'init':
                    ignore = True
                else:
                    ignore = False
                cnn_feature=self.train_decoder(batch, cnn_feature, output, is_training=False, ignore_gloabal_deform=ignore)
        output = self.gcn(output, cnn_feature, batch, test_stage=self.test_stage)
        return output

def get_network(cfg):
    network = Network(cfg)
    return network
