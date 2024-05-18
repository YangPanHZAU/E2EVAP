import torch.nn as nn
import torch.nn.functional as F
from .snake import Snake
from .utils import img_poly_to_can_poly_minmax, prepare_training, prepare_testing_init, img_poly_to_can_poly, get_gcn_feature
import torch
from network.detector_decode.refine_decode import Multi_attention

def conv3x3_gn_relu(c_in,c_cout):
    return nn.Sequential(
            nn.Conv2d(
                c_in, c_cout, (3, 3), stride=1, padding=1, bias=False
            ),
            nn.GroupNorm(4, c_cout),
            nn.ReLU(inplace=True),
        )
def conv3x3_sigmoid(c_in,c_cout):
    return nn.Sequential(
            nn.Conv2d(
                c_in, c_cout, (3, 3), stride=1, padding=1, bias=False
            ),
            nn.Sigmoid(),
        )
class Edge_att(nn.Module):
    def __init__(self, c_in, c_feat,cout):
        super(Edge_att, self).__init__()
        self.att=nn.Sequential(
            conv3x3_gn_relu(c_in,c_feat),
            conv3x3_gn_relu(c_feat,c_feat),
            conv3x3_sigmoid(c_feat,cout),
        )
    def forward(self,x):
        return self.att(x)
class Edge_predict(nn.Module):
    def __init__(self, c_in=64, c_feat=32,cout=1):
        super(Edge_predict, self).__init__()
        self.predict=nn.Sequential(nn.Conv2d(c_in, c_feat, 1, bias=False),
                                         nn.BatchNorm2d(c_feat),
                                         nn.ReLU(inplace=True),
                                         nn.Conv2d(c_feat, cout, 1, bias=False))
    def forward(self,x):
        return self.predict(x)
class Evolution_with_edge_att_no_share(nn.Module):
    def __init__(self, evole_ietr_num=3, evolve_stride=1., ro=4.,evolve_poly_multi_attention=False):
        super(Evolution_with_edge_att_no_share, self).__init__()
        assert evole_ietr_num >= 1
        self.evolve_stride = evolve_stride
        self.ro = ro
        self.evolve_gcn = Snake(state_dim=128, feature_dim=64+2, conv_type='dgrid')
        self.iter = evole_ietr_num - 1
        self.evolve_poly_multi_attention=evolve_poly_multi_attention
        self.edge_predictor=Edge_predict(c_in=64, c_feat=32,cout=1)
        self.sigmoid=nn.Sigmoid()
        self.edge_att=Edge_att(c_in=1,c_feat=32,cout=1)
        # self.evolve_poly_att=None
        if self.evolve_poly_multi_attention:
            self.evolve_poly_att=self.multi_attention(1)
        for i in range(self.iter):
            evolve_gcn = Snake(state_dim=128, feature_dim=64+2, conv_type='dgrid')
            edge_predictor = Edge_predict(c_in=64, c_feat=32,cout=1)
            edge_att = Edge_att(c_in=1,c_feat=32,cout=1)
            self.__setattr__('evolve_gcn'+str(i), evolve_gcn)
            self.__setattr__('edge_predictor'+str(i), edge_predictor)
            self.__setattr__('edge_att'+str(i), edge_att)


        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def multi_attention(self,block_num):
        return Multi_attention(block_num=block_num,d_model=128,nhead=4)
    def prepare_training(self, output, batch):
        init = prepare_training(output, batch, self.ro)
        return init

    def prepare_testing_init(self, output):
        init = prepare_testing_init(output['poly_coarse'], self.ro)
        return init

    def prepare_testing_evolve(self, output, h, w):
        img_init_polys = output['img_init_polys']
        img_init_polys[..., 0] = torch.clamp(img_init_polys[..., 0], min=0, max=w-1)
        img_init_polys[..., 1] = torch.clamp(img_init_polys[..., 1], min=0, max=h-1)
        output.update({'img_init_polys': img_init_polys})
        return img_init_polys
    
    def evolve_poly(self, snake,edge_predictor,edge_att, cnn_feature, i_it_poly, c_it_poly, ind, stride=1., ignore=False):
        if ignore:
            egde_result=torch.zeros(cnn_feature.size(0),1,cnn_feature.size(2),cnn_feature.size(3))
            return i_it_poly * self.ro,egde_result
        if len(i_it_poly) == 0:
            egde_result=torch.zeros(cnn_feature.size(0),1,cnn_feature.size(2),cnn_feature.size(3))
            return torch.zeros_like(i_it_poly),egde_result
        h, w = cnn_feature.size(2), cnn_feature.size(3)#8,64,512,512
        egde_result=edge_predictor(cnn_feature)#bs,1,h,w
        edge_feature=edge_att(1-self.sigmoid(egde_result))#bs,1,h,w
        init_feature = get_gcn_feature(cnn_feature, i_it_poly, ind, h, w)#[poly_num,64,128]
        edge_poly_feat = get_gcn_feature(edge_feature, i_it_poly, ind, h, w).permute(0, 2, 1).expand(-1, -1, 2)
        #[poly_num,1,128]->[poly_num,128,1]->[poly_num,128,2]
        c_it_poly = c_it_poly * self.ro#[poly_num, 128, 2]*1
        init_input = torch.cat([init_feature, c_it_poly.permute(0, 2, 1)], dim=1)##[80, 66,128]
        offset = snake(init_input).permute(0, 2, 1)#[80,128,2]
        i_poly = i_it_poly * self.ro + offset* edge_poly_feat * stride
        return i_poly,egde_result
    def foward_train(self, output, batch, cnn_feature):
        ret = output#{dict:ct_hm,wh,poly_init,poly_coarse}
        init = self.prepare_training(output, batch)
#init['img_gt_polys'] [80,128,2],img_init_polys [80,128,2] can_init_polys [80,128,2] py_ind [80]
        py_pred,egde_result = self.evolve_poly(self.evolve_gcn,self.edge_predictor,self.edge_att, cnn_feature, init['img_init_polys'],
                                   init['can_init_polys'], init['py_ind'], stride=self.evolve_stride)
        py_preds = [py_pred]#[80,128,2]
        egde_results =[egde_result]
        for i in range(self.iter):
            py_pred = py_pred / self.ro
            c_py_pred = img_poly_to_can_poly(py_pred)
            evolve_gcn = self.__getattr__('evolve_gcn' + str(i))
            edge_predictor = self.__getattr__('edge_predictor' + str(i))
            edge_att = self.__getattr__('edge_att' + str(i))
            py_pred,egde_result = self.evolve_poly(evolve_gcn,edge_predictor,edge_att, cnn_feature, py_pred, c_py_pred,
                                       init['py_ind'], stride=self.evolve_stride)
            py_preds.append(py_pred)
            egde_results.append(egde_result)
        ret.update({'py_pred': py_preds,'egde_result': egde_results, 'img_gt_polys': init['img_gt_polys'] * self.ro})
        return output

    def foward_test(self, output, cnn_feature, ignore):
        #print(ignore)
        ret = output
        with torch.no_grad():
            init = self.prepare_testing_init(output)
            img_init_polys = self.prepare_testing_evolve(init, cnn_feature.size(2), cnn_feature.size(3))
            py,egde_result = self.evolve_poly(self.evolve_gcn,self.edge_predictor,self.edge_att, cnn_feature, img_init_polys, init['can_init_polys'], init['py_ind'],
                                  ignore=ignore[0], stride=self.evolve_stride)
            pys = [py, ]
            for i in range(self.iter):
                py = py / self.ro
                c_py = img_poly_to_can_poly(py)
                evolve_gcn = self.__getattr__('evolve_gcn' + str(i))
                edge_predictor = self.__getattr__('edge_predictor' + str(i))
                edge_att = self.__getattr__('edge_att' + str(i))
                py,egde_result = self.evolve_poly(evolve_gcn,edge_predictor,edge_att, cnn_feature, py, c_py, init['py_ind'],
                                      ignore=ignore[i + 1], stride=self.evolve_stride)
                pys.append(py)
            ret.update({'py': pys})
        return output

    def forward(self, output, cnn_feature, batch=None, test_stage='final-dml'):
        if batch is not None and 'test' not in batch['meta']:
            self.foward_train(output, batch, cnn_feature)
        else:
            ignore = [False] * (self.iter + 1)
            if test_stage == 'coarse' or test_stage == 'init':
                ignore = [True for _ in ignore]
            if test_stage == 'final':
                ignore[-1] = True
            self.foward_test(output, cnn_feature, ignore=ignore)
        return output
