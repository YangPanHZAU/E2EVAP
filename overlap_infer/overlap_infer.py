import sys
sys.path.append('./')
from network import make_network
import tqdm
import torch
import time
import nms
import os
from dataset.data_loader import make_data_loader
from evaluator.coco.snake_save import Evaluator
from termcolor import colored
import argparse
import importlib
def load_network(net, model_dir, strict=True):

    if not os.path.exists(model_dir):
        print(colored('WARNING: NO MODEL LOADED !!!', 'red'))
        return 0

    print('load model: {}'.format(model_dir))
    pretrained_model = torch.load(model_dir, map_location={'cuda:0': 'cpu', 'cuda:1': 'cpu',
                                                           'cuda:2': 'cpu', 'cuda:3': 'cpu'})
    if 'epoch' in pretrained_model.keys():
        epoch = pretrained_model['epoch'] + 1
    else:
        epoch = 0
    pretrained_model = pretrained_model['net']

    net_weight = net.state_dict()
    #print(pretrained_model.keys())
    for key1, key2 in zip(net_weight.keys(), pretrained_model.keys()):
        net_weight.update({key1: pretrained_model[key2]})

    net.load_state_dict(net_weight, strict=strict)
    return epoch
parser = argparse.ArgumentParser()
parser.add_argument("config_file", help='/path/to/config_file.py')
parser.add_argument("--checkpoint", default='', help='/path/to/model_weight.pth')
parser.add_argument("--dataset", default='None', help='test dataset name')
parser.add_argument("--with_nms", default=False, type=bool,
                    help='if True, will use nms post-process operation', choices=[True, False])
parser.add_argument("--eval", default='segm', help='evaluate the segmentation or detection result',
                    choices=['segm', 'bbox'])
parser.add_argument("--stage", default='final-dml', help='which stage of the contour will be generated',
                    choices=['init', 'coarse', 'final', 'final-dml'])
parser.add_argument("--type", default='accuracy', help='evaluate the accuracy or speed',
                    choices=['speed', 'accuracy'])
parser.add_argument("--device", default=0, type=int, help='device idx')

args = parser.parse_args()
def get_cfg(args):
    cfg = importlib.import_module('configs_overlap_infer.' + args.config_file).config
    cfg.test.with_nms = bool(args.with_nms)
    cfg.test.segm_or_bbox = args.eval
    cfg.test.test_stage = args.stage
    if args.dataset != 'None':
        cfg.test.dataset = args.dataset
    return cfg
def run_evaluate(cfg):
    network = make_network.get_network(cfg).cuda()
    load_network(network, args.checkpoint)
    network.eval()
    data_loader = make_data_loader(is_train=False, cfg=cfg)
    from dataset.info import DatasetInfo
    anno_file = DatasetInfo.dataset_info[cfg.test.dataset]['anno_dir']
    evaluator = Evaluator(cfg.commen.result_dir, anno_file)
    thr=0.1
    #evaluator = make_evaluator(cfg)
    for batch in tqdm.tqdm(data_loader):
        inp = batch['inp'].cuda()
        with torch.no_grad():
            output = network(inp, batch)
        if cfg.test.with_nms:
            nms.post_process(output)
        evaluator.evaluate(output, batch,thr)
    evaluator.save()
if __name__ == "__main__":
    cfg = get_cfg(args)
    torch.cuda.set_device(args.device)
    run_evaluate(cfg)
