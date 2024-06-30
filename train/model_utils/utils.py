import torch
import os
import torch.nn.functional
from termcolor import colored

def load_model(net, optim, scheduler, recorder, model_path):
    strict = True

    if not os.path.exists(model_path):
        print(colored('WARNING: NO MODEL LOADED !!!', 'red'))
        return 0

    print('load model: {}'.format(model_path))
    pretrained_model = torch.load(model_path, map_location={'cuda:0': 'cpu', 'cuda:1': 'cpu',
                                                            'cuda:2': 'cpu', 'cuda:3': 'cpu'})
    net.load_state_dict(pretrained_model['net'], strict=strict)
    optim.load_state_dict(pretrained_model['optim'])
    scheduler.load_state_dict(pretrained_model['scheduler'])
    recorder.load_state_dict(pretrained_model['recorder'])
    return pretrained_model['epoch'] + 1

def save_model(net, optim, scheduler, recorder, epoch, model_dir):
    os.system('mkdir -p {}'.format(model_dir))
    torch.save({
        'net': net.state_dict(),
        'optim': optim.state_dict(),
        'scheduler': scheduler.state_dict(),
        'recorder': recorder.state_dict(),
        'epoch': epoch
    }, os.path.join(model_dir, '{}.pth'.format(epoch)))
    return

def save_weight(net, model_dir):
    os.system('mkdir -p {}'.format(model_dir))
    torch.save({
        'net': net.state_dict(),
    }, os.path.join(model_dir, '{}.pth'.format('final')))
    return

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
