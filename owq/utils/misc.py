import torch
import torch.nn as nn
import math

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_arguments(args):
    if args.target_bit:
        assert args.wbits < 16, 'FP16 does not need target_bit option'
        assert args.wbits == math.floor(args.target_bit), 'target_bit should be (wbits <= target_bit < wbits+1)'
        if args.tuning != 'mse':
            print('\nWe highly recommend using the mse option together when using OWQ')

