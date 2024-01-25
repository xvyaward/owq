import torch
import torch.nn as nn
import math

layer_list = ['q','k','v','qkv','o','out','dense','fc1','fc2','up','gate','down']

def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def parsing_layers(model, meta):
    from collections import OrderedDict
    results = OrderedDict({'layers':None,'pre_layers':[],'post_layers':[]})
    for data_name in results.keys():
        data = meta[data_name]
        if isinstance(data, list):
            for data_ in data:
                root_attr = model
                attrs = data_.split('.')[1:]
                for attr in attrs:
                    root_attr = getattr(root_attr,attr)
                results[data_name].append(root_attr)
        else: # str
            root_attr = model
            attrs = data.split('.')[1:]
            for attr in attrs:
                root_attr = getattr(root_attr,attr)
            results[data_name] = root_attr

    return results.values()

def interpret_dtype(dtype):
    if isinstance(dtype, str):
        if dtype in ['float16', 'fp16']:
            return torch.half
        elif dtype in ['bfloat16', 'bf16']:
            return torch.bfloat16
        elif dtype in ['float', 'float32', 'fp32', 'fp']:
            return torch.float32
        elif dtype == 'auto':
            return dtype
        else:
            raise ValueError
    elif isinstance(dtype, torch.dtype):
        return dtype
    elif dtype is None:
        return 'auto' # for torch_dtype in AutoModelLM.from_pretrained
    else:
        raise ValueError

def seed_all(seed):
    import random
    import os
    import numpy as np
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def processing_arguments(args):
    import json
    if args.target_bit:
        assert args.wbits < 16, 'FP16 does not need target_bit option'
        assert args.wbits == math.floor(args.target_bit), 'target_bit should be (wbits <= target_bit < wbits+1)'
        if args.tuning != 'mse':
            print('\nWe highly recommend using the mse option together when using OWQ.')
    elif args.target_rank:
        assert args.target_rank > 0
    else:
        if args.wbits < 16 and not args.nearest and args.tuning == 'mse':
            args.tuning = 'minmax'
            print("GPTQ use minmax rtn quantization. args.tuning is manually set minmax.")
    
    if args.save:
        if not (args.save.endswith('.pth') or args.save.endswith('.pt')):
            raise ValueError("The save path '--args.save' must end in .pth or .pt.")
        if not (args.fake or args.packing):
            raise ValueError("""At least one of --fake or --packing must be given to save the model.
                             If both options are given, both methods are saved."""
                )
        if args.packing and args.wbits not in [3, 4]:
            raise ValueError("Only 3 or 4 bit quantized model packing is supported.")
    else:
        if args.fake or args.packing:
            raise ValueError("You must specify a save path in --save when given --fake or --packing.")
    
    with open('model_config.json') as f:
        metas = json.load(f)

    args.recon = True
    args.dtype = interpret_dtype(args.dtype)
    
    # model config
    if 'opt' in args.model:
        meta = metas['opt']
        if '350m' in args.model:
            meta['pre_layers'].append('model.model.decoder.project_in')
            meta['post_layers'].append('model.model.decoder.project_out')
        else:
            meta['post_layers'].append('model.model.decoder.final_layer_norm')
    elif 'llama' in args.model or 'vicuna' in args.model:
        meta = metas['llama']
    elif 'bloom' in args.model:
        meta = metas['bloom']
    elif 'falcon' in args.model:
        meta = metas['falcon']
        args.trust_remote_code = True
        if args.percdamp < 1.0:
            print(f"In the falcon model, change --percdamp from {args.percdamp} to 1.0 for numerical stability.")
            args.percdamp = 1.0
    else:
        raise NotImplementedError(f"{args.model} model is not implemented.")
    
    map_layer = meta['map_layer']
    layers_owq = {l:False for l in map_layer.values()}
    if args.layers is None: # apply owq on all layers
        for l in layers_owq:
            layers_owq[l] = True
    else:
        for l in args.layers:
            if l in map_layer:
                layers_owq[map_layer[l]] = True
            else:
                raise ValueError(f"{args.model} model doesn't have \'{l}\' layer. available layers : {list(map_layer.keys())}")
    for l in layers_owq:
        if not layers_owq[l]:
            meta['ratios'][l] = 0.0
    
    meta['owq_layers'] = layers_owq

    return meta
