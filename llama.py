import time

import torch
import torch.nn as nn

import transformers

from owq.recon import GPTQ_OWQ
from owq.quant import *
from owq.utils.misc import find_layers, check_arguments
from owq.utils.datautils import *

import argparse
import random
import os
import numpy as np
from tqdm import tqdm

layer_list = ['k','v','q','o','up','gate','down']
n_out_dict = {'self_attn.k_proj':0,
            'self_attn.v_proj':0,
            'self_attn.q_proj':0,
            'self_attn.o_proj':0,
            'mlp.up_proj':0,
            'mlp.gate_proj':0,
            'mlp.down_proj':0 }

def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model

@torch.no_grad()
def llama_sequential(model, dataloader, dev):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )

    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    if args.target_bit is not None:
        args.layers = layer_list if args.layers is None else args.layers
        n_mp_layers = len(args.layers)
        
        r = (12 / (16 - args.wbits)) * (args.target_bit - args.wbits)
        # r = (args.target_bit - args.wbits) * 16 / 12
        r /= n_mp_layers

        layer = find_layers(layers[0])
        
        for i in range(len(args.layers)):
            if args.layers[i] in ('k','v','q','o'):
                name = 'self_attn.' + args.layers[i] + '_proj'
                n_out_dict[name] = round(layer[name].weight.data.shape[1] * r)
            else:
                name = 'mlp.' + args.layers[i] + '_proj'
                n_out_dict[name] = round(layer[name].weight.data.shape[1] * r * 3 / 8)
        
    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        block_layers = find_layers(layer)

        if args.true_sequential:
            sequential = [
                ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                ['self_attn.o_proj'],
                ['mlp.up_proj', 'mlp.gate_proj'],
                ['mlp.down_proj']
            ]
        else:
            sequential = [list(block_layers.keys())]
       
        for names in sequential:
            subset = {n: block_layers[n] for n in names}

            gptq = {}
            for name in subset:
                gptq[name] = GPTQ_OWQ(subset[name], n_out=n_out_dict[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=False, mse=(args.tuning == 'mse')
                )
                gptq[name].quantizer.n_out = n_out_dict[name]
                
            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()
            
            for name in subset:
                if not args.no_frob_norm:
                    W = subset[name].weight.data.clone().to(torch.float)
                    temp_quantizer = Quantizer()
                    temp_quantizer.configure(args.wbits, perchannel=True, sym=False, mse=(args.tuning == 'mse'))
                    temp_quantizer.find_params(W, weight=True, num=40)
                    W_quant = temp_quantizer.quantize(W)
                    frob_norm_error = (W - W_quant).pow(2).sum(dim=0)
                else:
                    frob_norm_error = None
                out_ids = gptq[name].hessian_sorting(actorder=args.act_order, frob_norm=frob_norm_error)
                gptq[name].quantizer.out_ids = out_ids.cpu()
                    
            if not args.no_frob_norm:
                del W
                del W_quant
                del temp_quantizer
                torch.cuda.empty_cache()
            
            for name in subset:
                print(f"Quantizing model.decoder.layers.{i}.{name}")
                gptq[name].fasterquant(percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order)
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer.cpu()
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        outs = torch.nan_to_num(outs)

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    
    return quantizers

@torch.no_grad()
def llama_eval(model, testenc, dev):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        
        if args.nearest:
            subset = find_layers(layer)
            for name in subset:
                quantizer = Quantizer()
                quantizer.configure(
                    args.wbits, perchannel=True, sym=False, mse=False
                )
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantize(
                    W, quantizer.scale, quantizer.zero, quantizer.maxq
                ).to(next(iter(layer.parameters())).dtype)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        outs = torch.nan_to_num(outs)

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache
    return ppl.item()

def load_quant3(model, checkpoint, faster=False):
    from transformers import LlamaConfig, LlamaForCausalLM
    config = LlamaConfig.from_pretrained(model)
    def noop(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = noop
    torch.nn.init.uniform_ = noop
    torch.nn.init.normal_ = noop

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = LlamaForCausalLM(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()
    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]
            
    ckpt = torch.load(checkpoint)
    n_out_dict = ckpt['n_out_dict']
    
    make_quant3(model, n_out_dict, faster=faster)

    model.load_state_dict(ckpt['model_state_dict'])
    model.seqlen = model.config.max_position_embeddings

    return model

def llama_multigpu(model, gpus):
    model.model.embed_tokens = model.model.embed_tokens.to(gpus[0])
    model.model.norm = model.model.norm.to(gpus[-1])
    import copy
    import math
    model.lm_head = copy.deepcopy(model.lm_head).to(gpus[-1])

    cache = {'mask': None, 'pos_ids': None}

    class MoveModule(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.dev = next(iter(self.module.parameters())).device
        def forward(self, *inp, **kwargs):
            inp = list(inp)
            if inp[0].device != self.dev:
                inp[0] = inp[0].to(self.dev)
            if cache['mask'] is None or cache['mask'].device != self.dev:
                cache['mask'] = kwargs['attention_mask'].to(self.dev)
            if cache['pos_ids'] is None or cache['pos_ids'].device != self.dev:
                cache['pos_ids'] = kwargs['position_ids'].to(self.dev)
            kwargs['attention_mask'] = cache['mask']
            kwargs['position_ids'] = cache['pos_ids']
            tmp = self.module(*inp, **kwargs)
            return tmp

    layers = model.model.layers
    pergpu = math.ceil(len(layers) / len(gpus))
    for i in range(len(layers)):
        layers[i] = MoveModule(layers[i].to(gpus[i // pergpu]))

    model.gpus = gpus

def benchmark(model, input_ids):
    dev = torch.device('cuda:0')
    input_ids = input_ids.to(model.gpus[0] if hasattr(model, 'gpus') else dev)
    torch.cuda.synchronize()

    cache = {'past': None}
    def clear_past(i):
        def tmp(layer, inp, out):
            if cache['past']:
                cache['past'][i] = None
        return tmp
    for i, layer in enumerate(model.model.layers):
        layer.register_forward_hook(clear_past(i))

    print('Benchmarking ...')

    loss = nn.CrossEntropyLoss()
    tot = 0.

    def sync():
        if hasattr(model, 'gpus'):
            for gpu in model.gpus:
                torch.cuda.synchronize(gpu)
        else:
            torch.cuda.synchronize()
    with torch.no_grad():
        attention_mask = torch.ones((1, input_ids.numel()), device=dev)
        position_ids = torch.arange(0,input_ids.numel(), device=dev)
        times = []
        for i in range(input_ids.numel()):
            print(i)
            tick = time.time()
            out = model(input_ids[:, i].reshape(1,-1),past_key_values=cache['past'],
                        attention_mask=attention_mask[:, :(i + 1)].reshape((1, -1)),
                        position_ids=position_ids[i])
            sync()
            times.append(time.time() - tick)
            if i != input_ids.numel() - 1:
                tot += loss(out.logits[0].to(dev), input_ids[:, (i + 1)].to(dev)).float()
            cache['past'] = list(out.past_key_values)
            del out
        sync()
        
        print('Median:', np.median(times))
        print('PPL:', torch.exp(tot / (input_ids.numel() - 1)).item())

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='LlaMa model to load; /path/to/llama_hf'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 16],
        help='The number of bits to use for weight quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--target_bit', type=float, default=None,
        help='Effctive target bits for OWQ.'
    )
    parser.add_argument(
        '--tuning', type=str, default='mse', choices=['mse', 'minmax'],
        help='Method for quantization parameter tuning.'
    )
    parser.add_argument(
        '--no_frob_norm', action='store_true',
        help='Whether to use Frobenius norm for OWQ.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--layers', nargs='+', type=str, default=None, choices=layer_list,
        help='Layers to apply OWQ.'
    )
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the round-to-nearest quantization.'
    ) 
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize for fine-grained quantization; default uses full row.'
    )

    parser.add_argument(
        '--no-eval', action='store_true',
        help='Whether to evaluate model on WikiText-2, PTB and C4'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )
    parser.add_argument(
        '--load', type=str, default='',
        help='Load fake or 3bit quantized checkpoint.'
    )
    parser.add_argument(
        '--logfile', type=str, default='',
        help='Logging file name'
    )
    parser.add_argument(
        '--packing', action='store_true',
        help='Whether to save 3bit quantized model.'
    )
    parser.add_argument(
        '--faster-kernel', action='store_true',
        help='Whether to save and load 3bit quantized model using the faster kernel for benchmarking.'
    )
    parser.add_argument(
        '--benchmark', type=int, default=0,
        help='Number of tokens to use for benchmarking.'
    )
    
    parser.add_argument(
        '--old-eval', action='store_true',
        help='Whether to use the old version of PTB and C4 evaluation.'
    )
    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        '--true-sequential', action='store_true',
        help='Whether to run in true sequential model.'
    )

    args = parser.parse_args()
    check_arguments(args)
    device = torch.device('cuda:0')
    
    def seed_all(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    seed_all(args.seed)
    
    t = 0
    if args.load:
        print(f"Loading {args.load} ....")
        if args.packing:
            model = load_quant3(args.model, args.load, args.faster_kernel)
        else:
            model = get_llama(args.model)    
            model.load_state_dict(torch.load(args.load))
        model.eval()
        print("Done.")
    else:
        model = get_llama(args.model)
        model.eval()
    
        if args.wbits < 16 and not args.nearest:
            dataloader = get_loaders(
                args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen, train=True
            )
            tick = time.time()
            quantizers = llama_sequential(model, dataloader, device)
            t = round((time.time() - tick),1)
            print(f"Running Time : {t}")

    if args.benchmark:
        dataloader = get_loaders(
            args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen, train=False
        )
        gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            llama_multigpu(model, gpus)
        else:
            model = model.to(device)
        if args.benchmark:
            input_ids = dataloader.input_ids[:, :args.benchmark]
            benchmark(model, input_ids)
        exit()

    t1 = time.time()
    ppl_scores = []
    if not args.no_eval:
        if args.old_eval:
            ppl_tasks = ['wikitext2', 'ptb', 'c4']
        else:
            ppl_tasks = ['wikitext2','ptb-new', 'c4-new']
        for dataset in ppl_tasks:
            testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen, train=False
            )
            print(dataset)
            ppl_score = llama_eval(model, testloader, device)
            ppl_scores.append((dataset,ppl_score))
    t2 = time.time() - t1
    
    if args.logfile:
        with open(f'{args.logfile}','a') as fp:
            add_str = f"| layers : {args.layers}" + f"| target_bit : {args.target_bit}\n" if args.target_bit is not None else '\n'
            fp.write(f"model : {args.model} | owq time : {round(t/60,1)}m / eval time : {round(t2/60,1)}m | seed : {args.seed} {add_str}")
            for i in range(len(ppl_scores)):
                fp.write(f"{ppl_scores[i][1]} ")
            fp.write(f"\n")

    if args.save:
        torch.save(model.state_dict(), args.save)
        print(f"fake quantized model is saved to {args.save}")
        if args.packing and args.wbits == 3:
            temp = args.save.split('/')
            temp[-1] = 'pack3_' + f"{'faster_' if args.faster_kernel else ''}" + temp[-1]
            ckpt_path = '/'.join(temp)
            n_out_dict = {n: n_out_saver(quantizers[n].n_out) for n in quantizers}
            lm_pack3(model, quantizers, faster=args.faster_kernel)
            torch.save({
                'model_state_dict' : model.state_dict(),
                'n_out_dict' : n_out_dict}, ckpt_path)
            print(f"3bit quantized model is saved to {ckpt_path}")
        else:
            print("Only 3bits quantized model is supported")