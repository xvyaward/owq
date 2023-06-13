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

layer_list = ['qkv','dense','fc1','fc2']
n_out_dict = {'self_attention.query_key_value':0,
            'self_attention.dense':0,
            'mlp.dense_h_to_4h':0,
            'mlp.dense_4h_to_h':0}

def get_bloom(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import BloomForCausalLM
    model = BloomForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model

@torch.no_grad()
def bloom_sequential(model, dataloader, dev, means=None, stds=None):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.transformer.h

    model.transformer.word_embeddings = model.transformer.word_embeddings.to(dev)
    model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, 'alibi': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['alibi'] = kwargs['alibi']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.transformer.word_embeddings = model.transformer.word_embeddings.cpu()
    model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    alibi = cache['alibi']

    print('Ready.')
    
    if args.target_bit is not None:
        args.layers = layer_list if args.layers is None else args.layers
        n_mp_layers = len(args.layers)
        if 'qkv' in args.layers:
            n_mp_layers += 2 # q k v
        
        r = (12 / (16 - args.wbits)) * (args.target_bit - args.wbits)
        # r = (args.target_bit - args.wbits) * 16 / 12
        r /= n_mp_layers

        layer = find_layers(layers[0])
        
        for i in range(len(args.layers)):
            if args.layers[i] == 'qkv':
                name = 'self_attention.query_key_value'
                n_out_dict[name] = round(layer[name].weight.data.shape[1] * r) * 3
            elif args.layers[i] == 'dense':
                name = 'self_attention.dense'
                n_out_dict[name] = round(layer[name].weight.data.shape[1] * r)
            elif args.layers[i] == 'fc1':
                name = 'mlp.dense_h_to_4h'
                n_out_dict[name] = round(layer[name].weight.data.shape[1] * r / 4)
            elif args.layers[i] == 'fc2':
                name = 'mlp.dense_4h_to_h'
                n_out_dict[name] = round(layer[name].weight.data.shape[1] * r / 4)
    
    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        block_layers = find_layers(layer)
        
        if args.true_sequential:
            sequential = [
                ['self_attention.query_key_value'], ['self_attention.dense'],
                ['mlp.dense_h_to_4h'], ['mlp.dense_4h_to_h']
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
                layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)
            for h in handles:
                h.remove()
            
            for name in names:
                if name.endswith('query_key_value') and args.target_bit is not None:
                    name = 'self_attention.query_key_value'
                    layer_qkv = subset[name]
                    W_q, W_k, W_v = torch.chunk(layer_qkv.weight.data, 3, dim=0)
                    W_attn_dict = {'self_attention.query':W_q, 'self_attention.key':W_k, 'self_attention.value':W_v}
                    for name1 in W_attn_dict:
                        W = W_attn_dict[name1]
                        subset[name1] = nn.Linear(W.shape[1], W.shape[0], device=W.device, dtype=W.dtype)
                        subset[name1].weight.data = W.clone()
                        gptq[name1] = GPTQ_OWQ(subset[name], n_out=n_out_dict[name])
                        gptq[name1].quantizer = Quantizer()
                        gptq[name1].quantizer.configure(
                            args.wbits, perchannel=True, sym=False, mse=(args.tuning == 'mse')
                        )
                        gptq[name1].quantizer.n_out = n_out_dict[name] // 3
                        gptq[name1].H = gptq[name].H.clone()
                            
                    del subset[name]
                    del W_q, W_k, W_v
                    del gptq[name]
                    torch.cuda.empty_cache()
                    break

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
                gptq[name].free()

            for name in names:
                if name.endswith('query_key_value') and args.target_bit is not None:
                    W_qkv = [subset[n].weight.data.clone() for n in W_attn_dict]
                    layer_qkv.weight.data = torch.concat(W_qkv,dim=0)
                    del W_qkv
                    break
        
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()
        
        inps, outs = outs, inps

    model.config.use_cache = use_cache

@torch.no_grad()
def bloom_eval(model, testenc, dev):
    print('Evaluation...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.transformer.h

    model.transformer.word_embeddings = model.transformer.word_embeddings.to(dev)
    model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, 'alibi': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['alibi'] = kwargs['alibi']
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
    model.transformer.word_embeddings = model.transformer.word_embeddings.cpu()
    model.transformer.word_embeddings_layernorm = model.transformer.word_embeddings_layernorm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    alibi = cache['alibi']

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
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, alibi=alibi)[0]
        layers[i] = layer.cpu() 
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.transformer.ln_f = model.transformer.ln_f.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        hidden_states = model.transformer.ln_f(hidden_states)
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='BLOOM model to load; pass `bigscience/bloom-X`.'
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
    
    model = get_bloom(args.model)
    model.eval()
    t = 0
    if args.load:
        print(f"Loading {args.load} ....")
        model.load_state_dict(torch.load(args.load))
        print("Done.")
    else:
        dataloader = get_loaders(
            args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen, train=True
        )
        if args.wbits < 16 and not args.nearest:
            tick = time.time()
            quantizers = bloom_sequential(model, dataloader, device)
            t = round((time.time() - tick),1)
            print(f"Running Time : {t}")

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
            ppl_score = bloom_eval(model, testloader, device)
            ppl_scores.append((dataset,ppl_score))
    t2 = time.time() - t1
    
    if args.logfile:
        with open(f'{args.logfile}','a') as fp:
            add_str = f"\nlayers : {args.layers}" + f"| target_bit : {args.target_bit}\n" if args.target_bit is not None else '\n'
            fp.write(f"model : {args.model} | owq time : {round(t/60,1)}m / eval time : {round(t2/60,1)}m | seed : {args.seed} {add_str}")
            for i in range(len(ppl_scores)):
                fp.write(f"{ppl_scores[i][1]} ")
            fp.write(f"\n\n")

    if args.save:
        torch.save(model.state_dict(), args.save)
