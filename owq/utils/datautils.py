import numpy as np
import random
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, LlamaTokenizer

def get_wikitext2(nsamples, seed, seqlen, tokenizer, train):
    if train:
        traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
        
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
            
        return trainloader
            
    else:
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
        
        return testenc

def get_ptb(nsamples, seed, seqlen, tokenizer, train):
    if train:
        traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
        trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')

        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
            
        return trainloader
    else:
        testdata = load_dataset("ptb_text_only", "penn_treebank", split="test")
        testenc = tokenizer(" ".join(testdata["sentence"]), return_tensors="pt")
        
        return testenc

def get_c4(nsamples, seed, seqlen, tokenizer, train):
    if train:
        traindata = load_dataset(
            'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
        )
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
                if trainenc.input_ids.shape[1] > seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        
        return trainloader
    else:
        valdata = load_dataset(
            'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
        )
        valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
        valenc = valenc.input_ids[:, :(256 * seqlen)]

        class TokenizerWrapper:
            def __init__(self, input_ids):
                self.input_ids = input_ids
        valenc = TokenizerWrapper(valenc)

        return valenc

def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model='', train=True
):
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    if isinstance(tokenizer, LlamaTokenizer) and 'ptb' in name:
        tokenizer.tokens_trie.data = {}
    
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer, train)
    elif 'ptb' in name:
        return get_ptb(nsamples, seed, seqlen, tokenizer, train)
    elif 'c4' in name:
        return get_c4(nsamples, seed, seqlen, tokenizer, train)
    else: # custom dataset
        print(f"Custom dataset load from {name}")
        datas = torch.load(name)
        ids_shuffle = list(range(len(datas)))
        random.shuffle(ids_shuffle)
        return [tuple(datas[idx].unsqueeze(0)) for idx in ids_shuffle[:nsamples]]