import transformers
import torch
from lm_eval.base import BaseLM
from transformers import OPTForCausalLM, AutoTokenizer
import torch.nn.functional as F
from torch import nn
import torch
from tqdm import tqdm

import sys
sys.path.append('../../')

from owq.quant import *
from owq.recon import *
from owq.utils.misc import find_layers

class OPTLM(BaseLM):
    def __init__(
        self,
        model,
        batch_size=1,
        device=None,
        args='',
    ):
        
        super().__init__()

        if device == None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = device
        self.model_name = model
        self.batch_size_per_gpu = batch_size
        self.args = args

        def skip(*args, **kwargs):
            pass
        torch.nn.init.kaiming_uniform_ = skip
        torch.nn.init.uniform_ = skip
        torch.nn.init.normal_ = skip
        self.model = OPTForCausalLM.from_pretrained(self.model_name, torch_dtype='auto')
        self.seqlen = self.model.config.max_position_embeddings
        self.model.eval()

        # pretrained tokenizer for neo is broken for now so just hard-coding this to gpt2
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self.vocab_size = self.tokenizer.vocab_size
        print('OPT vocab size: ', self.vocab_size)

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self.model.config.max_position_embeddings
    
    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        # TODO: fix multi-gpu
        return self.batch_size_per_gpu  # * gpus

    @property
    def device(self):
        # TODO: fix multi-gpu
        return self._device

    def tok_encode(self, string: str):
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps):
        """
        inps: a torch tensor of shape [batch, sequence]
        the size of sequence may vary from call to call
        returns: a torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model
        """
        with torch.no_grad():
            return self.model(inps)[0][:, :, :50272]

    @torch.no_grad()
    def _model_logits_on_dataset(self, dataset_inps):
        print('Evaluating ...')

        nsamples = len(dataset_inps)

        model = self.model
        dev = self.device

        use_cache = model.config.use_cache
        model.config.use_cache = False
        layers = model.model.decoder.layers

        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
        layers[0] = layers[0].to(dev)

        dtype = next(iter(model.parameters())).dtype
        inps = []
        outs = []
        for batch_idx, batch in enumerate(dataset_inps):
            inps.append(torch.zeros(
                (batch.shape[1], self.model.config.hidden_size), dtype=dtype,
            ))
            outs.append(torch.zeros(
                (batch.shape[1], self.model.config.hidden_size), dtype=dtype,
            ))

        cache = {'i': 0, 'attention_masks': []}

        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, inp, **kwargs):
                inps[cache['i']] = inp
                cache['i'] += 1
                cache['attention_masks'].append(kwargs['attention_mask'].detach().cpu())
                raise ValueError

        layers[0] = Catcher(layers[0])
        for i in range(nsamples):
            batch = dataset_inps[i].to(dev)
            try:
                model(batch)
            except ValueError:
                pass
        layers[0] = layers[0].module

        layers[0] = layers[0].cpu()
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
        torch.cuda.empty_cache()

        attention_masks = cache['attention_masks']

        for i in tqdm(range(len(layers))):
            layer = layers[i].to(dev)

            if self.args.nearest:
                subset = find_layers(layer)
                for name in subset:
                    quantizer = Quantizer()
                    quantizer.configure(
                        self.args.wbits, perchannel=True, sym=False, mse=False
                    )
                    W = subset[name].weight.data
                    quantizer.find_params(W, weight=True)
                    subset[name].weight.data = quantize(
                        W, quantizer.scale, quantizer.zero, quantizer.maxq
                    ).to(next(iter(layer.parameters())).dtype)

            for j in range(nsamples):
                outs[j] = layer(inps[j].to(self.device), attention_mask=attention_masks[j].to(self.device))[0].detach().cpu()

            layers[i] = layer.cpu()
            del layer
            torch.cuda.empty_cache()
            inps, outs = outs, inps

        if model.model.decoder.final_layer_norm is not None:
            model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        if model.model.decoder.project_out is not None:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        model.lm_head = model.lm_head.to(dev)


        if self.model.model.decoder.final_layer_norm is not None:
            self.model.model.decoder.final_layer_norm = self.model.model.decoder.final_layer_norm.to(self.device)
        if self.model.model.decoder.project_out is not None:
            self.model.model.decoder.project_out = self.model.model.decoder.project_out.to(self.device)
        self.model.lm_head = self.model.lm_head.to(self.device)

        dataset_logits = []

        for i in tqdm(range(nsamples), desc='Last Layer'):
            hidden_states = inps[i].unsqueeze(0).to(self.device)
            if self.model.model.decoder.final_layer_norm is not None:
                hidden_states = self.model.model.decoder.final_layer_norm(hidden_states)
            if self.model.model.decoder.project_out is not None:
                hidden_states = self.model.model.decoder.project_out(hidden_states)
            batch_logits = F.log_softmax(self.model.lm_head(hidden_states)[0][:, :, :50272], dim=-1).cpu()
            dataset_logits.append(batch_logits)
        model.config.use_cache = use_cache
        return dataset_logits

    def _model_generate(self, context, max_length, eos_token_id):
        return self.model.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )

# for backwards compatibility
OPT = OPTLM
