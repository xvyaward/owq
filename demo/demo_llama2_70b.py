import torch
import argparse

from threading import Thread
import subprocess
from typing import Iterator

from transformers import AutoTokenizer, TextIteratorStreamer
import gradio as gr

import os
import sys
sys.path.append('../')
from owq.utils.modelutils import *
from owq.utils.misc import *
from owq.quant import *

'''
    This demo referenced the llama-2-13b-chat demo by huggingface,
    https://huggingface.co/spaces/huggingface-projects/llama-2-13b-chat.
'''

## new llama-2 options
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""
MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = 4000

def processing_arguments_simple():
    import json

    with open('../model_config.json') as f:
        metas = json.load(f)

    return metas['llama']

def model_multigpu(model, gpus, args):
    import math

    layers, pre_layers, post_layers = parsing_layers(model=model, meta=args.meta)
    
    for pre_layer in pre_layers:
        pre_layer = pre_layer.to(gpus[0])
    
    for post_layer in post_layers:
        post_layer = post_layer.to(gpus[0])
    
    model.lm_head = model.lm_head.to(gpus[0])
    cache = {kw:None for kw in meta['inp_kwargs']}

    class MoveModule(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.dev = next(iter(self.module.parameters())).device
        def forward(self, *inp, **kwargs):
            inp = list(inp)
            if inp[0].device != self.dev:
                inp[0] = inp[0].to(self.dev)
            if kwargs['attention_mask'].device != self.dev:
                kwargs['attention_mask'] = kwargs['attention_mask'].to(self.dev)
            if kwargs['position_ids'].device != self.dev:
                kwargs['position_ids'] = kwargs['position_ids'].to(self.dev)
            tmp = self.module(*inp, **kwargs)
            return tmp

    pergpu = math.ceil(len(layers) / len(gpus))
    for i in range(len(layers) - 1):
        layers[i] = MoveModule(layers[i].to(gpus[i // pergpu]))
    layers[-1] = MoveModule(layers[-1].to(gpus[0]))

    model.gpus = gpus

def main(args):
    model_name = args.model.split('/')[-1].upper()

    global multigpu
    global id1, id2
    id1, id2 = None, None
    gpus_list = args.gpus.split(',')
    
    assert len(gpus_list) < 3, "support only 1,2 gpus"
    multigpu = True if len(gpus_list) > 1 else False

    if multigpu:
        id1, id2 = gpus_list
        
        dev1 = torch.device(f'cuda:{id1}')
        dev2 = torch.device(f'cuda:{id2}')

        if args.load:
            model = load_model(args.model, args.load, device='cpu')
        else:
            model = get_hfmodel(args.model, device_map='cpu')

        tokenizer = AutoTokenizer.from_pretrained(args.model)

        gpus = [dev1, dev2]
        model_multigpu(model, gpus, args)
    
    else:
        id1 = args.gpus
        
        dev1 = torch.device(f'cuda:{id1}')

        if args.load:
            model = load_model(args.model, args.load, device=dev1, cpu_load=True)
        else:
            model = get_hfmodel(args.model, device_map=dev1)

        tokenizer = AutoTokenizer.from_pretrained(args.model)

        gpus = [dev1]
    
    print(f"Quantized model {model_name} is successfully loaded into gpu:{args.gpus}")
    
    ## new model part
    def get_prompt(message: str, chat_history: list[tuple[str, str]],
                system_prompt: str) -> str:
        texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
        # The first user input is _not_ stripped
        do_strip = False
        for user_input, response in chat_history:
            user_input = user_input.strip() if do_strip else user_input
            do_strip = True
            texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
        message = message.strip() if do_strip else message
        texts.append(f'{message} [/INST]')
        return ''.join(texts)


    def get_input_token_length(message: str, chat_history: list[tuple[str, str]], system_prompt: str) -> int:
        prompt = get_prompt(message, chat_history, system_prompt)
        input_ids = tokenizer([prompt], return_tensors='np', add_special_tokens=False)['input_ids']
        return input_ids.shape[-1]


    def run(message: str,
            chat_history: list[tuple[str, str]],
            system_prompt: str,
            max_new_tokens: int = 1024,
            temperature: float = 0.8,
            top_p: float = 0.95,
            top_k: int = 50) -> Iterator[str]:
        prompt = get_prompt(message, chat_history, system_prompt)
        inputs = tokenizer([prompt], return_tensors='pt', add_special_tokens=False).to(model.device)

        streamer = TextIteratorStreamer(tokenizer,
                                        timeout=10.,
                                        skip_prompt=True,
                                        skip_special_tokens=True)
        generate_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            num_beams=1,
        )
        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()

        outputs = []
        for text in streamer:
            outputs.append(text)
            yield ''.join(outputs)

    ## new gradio part
    def clear_and_save_textbox(message: str) -> tuple[str, str]:
        return '', message


    def display_input(message: str,
                    history: list[tuple[str, str]]) -> list[tuple[str, str]]:
        history.append((message, ''))
        return history


    def delete_prev_fn(
            history: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], str]:
        try:
            message, _ = history.pop()
        except IndexError:
            message = ''
        return history, message or ''


    def generate(
        message: str,
        history_with_input: list[tuple[str, str]],
        system_prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> Iterator[list[tuple[str, str]]]:
        if max_new_tokens > MAX_MAX_NEW_TOKENS:
            raise ValueError

        history = history_with_input[:-1]
        generator = run(message, history, system_prompt, max_new_tokens, temperature, top_p, top_k)
        try:
            first_response = next(generator)
            yield history + [(message, first_response)]
        except StopIteration:
            yield history + [(message, '')]
        for response in generator:
            yield history + [(message, response)]


    def process_example(message: str) -> tuple[str, list[tuple[str, str]]]:
        generator = generate(message, [], DEFAULT_SYSTEM_PROMPT, 1024, 1, 0.95, 50)
        for x in generator:
            pass
        return '', x


    def check_input_token_length(message: str, chat_history: list[tuple[str, str]], system_prompt: str) -> None:
        input_token_length = get_input_token_length(message, chat_history, system_prompt)
        if input_token_length > MAX_INPUT_TOKEN_LENGTH:
            raise gr.Error(f'The accumulated input is too long ({input_token_length} > {MAX_INPUT_TOKEN_LENGTH}). Clear your chat history and try again.')


    def GPUChecker():
        mem_dict = {}
        lines = subprocess.check_output('nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader',shell=True).decode().split('\n')
        for line in lines:
            if line != '':
                idx, used, total = line.split(', ')
                if idx in [id1, id2]:
                    mem_dict[idx] = f'{used}  /  {total}'
        if multigpu:
            return f'Memory Usage : ' + mem_dict[id1], f'Memory Usage : ' + mem_dict[id2]
        else:
            return f'Memory Usage : ' + mem_dict[id1]
    
    examples = ['There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?',
                ]
    
    with gr.Blocks(
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("<h1><center>OWQ Demo</center></h1>")
        with gr.Row():
            with gr.Column():
                gr.Markdown(f"<font size='4'><p style='text-align: center;'> **LLaMA-2-70B-chat / OWQ 3.01 bit** </p></font>")
                chatbot = gr.Chatbot(height=600)
        
        with gr.Row():
            mem1 = gr.Textbox(value='',show_label=False)
            if multigpu:
                mem2 = gr.Textbox(value='',show_label=False)
        
        with gr.Row():
            msg = gr.Textbox(
                container=False,
                show_label=False,
                label="Input Message Box",
                placeholder="Enter text and press enter, or click the submit button.",
            )
        
        with gr.Row():
            with gr.Column():
                submit = gr.Button("Send message")
                examples = gr.Examples(examples=examples, inputs=msg)
            with gr.Column():
                stop = gr.Button("Stop generation")
                clear = gr.Button("Clear chat")
        
        saved_input = gr.State()
        system_prompt = gr.State(DEFAULT_SYSTEM_PROMPT)

        with gr.Row():
            with gr.Accordion("Generate Options:", open=False):
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            temperature = gr.Slider(
                                label="Temperature",
                                value=0.7,
                                minimum=0.0,
                                maximum=1.0,
                                step=0.1,
                                interactive=True,
                                info="Higher values produce more varied output",
                            )
                    with gr.Column():
                        with gr.Row():
                            top_p = gr.Slider(
                                label="Top-p",
                                value=0.9,
                                minimum=0.0,
                                maximum=1,
                                step=0.01,
                                interactive=True,
                                info=(
                                    "Sampling from the smallest set of tokens whose cumulative probability is top_p."
                                    "If set to 1, sample from all tokens."
                                ),
                            )
                    with gr.Column():
                        with gr.Row():
                            top_k = gr.Slider(
                                label="Top-k",
                                value=0,
                                minimum=0,
                                maximum=100,
                                step=1,
                                interactive=True,
                                info=(
                                    "Sampling from k tokens in descending order of probability.",
                                    "If set to 0, sample from all tokens."
                                )
                            )
                    with gr.Column():
                        with gr.Row():
                            max_new_tokens = gr.Slider(
                                label="Max-new-token",
                                value=DEFAULT_MAX_NEW_TOKENS,
                                minimum=128,
                                maximum=1536,
                                step=128,
                                interactive=True,
                                info="Maximum number of generated token.",
                            )

        submit_event = msg.submit(
            fn=clear_and_save_textbox,
            inputs=msg,
            outputs=[msg, saved_input],
            api_name=False,
            queue=False,
        ).then(
            fn=display_input,
            inputs=[saved_input, chatbot],
            outputs=chatbot,
            queue=False,
        ).then(
            fn=check_input_token_length,
            inputs=[saved_input, chatbot, system_prompt],
            api_name=False,
            queue=False,
        ).success(
            fn=generate,
            inputs=[
                saved_input,
                chatbot,
                system_prompt,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
            ],
            outputs=chatbot,
            api_name=False,
        )
        submit_click_event = submit.click(
            fn=clear_and_save_textbox,
            inputs=msg,
            outputs=[msg, saved_input],
            queue=False,
        ).then(
            fn=display_input,
            inputs=[saved_input, chatbot],
            outputs=chatbot,
            queue=False,
        ).then(
            fn=check_input_token_length,
            inputs=[saved_input, chatbot, system_prompt],
            api_name=False,
            queue=False,
        ).success(
            fn=generate,
            inputs=[
                saved_input,
                chatbot,
                system_prompt,
                max_new_tokens,
                temperature,
                top_p,
                top_k,
            ],
            outputs=chatbot,
            api_name=False,
        )
        stop.click(
            fn=None,
            inputs=None,
            outputs=None,
            cancels=[submit_event, submit_click_event, submit_event, submit_click_event],
            queue=False,
        )
        clear.click(torch.cuda.empty_cache, None, chatbot, queue=False)

        if multigpu:
            demo.load(GPUChecker,None,[mem1,mem2],every=1)
        else:
            demo.load(GPUChecker,None,[mem1],every=1)
        
    demo.queue()
    demo.launch(share=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model', type=str,
        help='chatbot hf model path.'
    )
    parser.add_argument(
        '--load', type=str, default=None,
        help='model owq ckpt path.'
    )
    parser.add_argument(
        '--gpus', type=str, required=True,
        help='local rank of gpu devices.'
    )
    parser.add_argument(
        '--trust_remote_code', action='store_true',
    )
    
    args = parser.parse_args()
    meta = processing_arguments_simple()
    args.meta = meta

    main(args)