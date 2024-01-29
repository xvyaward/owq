import torch
import argparse

from threading import Thread
import subprocess

from transformers import AutoTokenizer, TextIteratorStreamer
import gradio as gr

import os
import sys
sys.path.append('../')
from owq.utils.modelutils import *
from owq.utils.misc import *
from owq.quant import *

def main(args):
    assert len(args.gpus.split(',')) == 2, "Two GPU devices are required. Please enter them separated by commas"

    global id1, id2
    id1, id2 = args.gpus.split(',')
    fmodel_name = args.fmodel.split('/')[-1].upper()
    qmodel_name = args.qmodel.split('/')[-1].upper()
    
    dev1 = torch.device(f'cuda:{id1}')
    dev2 = torch.device(f'cuda:{id2}')
    
    if args.load1:
        fmodel = load_model(args.fmodel, args.load1, device=dev1, cpu_load=True)
    else:
        fmodel = get_hfmodel(args.fmodel, device_map=dev1)

    ftok = AutoTokenizer.from_pretrained(args.fmodel)

    print(f"Model {fmodel_name} is successfully loaded into gpu:{id1}")

    if args.load2:
        qmodel = load_model(args.qmodel, args.load2, device=dev2, cpu_load=True)
    else:
        qmodel = get_hfmodel(args.qmodel, device_map=dev2)
    
    qtok = AutoTokenizer.from_pretrained(args.qmodel)
    
    print(f"Quantized model {qmodel_name} is successfully loaded into gpu:{id2}")
    
    # default_max_new_tokens = 1536
    default_max_new_tokens = 512
    start_message = """A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."""
    
    def convert_history_to_text(history):
        text = start_message + "".join(
            [
                "".join(
                    [
                        f"### Human: {item[0]}\n",
                        f"### Assistant: {item[1]}\n",
                    ]
                )
                for item in history[:-1]
            ]
        )
        text += "".join(
            [
                "".join(
                    [
                        f"### Human: {history[-1][0]}\n",
                        f"### Assistant: {history[-1][1]}\n",
                    ]
                )
            ]
        )
        return text

        
    def user(message, history):
        return "", history + [[message, ""]]

        
    def bot1(history, temperature, top_p, top_k, max_new_tokens):
        messages = convert_history_to_text(history)

        input_ids = ftok(messages, return_tensors="pt").input_ids
        input_ids = input_ids.to(fmodel.device)
        streamer = TextIteratorStreamer(ftok, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0.0,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=1.0,
            streamer=streamer,
        )

        thread = Thread(target=fmodel.generate, kwargs=generate_kwargs)
        thread.start()

        partial_text = ""
        for new_text in streamer:
            partial_text += new_text
            history[-1][1] = partial_text
            yield history
            
    def bot2(history, temperature, top_p, top_k, max_new_tokens):
        messages = convert_history_to_text(history)

        input_ids = qtok(messages, return_tensors="pt").input_ids
        input_ids = input_ids.to(qmodel.device)
        streamer = TextIteratorStreamer(qtok, timeout=10.0, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0.0,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=1.0, # disable
            streamer=streamer,
        )

        thread = Thread(target=qmodel.generate, kwargs=generate_kwargs)
        thread.start()

        partial_text = ""
        for new_text in streamer:
            partial_text += new_text
            history[-1][1] = partial_text
            yield history


    def GPUChecker():
        mem_dict = {}
        lines = subprocess.check_output('nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader',shell=True).decode().split('\n')
        for line in lines:
            if line != '':
                idx, used, total = line.split(', ')
                if idx in [id1, id2]:
                    mem_dict[idx] = f'{used}  /  {total}'
        return f'Memory Usage : ' + mem_dict[id1], f'Memory Usage : ' + mem_dict[id2]
    
    examples = ['There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?',
                ]
    
    with gr.Blocks(
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown("<h1><center>OWQ Demo</center></h1>")
        with gr.Row():
            with gr.Column():
                gr.Markdown(f"<font size='4'><p style='text-align: center;'> {fmodel_name} / FP16 </p></font>")
                chatbot1 = gr.Chatbot(height=600)
            with gr.Column():
                gr.Markdown(f"<font size='4'><p style='text-align: center;'> **{qmodel_name} / OWQ 3.01 bit** </p></font>")
                chatbot2 = gr.Chatbot(height=600)
        
        with gr.Row():
            mem1 = gr.Textbox(value='',show_label=False)
            mem2 = gr.Textbox(value='',show_label=False)
        
        with gr.Row():
            msg = gr.Textbox(
                label="Input Message Box",
                placeholder="Enter text and press enter, or click the submit button.",
                show_label=False,
                container=False
            )
        
        with gr.Row():
            with gr.Column():
                submit = gr.Button("Send message")
                examples = gr.Examples(examples=examples, inputs=msg)
            with gr.Column():
                stop = gr.Button("Stop generation")
                clear = gr.Button("Clear chat")
        
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
                                value=default_max_new_tokens,
                                minimum=128,
                                maximum=1536,
                                step=128,
                                interactive=True,
                                info="Maximum number of generated token.",
                            )
        
        submit_event2 = msg.submit(
            fn=user,
            inputs=[msg, chatbot2],
            outputs=[msg, chatbot2],
            queue=False,
        ).then(
            fn=bot2,
            inputs=[
                chatbot2,
                temperature,
                top_p,
                top_k,
                max_new_tokens,
            ],
            outputs=chatbot2,
            queue=True,
        )
        submit_event1 = msg.submit(
            fn=user,
            inputs=[msg, chatbot1],
            outputs=[msg, chatbot1],
            queue=False,
        ).then(
            fn=bot1,
            inputs=[
                chatbot1,
                temperature,
                top_p,
                top_k,
                max_new_tokens,
            ],
            outputs=chatbot1,
            queue=True,
        )
        submit_click_event2 = submit.click(
            fn=user,
            inputs=[msg, chatbot2],
            outputs=[msg, chatbot2],
            queue=False,
        ).then(
            fn=bot2,
            inputs=[
                chatbot2,
                temperature,
                top_p,
                top_k,
                max_new_tokens,
            ],
            outputs=chatbot2,
            queue=True,
        )
        submit_click_event1 = submit.click(
            fn=user,
            inputs=[msg, chatbot1],
            outputs=[msg, chatbot1],
            queue=False,
        ).then(
            fn=bot1,
            inputs=[
                chatbot1,
                temperature,
                top_p,
                top_k,
                max_new_tokens,
            ],
            outputs=chatbot1,
            queue=True,
        )
        stop.click(
            fn=None,
            inputs=None,
            outputs=None,
            cancels=[submit_event1, submit_click_event1, submit_event2, submit_click_event2],
            queue=False,
        )
        clear.click(torch.cuda.empty_cache, None, chatbot2, queue=False)
        clear.click(torch.cuda.empty_cache, None, chatbot1, queue=False)

        demo.load(GPUChecker,None,[mem1,mem2],every=1)
        
    demo.queue()
    demo.launch(share=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'fmodel', type=str,
        help='chatbot1 hf model path.'
    )
    parser.add_argument(
        'qmodel', type=str,
        help='chatbot2 hf model path.'
    )
    parser.add_argument(
        '--load1', type=str, default=None,
        help='fmodel owq ckpt path.'
    )
    parser.add_argument(
        '--load2', type=str, default=None,
        help='qmodel owq ckpt path.'
    )
    parser.add_argument(
        '--gpus', type=str, required=True,
        help='local rank of gpu devices.'
    )
    parser.add_argument(
        '--trust_remote_code', action='store_true',
    )
    
    args = parser.parse_args()
    main(args)