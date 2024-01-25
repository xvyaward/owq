# Chatbot Demo for OWQ

We are providing chatbot demo using OWQ.

The current release supports following demos:
* comparing Vicuna-**7B**-fp16 vs. Vicuna-**33B**-OWQ-3.01bit: `demo_2model.py`
* A cutting-edge **LLaMA-2 70B** model + OWQ-3.01bit: `demo_llama2_70b.py`

## Install & Preparation
0. Install OWQ dependencies following [here](https://github.com/xvyaward/GPTQ_PV/tree/for_release#install).

1. Install additional packages for demo.
```
pip install gradio
pip install protobuf
```

2. Prepare packed 3bit OWQ models


## Usage
### Vicuna-7B (fp16) vs. Vicuna-33B (OWQ 3.01 bit)
Launch two models using local resources.
```
python demo_2model.py lmsys/vicuna-7b-v1.3 lmsys/vicuna-33b-v1.3 --load2 {quantized-vicuna-33b-weight-location} --gpus 0,1
```
Then you can get accessible Link to the demo page. Please enjoy!

Note that **Quantized Vicuna-33B model using our OWQ method gives comparable or better chat quality, with similar memory usage comparing to FP vicuna-7B model.**


### LLaMA-2 70B + OWQ 3.01 bit
Lanuch quantized llama-2-70b model using local resources. (Currently, this need 1x A100 or 2x consumer GPU (e.g. 24GB memory RTX 3090))
* Using a single A100 GPU
```
python demo_llama2_70b.py meta-llama/Llama-2-70b-chat-hf --load {quantized-llama-2-70b-weight-location} --gpus 0
```
* Using two RTX 3090 GPUs
```
python demo_llama2_70b.py meta-llama/Llama-2-70b-chat-hf --load {quantized-llama-2-70b-weight-location} --gpus 0,1
```

Please Note that we can run powerful chatbot model based on **LLaMA-2 70B** model just using **2x consumer GPUs (RTX 3090)**.



## Reference
[LLaMA-2](https://ai.meta.com/llama/)

[Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/)

[Gradio](https://www.gradio.app/)
