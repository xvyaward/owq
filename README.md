# OWQ: Lessons learned from activation outliers for weight quantization in large language models

This is the code for the paper [OWQ: Lessons learned from activation outliers for weight quantization in large language models](https://arxiv.org/abs/2306.02272). OWQ preserves few weak columns as FP16, while compressing other weight coulmns to 3/4-bits. OWQ achieves substantial quality improvements with only negligible storage
and computation overhead, effectively preserving the benefits of low-precision acceleration.


The current release supports following features:
* Implementation of the OWQ algorithm: `recon.py`
* 3/4-bit weight quantization of OPT, LLaMA, and BLOOM families: `opt.py`, `llama.py`, `bloom.py`
* Evaluating the perplexity of quantized models: `opt.py`, `llama.py`, `bloom.py`
* Evaluating the zero-shot accuracy of quantized models: `zeroshot.py`
* Supports 3-bit packed weight save / load (~1/5 file size of FP16 checkpoint)
* Efficient 3-bit matrix - FP16 vector product CUDA kernel for OWQ: `owq/kernel`


Features we are working on:
* Integrating all models (OPT, LLaMA, BLOOM) into single file
* Efficient matrix-matrix multiplication CUDA kernel for OWQ
* Efficient W4A16 CUDA kernel 

## Table of contents
* [Install](#install)
* [Usage (measuring perplexity)](#usage)
* [Zero-shot](#zero-shot)
* [3-bit CUDA kernel](#3-bit-cuda-kernels)

## Install
We highly recommend to use docker image that supports CUDA. If you use anaconda instead, you need to setup CUDA for kernel use.

0. A) Using Docker
```
docker run -it --gpus all --ipc=host -v {local_storage}:{docker_container_storage} pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

# install git
apt update && apt install git -y
```

0. B) Using anaconda instead of docker
```
conda create -n owq python=3.10 -y
conda activate owq
```

1. Clone the OWQ repository
```
git clone https://github.com/xvyaward/owq
cd owq
```
2. Install all the dependencies

```
pip install -r requirements.txt
```
3. Install 3-bit CUDA kernel (3bit_W x FP16_A)
```
cd owq/kernel
python setup_cuda.py install
```
* `torch`: tested on v2.0.0+cu117
* `transformers`: tested on v4.29.2
* `datasets`: tested on v2.12.0

Experiments were conducted on a single NVIDIA A100 GPU with 80GB memory. We also confirmed that reconstruction using OWQ works on RTX 3090 GPU (24GB memory) for <= 30B models.

We have tested 3-bit CUDA kernel on the NVIDIA A100 GPU and A6000 GPU.

## Usage

### Running OWQ & measuring the perplexity (PPL)


#### OPT example
Here we use OPT-1.3b model as an example. You can replace the model argument `opt-1.3b` among `opt-125m`, `opt-350m`, `opt-2.7b`, `opt-6.7b`, `opt-13b`, `opt-66b`.

* OWQ using 3.01-bit (3-bit quantization + few FP16 weight columns)
```
python opt.py facebook/opt-1.3b c4 --wbits 3 --target_bit 3.01
```
* OWQ using 4.01-bit (4-bit quantization + few FP16 weight columns)
```
python opt.py facebook/opt-1.3b c4 --wbits 4 --target_bit 4.01
```
Please refer to `scripts/` for more examples.

Below are the example for the other options (FP16, RTN, GPTQ). 
```
# Measuring the ppl of the full precision (FP16) model
python opt.py facebook/opt-1.3b c4 --wbits 16

# 4-bit Round-to-Nearest (RTN) quantization
python opt.py facebook/opt-1.3b c4 --wbits 4 --nearest

# GPTQ with 3-bit quantization
python opt.py facebook/opt-1.3b c4 --wbits 3 --tuning minmax
```



The above usage examples for OPT models can be used same for other model families as well.
### LLaMA
* OWQ using 3.01-bit (3-bit quantization + few FP16 weight columns)
```
python llama.py {llama-model-location} c4 --wbits 3 --target_bit 3.01
```

### BLOOM
* OWQ using 3.01-bit (3-bit quantization + few FP16 weight columns)
```
python bloom.py bigscience/bloom-560m c4 --wbits 3 --target_bit 3.01
```

To run other BLOOM models replace `bloom-560m` with one of: `bloom-1b1`, `bloom-1b7`, `bloom-3b`, `bloom-7b1`, `bloom`.


## Zero-shot
Here we give an example of measuring zero-shot accuracy on `lambada_openai` and `piqa` tasks using opt-125m model.
Current version only supports measuring zeroshot accuracy from the saved model. You need checkpoint file before measuring the zero-shot accuracy.
```
# making checkpoint file of OWQ reconstruction
python opt.py facebook/opt-125m c4 --wbits 3 --target_bit 3.05 --no-eval --save {checkpoint-file}

# measuring zero-shot accuracy
python zeroshot.py facebook/opt-125m --load {checkpoint-file} --batch_size 8 --task lambada_openai,piqa
```



## 3-bit CUDA Kernels 

### Benchmark kernel performance
```
# Benchmark performance for the matrix multiplication
cd owq/kernel/
python test_kernel.py
```

### Benchmark language generation with 3-bit packed model (opt, llama)
```
# Example of OPT-65b language generation (single token)

# Save compressed model
python opt.py facebook/opt-66b c4 --wbits 3 --target_bit 3.01 --no-eval --save {checkpoint-file} --packing

# Benchmark generating a 128 token sequence with the saved model
CUDA_VISIBLE_DEVICE=0 python opt.py facebook/opt-66b c4 --load {pack3_checkpoint-file} --packing --benchmark 128

# Benchmark FP16 baseline, note that the model will be split across all listed GPUs
CUDA_VISIBLE_DEVICES=0,1,2 python opt.py facebook/opt-66b c4 --benchmark 128
```
if you save quantized model with `--packing` option, this gives 3-bit packed checkpoint with name `pack3_{checkpoint-file}` together with fake quantized model `{checkpoint-file}`.

Please note that our 3-bit kernels are currently only optimized for A100 or A6000 GPUs and may thus yield suboptimal performance on smaller models or on other GPUs.



## Reference

[GPTQ: Accurate Post-training Compression for Generative Pretrained Transformers](https://arxiv.org/abs/2210.17323)

This code is based on [GPTQ](https://github.com/IST-DASLab/gptq).

Our zero-shot experiment codes are based on [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

Thanks to Meta AI for releasing powerful LLM [OPT](https://arxiv.org/abs/2205.01068) and [LLaMA](https://arxiv.org/abs/2302.13971).
## Cite

If you find our code or OWQ useful for your research, please consider citing:

```
@article{lee2023owq,
  title={OWQ: Lessons learned from activation outliers for weight quantization in large language models},
  author={Lee, Changhun and Jin, Jungyu and Kim, Taesu and Kim, Hyungjun and Park, Eunhyeok},
  journal={arXiv preprint arXiv:2306.02272},
  year={2023}
}
```
