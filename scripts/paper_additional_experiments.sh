#!/bin/bash

model=6.7b
dataset='c4'
wbits=3

project_path=$(dirname $(dirname $(realpath $0)))
cd $project_path     # move to project directory

# Analysis on layer-wise quantization sensitivity
# For the name of the layers, please refer to the layer_list in each {model_familiy}.py
python opt.py facebook/opt-$model $dataset --wbits $wbits --target_bit 3.01 --layers k q

# Fine-grained (grouped) quantization
python opt.py facebook/opt-$model $dataset --wbits $wbits --target_bit 3.01 --groupsize 1024

# Using True sequential and Activation order options
python opt.py facebook/opt-$model $dataset --wbits $wbits --target_bit 3.01 --true-sequential
python opt.py facebook/opt-$model $dataset --wbits $wbits --target_bit 3.01 --act-order
python opt.py facebook/opt-$model $dataset --wbits $wbits --target_bit 3.01 --true-sequential --act-order