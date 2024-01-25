#!/bin/bash

:<<'END'
    This is an example script for the process of
        Reconstruction using OWQ & PPL measuring    
    =>  model save
    =>  measuring zero-shot accuracies on several tasks using saved checkpoint
END

dev=$1          # gpu number
model=$2        # select among 125m 350m 1.3b 2.7b 6.7b 13b 30b 66b
dataset='c4'    # calibration dataset
wbits=3
target_bit=3.01

project_path=$(dirname $(dirname $(realpath $0)))
cd $project_path     # move to project directory

checkpoint="./opt-${model}_${target_bit}.pth"
CUDA_VISIBLE_DEVICES=$dev python main.py facebook/opt-$model $dataset --wbits $wbits --target_bit $target_bit --packing --save $checkpoint
for task in 'lambada_openai' 'piqa'; do
    CUDA_VISIBLE_DEVICES=$dev python zeroshot.py --model hf-causal-owq --model_args pretrained=facebook/opt-${model},load=${checkpoint} --batch_size 4 --tasks $task --no_cache
done

### example usage
# sh opt_end_to_end_evaluation.sh 0 1.3b
