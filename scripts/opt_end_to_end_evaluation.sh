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

project_path=$(dirname $(dirname $(realpath $0)))
cd $project_path     # move to project directory

logfile="./opt-${model}_owq.txt"
checkpoint="./opt-${model}_owq.pth"
CUDA_VISIBLE_DEVICES=$dev python opt.py facebook/opt-$model $dataset --wbits $wbits --target_bit 3.01 --logfile $logfile --save $checkpoint
for task in 'lambada_openai' 'piqa' 'arc_challenge,arc_easy'; do
    CUDA_VISIBLE_DEVICES=$dev python zeroshot.py facebook/opt-$model --load $checkpoint --batch_size 1 --tasks $task --logfile $logfile
done
# rm $checkpoint

### example usage
# sh opt_end_to_end_evaluation.sh 0 1.3b
