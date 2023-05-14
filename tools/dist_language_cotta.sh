#!/usr/bin/env bash

CONFIG=$1
#CHECKPOINT=$2
GPUS=$2
z_threshold=$3
lang_rgz=$4
adp_termination=$5
model_name=$6

#PORT=${PORT:-29505}
PORT=$(expr 20000 + $(echo -n $SLURM_JOBID | tail -c 4)) #wuhh
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/language_cotta.py $CONFIG --launcher pytorch --z_score_threshold z_threshold --lang_rgz lang_rgz --adp_termination adp_termination --model_name model_name ${@:7}
