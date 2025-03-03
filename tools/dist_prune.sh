#!/usr/bin/env bash

CONFIG=$1
CONFIG_TEST=$2
num_rm_blocks=$3
GPUS=$4
PORT=$(expr 20000 + $(echo -n $SLURM_JOBID | tail -c 4))
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/prune_TENT.py $CONFIG $CONFIG_TEST --launcher pytorch --num_rm_blocks $num_rm_blocks ${@:5}


#CONFIG=$1
#num_rm_blocks=$2
#GPUS=$3
#PORT=$(expr 20000 + $(echo -n $SLURM_JOBID | tail -c 4))
#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#    $(dirname "$0")/prune.py $CONFIG $CONFIG_TEST --launcher pytorch --num_rm_blocks $num_rm_blocks  ${@:4}
