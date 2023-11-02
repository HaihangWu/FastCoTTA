#!/usr/bin/env bash

CONFIG=$1
WorkDir=$2
GPUS=$3
#PORT=${PORT:-29505}
PORT=$(expr 20000 + $(echo -n $SLURM_JOBID | tail -c 4))
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG  --launcher pytorch --work-dir $WorkDir ${@:4}
