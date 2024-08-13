#!/usr/bin/env bash

CONFIG=$1
Method=$2
GPUS=$3
#PORT=${PORT:-29505}
PORT=$(expr 20000 + $(echo -n $SLURM_JOBID | tail -c 4))
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/svdp.py $CONFIG  --launcher pytorch --ema_rate=0.999 --model_lr=0.0001 --prompt_lr=0.0001 --prompt_sparse_rate=0.001 --scale=0.1
