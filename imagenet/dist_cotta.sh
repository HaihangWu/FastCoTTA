#!/usr/bin/env bash

#CONFIG=$1
#Method=$2
#GPUS=$3
#PORT=$(expr 20000 + $(echo -n $SLURM_JOBID | tail -c 4))
#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#    $(dirname "$0")/cotta.py $CONFIG  --launcher pytorch --method $Method ${@:4}

cd imagenet

#CUDA_VISIBLE_DEVICES=0 python -u imagenetc.py --cfg cfgs/source.yaml
#CUDA_VISIBLE_DEVICES=0 python -u imagenetc.py --cfg cfgs/norm.yaml

for i in {0..9}
do
#    CUDA_VISIBLE_DEVICES=0 python -u imagenetc.py --cfg cfgs/10orders/tent/tent$i.yaml
#    CUDA_VISIBLE_DEVICES=0 python -u imagenetc.py --cfg cfgs/10orders/cotta/cotta$i.yaml
#    CUDA_VISIBLE_DEVICES=0 python -u imagenetc.py --cfg cfgs/10orders/fastcotta/cotta$i.yaml
#    CUDA_VISIBLE_DEVICES=0 python -u imagenetc.py --cfg cfgs/10orders/ETA/cotta$i.yaml
done
# Run Mean and AVG for TENT, CoTTA
cd output
python3 -u ../eval.py | tee result.log