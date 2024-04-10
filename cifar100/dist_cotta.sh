#!/usr/bin/env bash

#CONFIG=$1
#Method=$2
#GPUS=$3
#PORT=$(expr 20000 + $(echo -n $SLURM_JOBID | tail -c 4))
#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#    $(dirname "$0")/cotta.py $CONFIG  --launcher pytorch --method $Method ${@:4}

cd cifar100

#CUDA_VISIBLE_DEVICES=0 python -u imagenetc.py --cfg cfgs/source.yaml
#CUDA_VISIBLE_DEVICES=0 python -u imagenetc.py --cfg cfgs/norm.yaml

for i in {0..0}
do
#  CUDA_VISIBLE_DEVICES=0 python cifar100c.py --cfg cfgs/cifar100/source.yaml
#  CUDA_VISIBLE_DEVICES=0 python cifar100c.py --cfg cfgs/cifar100/norm.yaml
  CUDA_VISIBLE_DEVICES=0 python cifar100c.py --cfg cfgs/cifar100/tent.yaml
#  CUDA_VISIBLE_DEVICES=0 python cifar100c.py --cfg cfgs/cifar100/cotta.yaml
#  CUDA_VISIBLE_DEVICES=0 python -u imagenetc.py --cfg cfgs/10orders/ETA/cotta$i.yaml
#  CUDA_VISIBLE_DEVICES=0 python -u imagenetc.py --cfg cfgs/10orders/fastcotta/cotta$i.yaml
done







# Run Mean and AVG for TENT, CoTTA
#cd output
#python3 -u ../eval.py | tee result.log