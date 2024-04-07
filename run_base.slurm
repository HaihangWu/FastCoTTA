#!/bin/bash
# The project ID which this job should run under:
#SBATCH --account="punim0512"

# The name of the job:
# --job-name="example.py"

# Partition for the job:
#SBATCH --partition deeplearn
#SBATCH --qos gpgpudeeplearn

# --partition=feit-gpu-a100
# --qos=feit

# -p gpu-a100

# Request different amount of resources for different nodes
# --constraint=[dlg1|dlg2|dlg3]
# --constraint=[dlg4|dlg5]
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G

# Maximum number of tasks/CPU cores used by the job:
# --ntasks=4
# --gpus-per-task=1  # number of gpus per task

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=00-23:00:00

module load Anaconda3/2022.10
eval "$(conda shell.bash hook)"
conda activate Uniformerv2

# The job command(s):
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
echo "r$SLURM_NODEID Launching python script"

#bash ./imagenet/dist_cotta.sh  | tee imagenetc.log

#bash ./tools/dist_cotta.sh local_configs/ours/segformer.b5.1024x1024.acdc.160k.py Ours 1 | tee Ours_segformer_acdc_k40.log
bash ./tools/dist_cotta.sh local_configs/ours/segformer.b5.1024x1024.night.160k.py Ours 1 | tee Ours_segformer_night_k30.log

#bash ./tools/dist_cotta.sh local_configs/ours/segnext.base.1024x1024.acdc.160k.py Ours 1 | tee Ours_segnext_acdc.log
#bash ./tools/dist_cotta.sh local_configs/ours/segnext.base.1024x1024.night.160k.py Ours 1 | tee Ours_segnext_night.log

#bash ./tools/dist_train.sh local_configs/ours/segformer.b5.1024x1024.city.160k.py  work_dirs/DPTsgf2000/ 1  | tee finetune_segformer4500.log
#bash ./tools/dist_train.sh local_configs/ours/segnext.base.1024x1024.city.160k.py   work_dirs/DPTsgnext2000/ 1  | tee finetune_segnext4500.log

#bash ./tools/dist_cotta.sh local_configs/ours/DPT_segformer.b5.1024x1024.acdc.160k.py DPT 1 | tee Ours_segformer_acdc.log
#bash ./tools/dist_cotta.sh local_configs/ours/DPT_segformer.b5.1024x1024.night.160k.py DPT 1 | tee Ours_segformer_night.log
#bash ./tools/dist_cotta.sh local_configs/ours/DPT_segnext.base.1024x1024.acdc.160k.py DPT 1 | tee Ours_segnext_acdc.log
#bash ./tools/dist_cotta.sh local_configs/ours/DPT_segnext.base.1024x1024.night.160k.py DPT 1 | tee Ours_segnext_night.log
