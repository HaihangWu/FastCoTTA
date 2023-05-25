#!/bin/bash
# The project ID which this job should run under:
#SBATCH --account="punim0512"

# The name of the job:
#SBATCH --job-name="example.py"

# Partition for the job:
#SBATCH --partition deeplearn
#SBATCH --qos gpgpudeeplearn

# --partition=feit-gpu-a100
# --qos=feit

# -p gpu-a100

# Request different amount of resources for different nodes
#SBATCH --constraint=[dlg1|dlg2|dlg3]
# --constraint=[dlg3|dlg4|dlg5]
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G

# Maximum number of tasks/CPU cores used by the job:
# --ntasks=4
# --gpus-per-task=1  # number of gpus per task

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time=00-15:00:0

module load anaconda3/2021.11
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

#bash ./tools/dist_language_cotta.sh local_configs/Language/Lsegformer.b0.1024x1024.night.160k.py  1 5 2.5 1.0 0.5 "segformer" | tee LSegb0-night-Lcotta-lambdadecay.log
bash ./tools/dist_language_cotta.sh local_configs/Language/Lsegformer.b5.1024x1024.acdc.160k.py  1 | tee LSegb5-acdc-Lcotta.log


#bash ./tools/dist_language_cotta-xiao.sh local_configs/Language/Lsegformer.b5.1024x1024.acdc.160k.py  1 5 2.5 1.0 0.5 "segformer" | tee LSegb5-acdc-cotta-lambdadecay-xiao.log
#bash ./tools/dist_language_cotta-xiao.sh local_configs/Language/Lsetr_vit-large_pup_8x1_768x768_80k_acdc.py  1 5 2.5 1.0 0.5 "setr" | tee SETR-acdc-Lcotta-lambdadecay-xiao.log


#bash ./tools/dist_language_cotta.sh local_configs/Language/Lsetr_vit-large_pup_8x1_768x768_80k_night.py  1 5 2.5 1.0 0.5 "setr" | tee Lsetr-night-Lcotta-lambdadecay.log
#bash ./tools/dist_language_cotta.sh local_configs/Language/Lsetr_vit-large_pup_8x1_768x768_80k_acdc.py  1 5 2.5 1.0 0.5 "setr" | tee Lsetr-acdc-Lcotta-lambdadecay.log

#bash ./tools/dist_cotta.sh local_configs/Language/Lsegformer.b5.1024x1024.night.160k.py  1 | tee LSegb5-night-cotta-train.log
#bash ./tools/dist_cotta.sh local_configs/Language/Lsetr_vit-large_pup_8x1_768x768_80k_night.py  1 | tee Lsetr-night-cotta-train.log

bash ./tools/dist_language_test.sh local_configs/Language/Lsegformer.b0.1024x1024.acdc.160k.py   1 | tee LSegb0-acdc-BN.log
#bash ./tools/dist_language_test.sh local_configs/Language/Lsetr_vit-large_pup_8x1_768x768_80k_acdc.py  1 | tee Lsetr-acdc-BN.log

#outlier_num 5
#z_threshold 2.5
#lang_rgz 1.0
#adp_termination 0.5
#model_name setr