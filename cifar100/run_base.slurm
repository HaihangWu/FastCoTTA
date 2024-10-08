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
#SBATCH --time=00-10:00:00

module load Anaconda3/2022.10
eval "$(conda shell.bash hook)"
conda deactivate
conda activate Uniformerv2

# The job command(s):
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR
echo "r$SLURM_NODEID Launching python script"


bash ./run_cifar100.sh




