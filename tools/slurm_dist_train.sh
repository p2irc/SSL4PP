#!/bin/bash

#SBATCH -J SSL4PP
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=2-00:00
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=8
#SBATCH --output=%N-%j.out
#SBATCH --ntasks-per-node=4

PY_ARGS=${@:1}

## load the necessary modules
module load arch/avx512 StdEnv/2020 gcc/9.3.0 python/3.8.10 cuda/11.0 cudacore/.11.0.2 cudnn/8.0.3 nccl/2.7.8

# set base path
BASE_PATH=/scratch/p2irc/p2irc_plotvision/flagship3/gmi672

# load virtual environment
source "$BASE_PATH/venvs/SSL4PP/bin/activate"
cd "$BASE_PATH/SSL4PP"

export NCCL_ASYNC_ERROR_HANDLING=1 # set to 1 if using NCCL backend for inter-GPU communication
export HYDRA_FULL_ERROR=1 # for full stack trace

export MASTER_ADDR=$(hostname)
export MASTER_PORT=45678

# “srun” executes the script <ntasks-per-node * nodes> times
srun python run_benchmark.py distributed.launcher="slurm" ${PY_ARGS}

echo "Job done!"
echo "Submitted at " echo $SLURM_SUBMIT_DIR
