#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH -J wandb_sweep
#SBATCH --output=%N-%j.out

ENTITY=$1
PROJECT=$2
SWEEP_ID=$3

# load the necessary modules
module load arch/avx512 StdEnv/2020 gcc/9.3.0 python/3.8.10 cuda/11.0 cudacore/.11.0.2 cudnn/8.0.3 nccl/2.7.8

# set base path
BASE_PATH=/scratch/p2irc/p2irc_plotvision/flagship3/gmi672

# load virtual environment
source "$BASE_PATH/venvs/SSL4PP/bin/activate"
cd "$BASE_PATH/SSL4PP"

export NCCL_ASYNC_ERROR_HANDLING=1
export HYDRA_FULL_ERROR=1 # for full stack trace

export MASTER_ADDR=$(hostname)
export MASTER_PORT=45674
export WANDB_AGENT_MAX_INITIAL_FAILURES=5

wandb agent --entity ${ENTITY} --project ${PROJECT} ${SWEEP_ID}

echo "JOB DONE!"
