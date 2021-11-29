#!/bin/sh
#SBATCH -p v
#SBATCH --gres=gpu:1
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

python3 run_finetune.py --base_model gpt2ja-medium --dataset finetune.npz --run_name gpt2ja-finetune_run1