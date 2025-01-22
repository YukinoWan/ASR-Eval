#!/bin/bash
#SBATCH --job-name=llm_eval
#SBATCH --partition=p2
#SBATCH --nodelist=cnode7-004
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source ~/miniconda3/etc/profile.d/conda.sh 


echo "Current directory: $(pwd)"
echo "Available Python: $(which python)"
echo "Conda environment: $(conda info --envs)"


conda activate audio

python llm_eval.py


