#!/bin/bash
#SBATCH --job-name=unet_train
#SBATCH --output=train_%j.out
#SBATCH --error=train_%j.err
#SBATCH -A plgaimed1-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00

cd $SCRATCH/deformation_project
source myenv/bin/activate

nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv

python -u train.py
