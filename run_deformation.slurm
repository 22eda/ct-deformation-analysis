#!/bin/bash
#SBATCH -J calc_deformation
#SBATCH -p plgrid-gpu-a100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH -o calc_def_%j.out
#SBATCH -e calc_def_%j.err

# Activate virtual environment
source myenv/bin/activate

# Run the Python script
python3 calculate_deformation.py
