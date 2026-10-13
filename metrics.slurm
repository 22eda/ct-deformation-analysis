#!/bin/bash
#SBATCH --job-name=metrics_eval
#SBATCH --output=metrics_%j.out
#SBATCH --error=metrics_%j.err
#SBATCH -A plgaimed1-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00

cd $SCRATCH/deformation_project
source myenv/bin/activate
python -u metrics_evaluation.py test
