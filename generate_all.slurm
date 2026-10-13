#!/bin/bash
#SBATCH --job-name=gen_variants_all
#SBATCH --output=genall_%j.out
#SBATCH --error=genall_%j.err
#SBATCH -A plgaimed1-gpu-a100
#SBATCH -p plgrid-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00

cd $SCRATCH/deformation_project
source myenv/bin/activate
python -u generate_variants.py
