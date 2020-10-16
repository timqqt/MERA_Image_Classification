#!/bin/bash

#SBATCH -e logs/slurm_%A_%a.err                                                   # error log file
#SBATCH --mem=50G                                                      # request 30G memory
#SBATCH -c 6                                                           # request 6 gpu cores
##SBATCH -p gpu-common --gres=gpu:1                                     # request 1 gpu for this job
#SBATCH -p carin-gpu --gres=gpu:1                                     # request 1 gpu for this job

#SBATCH -o logs/slurm_%A_%a.out

conda activate tfquant-gpu

python MERA64.py