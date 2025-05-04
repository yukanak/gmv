#!/bin/bash
#SBATCH --job-name=random
#SBATCH --time=01:00:00
#SBATCH --array=1-1
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --partition=kipac

export OMP_NUM_THREADS=12

python3 ~/gmv/pipeline/generate_gaussian_inputs.py
