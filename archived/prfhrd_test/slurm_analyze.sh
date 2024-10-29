#!/bin/bash
#SBATCH --job-name=analyze
#SBATCH --time=1:00:00
#SBATCH --array=1-1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition=kipac

export OMP_NUM_THREADS=12

python3 /home/users/yukanaka/gmv/prfhrd_test/analyze_gmv_unified_prfhrd_modified202402.py
