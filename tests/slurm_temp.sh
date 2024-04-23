#!/bin/bash
#SBATCH --job-name=temp
#SBATCH --time=2:00:00
#SBATCH --array=1-1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition=kipac

export OMP_NUM_THREADS=12

python3 ~/gmv/tests/compare_gmv_analytic_resp.py
