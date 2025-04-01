#!/bin/bash
#SBATCH --job-name=map2alm
#SBATCH --time=2:00:00
#SBATCH --array=100-250
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition=kipac

export OMP_NUM_THREADS=12

~/gmv/tests/convert_map2alm_gaussiancmb.py ${SLURM_ARRAY_TASK_ID}
#~/gmv/tests/convert_map2alm_20240904.py ${SLURM_ARRAY_TASK_ID}
