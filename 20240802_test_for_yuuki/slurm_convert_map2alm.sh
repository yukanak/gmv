#!/bin/bash
#SBATCH --job-name=map2alm
#SBATCH --time=1:00:00
#SBATCH --array=1-100
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition=kipac

export OMP_NUM_THREADS=12

/home/users/yukanaka/gmv/20240802_test_for_yuuki/convert_map2alm.py ${SLURM_ARRAY_TASK_ID}
