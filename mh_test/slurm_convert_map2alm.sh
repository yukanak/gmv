#!/bin/bash
#SBATCH --job-name=map2alm
#SBATCH --time=1:00:00
#SBATCH --array=41-99
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition=kipac

export OMP_NUM_THREADS=12

~/gmv/mh_test/convert_map2alm.py ${SLURM_ARRAY_TASK_ID}
