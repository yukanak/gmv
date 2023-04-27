#!/bin/bash
#SBATCH --job-name=test
#SBATCH --time=14:00:00
#SBATCH --array=1-11
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --partition=kipac

~/gmv/get_plms_test.py B ${SLURM_ARRAY_TASK_ID} qest_gmv_withBTBE_B
