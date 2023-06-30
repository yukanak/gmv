#!/bin/bash
#SBATCH --job-name=get_plms_prfhrd
#SBATCH --time=14:00:00
#SBATCH --array=7-8
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --partition=kipac

ests=(TT TE EE TB EB TTprf all TTEETE TBEB)
qe=${ests[$SLURM_ARRAY_TASK_ID-1]}

~/gmv/get_plms_prfhrd.py $qe 100
