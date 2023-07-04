#!/bin/bash
#SBATCH --job-name=get_plms_prfhrd
#SBATCH --time=14:00:00
#SBATCH --array=1-9
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --partition=kipac

ests=(TT TE EE TB EB TTprf all TTEETE TBEB)
qe=${ests[$SLURM_ARRAY_TASK_ID-1]}
sim1=$(((SLURM_ARRAY_TASK_ID-1)+100))
sim2=$(((SLURM_ARRAY_TASK_ID-1)+100+1))

~/gmv/get_plms_prfhrd.py $qe $sim1 $sim2
