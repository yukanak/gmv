#!/bin/bash
#SBATCH --job-name=get_plms_unified
#SBATCH --time=10:00:00
#SBATCH --array=1-800
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --partition=kipac

#ests=(TT TE EE TB EB TTprf all TTEETE TBEB TTEETEprf)
ests=(TT TE EE TB EB all TTEETE TBEB)
qe=${ests[$SLURM_ARRAY_TASK_ID%8-1]}
sim1=$(((SLURM_ARRAY_TASK_ID-1)/8+1))
sim2=$(((SLURM_ARRAY_TASK_ID-1)/8+2))

~/gmv/get_plms_unified.py $qe $sim1 $sim1
