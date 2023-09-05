#!/bin/bash
#SBATCH --job-name=get_plms_unified
#SBATCH --time=18:00:00
#SBATCH --array=1-300
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --partition=kipac

ests=(TT TE EE TB EB TTprf all TTEETE TBEB TTEETEprf)
#ests=(TT TE EE TB EB all TTEETE TBEB)
#qe=${ests[$SLURM_ARRAY_TASK_ID-1]}
qe=${ests[$SLURM_ARRAY_TASK_ID%10-1]}
sim1=$(((SLURM_ARRAY_TASK_ID-1)/10+100))
sim2=$(((SLURM_ARRAY_TASK_ID-1)/10+101))

~/gmv/get_plms_unified.py $qe $sim1 $sim1
