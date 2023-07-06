#!/bin/bash
#SBATCH --job-name=get_plms_prfhrd
#SBATCH --time=14:00:00
#SBATCH --array=1-90
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --partition=kipac

ests=(TT TE EE TB EB TTprf all TTEETE TBEB)
#qe=${ests[$SLURM_ARRAY_TASK_ID-1]}
qe=${ests[$SLURM_ARRAY_TASK_ID%9-1]}
#sim1=$(((SLURM_ARRAY_TASK_ID-1)/9+101))
sim1=$(((SLURM_ARRAY_TASK_ID-1)/9+100))
#sim2=$(((SLURM_ARRAY_TASK_ID-1)/9+101))
sim2=$(((SLURM_ARRAY_TASK_ID-1)/9+100+1))
#sim1=100
#sim2=100
#ests=(all TTEETE TBEB)
#qe=${ests[$SLURM_ARRAY_TASK_ID%3-1]}
#sim1=$(((SLURM_ARRAY_TASK_ID-1)/3+100))
#sim2=$(((SLURM_ARRAY_TASK_ID-1)/3+100+1))

~/gmv/get_plms_prfhrd.py $qe $sim1 $sim2
