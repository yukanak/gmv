#!/bin/bash
#SBATCH --job-name=ij_get_plms_unified
#SBATCH --time=1:00:00
#SBATCH --array=1-800
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --partition=kipac

#ests=(TT TE EE TB EB TTprf all TTEETE TBEB TTEETEprf)
ests=(TT TE EE TB EB all TTEETE TBEB)
#qe=${ests[$SLURM_ARRAY_TASK_ID-1]}
qe=${ests[$SLURM_ARRAY_TASK_ID%8-1]}
sim1=$(((SLURM_ARRAY_TASK_ID-1)/8+1))
sim2=$(((SLURM_ARRAY_TASK_ID-1)/8+2))

export OMP_NUM_THREADS=12

~/gmv/get_plms_unified.py $qe $sim1 $sim2
