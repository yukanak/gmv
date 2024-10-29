#!/bin/bash
#SBATCH --job-name=get_plms_test
#SBATCH --time=14:00:00
#SBATCH --array=1-8
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition=kipac

#ests=(TT TE EE TB EB TTprf all TTEETE TBEB TTEETEprf)
ests=(TT TE EE TB EB all TTEETE TBEB)
#qe=${ests[$SLURM_ARRAY_TASK_ID-1]}
qe=${ests[$SLURM_ARRAY_TASK_ID%8-1]}
sim1=$(((SLURM_ARRAY_TASK_ID-1)/8+100))
#sim2=$(((SLURM_ARRAY_TASK_ID-1)/8+100))
sim2=$(((SLURM_ARRAY_TASK_ID-1)/8+101))

export OMP_NUM_THREADS=12

~/gmv/get_plms_test.py $qe $sim1 $sim1
