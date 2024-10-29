#!/bin/bash
#SBATCH --job-name=get_plms_standard
#SBATCH --time=1:00:00
#SBATCH --array=1-800
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --partition=kipac

ests=(TT TE ET EE TB BT EB BE)
qe=${ests[$SLURM_ARRAY_TASK_ID%8-1]}
sim1=$(((SLURM_ARRAY_TASK_ID-1)/8+1))

export OMP_NUM_THREADS=12

~/gmv/tests/get_plms_standard_rdn0.py $qe r $sim1 standard test_yuka.yaml
~/gmv/tests/get_plms_standard_rdn0.py $qe $sim1 r standard test_yuka.yaml
