#!/bin/bash
#SBATCH --job-name=failed_get_plms_unified
#SBATCH --time=1:00:00
#SBATCH --array=1-35
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition=kipac

#ests=(TT TE EE TB EB TTprf all TTEETE TBEB TTEETEprf)
ests=(TT TE EE TB EB all TTEETE TBEB)
failed_idxs=(515 627 408 626 32 93 406 723 731 96 46 102 42 26 395 94 521 407 405 467 201 269 725 92 307 30 267 23 270 724 401 404 394 522 308)
idx=${failed_idxs[$SLURM_ARRAY_TASK_ID-1]}
qe=${ests[$idx%8-1]}
sim1=$(((idx-1)/8+1))
sim2=$(((idx-1)/8+2))

export OMP_NUM_THREADS=12

~/gmv/get_plms_unified.py $qe $sim1 $sim1
