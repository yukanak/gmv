#!/bin/bash
#SBATCH --job-name=get_plms_profhrd
#SBATCH --time=8:00:00
#SBATCH --array=1-1000
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --partition=kipac

ests=(TT TE ET EE TB BT EB BE TTprf all TTEETE TBEB TTEETEprf)
qe=${ests[$SLURM_ARRAY_TASK_ID%13-1]}
sim1=$(((SLURM_ARRAY_TASK_ID-1)/13+1))
sim2=$(((SLURM_ARRAY_TASK_ID-1)/13+2))

export OMP_NUM_THREADS=12

~/gmv/tests/get_plms_profhrd.py $qe $sim1 $sim1 profhrd
~/gmv/tests/get_plms_profhrd.py $qe $sim1 $sim2 profhrd
~/gmv/tests/get_plms_profhrd.py $qe $sim2 $sim1 profhrd
~/gmv/tests/get_plms_profhrd.py $qe $sim1 $sim1 profhrd_cmbonly_phi1_tqu1tqu2
~/gmv/tests/get_plms_profhrd.py $qe $sim1 $sim1 profhrd_cmbonly_phi1_tqu2tqu1
~/gmv/tests/get_plms_profhrd.py $qe $sim1 $sim2 profhrd_cmbonly
~/gmv/tests/get_plms_profhrd.py $qe $sim2 $sim1 profhrd_cmbonly
