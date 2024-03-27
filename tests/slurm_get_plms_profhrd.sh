#!/bin/bash
#SBATCH --job-name=get_plms_profhrd
#SBATCH --time=1:00:00
#SBATCH --array=1-1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --partition=kipac

ests=(TT TE ET EE TB BT EB BE TTprf all TTEETE TBEB TTEETEprf)
qe=${ests[$SLURM_ARRAY_TASK_ID%13-1]}
sim1=$(((SLURM_ARRAY_TASK_ID-1)/13+1))
sim2=$(((SLURM_ARRAY_TASK_ID-1)/13+2))

#ests=(TTprf TTEETEprf)
#qe=${ests[$SLURM_ARRAY_TASK_ID%2-1]}
#sim1=$(((SLURM_ARRAY_TASK_ID-1)/2+1))
#sim2=$(((SLURM_ARRAY_TASK_ID-1)/2+2))

export OMP_NUM_THREADS=12

#~/gmv/tests/get_plms_profhrd.py TT $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID profhrd
~/gmv/tests/get_plms_profhrd_flatgaussian.py TB 34 35 profhrd_flatgaussian

#~/gmv/tests/get_plms_profhrd_flatgaussian.py $qe $sim1 $sim1 profhrd_flatgaussian
#~/gmv/tests/get_plms_profhrd_flatgaussian.py $qe $sim1 $sim2 profhrd_flatgaussian
#~/gmv/tests/get_plms_profhrd_flatgaussian.py $qe $sim2 $sim1 profhrd_flatgaussian
#~/gmv/tests/get_plms_profhrd_flatgaussian.py $qe $sim1 $sim1 profhrd_flatgaussian_cmbonly_phi1_tqu1tqu2
#~/gmv/tests/get_plms_profhrd_flatgaussian.py $qe $sim1 $sim1 profhrd_flatgaussian_cmbonly_phi1_tqu2tqu1
#~/gmv/tests/get_plms_profhrd_flatgaussian.py $qe $sim1 $sim2 profhrd_flatgaussian_cmbonly
#~/gmv/tests/get_plms_profhrd_flatgaussian.py $qe $sim2 $sim1 profhrd_flatgaussian_cmbonly
