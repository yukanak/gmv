#!/bin/bash
#SBATCH --job-name=get_plms_crossilc
#SBATCH --time=8:00:00
#SBATCH --array=1-1000
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --partition=kipac

ests=(T1T2 T2T1 TE ET EE TB BT EB BE all TTEETE TBEB)
qe=${ests[$SLURM_ARRAY_TASK_ID%12-1]}
sim1=$(((SLURM_ARRAY_TASK_ID-1)/12+1))
sim2=$(((SLURM_ARRAY_TASK_ID-1)/12+2))

#ests=(all TTEETE TBEB)
#qe=${ests[$SLURM_ARRAY_TASK_ID%3-1]}
#sim1=$(((SLURM_ARRAY_TASK_ID-1)/3+1))
#sim2=$(((SLURM_ARRAY_TASK_ID-1)/3+2))

export OMP_NUM_THREADS=12

#~/gmv/tests/get_plms_crossilc.py T1T2 $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID crossilc_onesed test_yuka_lmaxT3500.yaml
#~/gmv/tests/get_plms_crossilc.py T1T2 $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID crossilc_twoseds test_yuka_lmaxT3500.yaml
#~/gmv/tests/get_plms_crossilc.py T1T2 $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID crossilc_onesed test_yuka_lmaxT4000.yaml
#~/gmv/tests/get_plms_crossilc.py T1T2 $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID crossilc_twoseds test_yuka_lmaxT4000.yaml

~/gmv/tests/get_plms_crossilc.py $qe $sim1 $sim1 crossilc_onesed test_yuka_lmaxT3500.yaml
~/gmv/tests/get_plms_crossilc.py $qe $sim1 $sim2 crossilc_onesed test_yuka_lmaxT3500.yaml
~/gmv/tests/get_plms_crossilc.py $qe $sim2 $sim1 crossilc_onesed test_yuka_lmaxT3500.yaml
~/gmv/tests/get_plms_crossilc.py $qe $sim1 $sim1 crossilc_onesed_cmbonly_phi1_tqu1tqu2 test_yuka_lmaxT3500.yaml
~/gmv/tests/get_plms_crossilc.py $qe $sim1 $sim1 crossilc_onesed_cmbonly_phi1_tqu2tqu1 test_yuka_lmaxT3500.yaml
~/gmv/tests/get_plms_crossilc.py $qe $sim1 $sim2 crossilc_onesed_cmbonly test_yuka_lmaxT3500.yaml
~/gmv/tests/get_plms_crossilc.py $qe $sim2 $sim1 crossilc_onesed_cmbonly test_yuka_lmaxT3500.yaml
