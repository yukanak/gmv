#!/bin/bash
#SBATCH --job-name=get_plms_mh_crossilc
#SBATCH --time=4:00:00
#SBATCH --array=601-1600
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --partition=kipac

#ests=(T1T2 T2T1 TE ET EE TB BT EB BE)
#qe=${ests[$SLURM_ARRAY_TASK_ID%9-1]}
#sim1=$(((SLURM_ARRAY_TASK_ID-1)/9+1))
#sim2=$(($sim1+1))

#ests=(T1T2 T2T1 T2E1 E2T1 E2E1)
#qe=${ests[$SLURM_ARRAY_TASK_ID%5-1]}
#sim1=$(((SLURM_ARRAY_TASK_ID-1)/5+1))
#sim2=$(($sim1+1))

ests=(TE ET EE TB BT EB BE)
qe=${ests[$SLURM_ARRAY_TASK_ID%7-1]}
sim1=$(((SLURM_ARRAY_TASK_ID-1)/7+1))
sim2=$(($sim1+1))

export OMP_NUM_THREADS=12

# Ran below just to get the total Cls to average, for each lmaxT and each foreground mitigation method
# Then used average_totalcls.py to average
#~/gmv/new/get_plms_mh_crossilc.py T1T2 $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID mh_cinv test_yuka_lmaxT3500.yaml noT3
#~/gmv/new/get_plms_mh_crossilc.py T1T2 $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID mh_cinv test_yuka_lmaxT4000.yaml noT3
#~/gmv/new/get_plms_mh_crossilc.py T1T2 $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID crossilc_twoseds_cinv test_yuka_lmaxT3500.yaml noT3
#~/gmv/new/get_plms_mh_crossilc.py T1T2 $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID crossilc_twoseds_cinv test_yuka_lmaxT4000.yaml noT3
#~/gmv/new/get_plms_mh_crossilc.py T1T2 $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID crossilc_onesed_cinv test_yuka_lmaxT3500.yaml noT3
#~/gmv/new/get_plms_mh_crossilc.py T1T2 $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID crossilc_onesed_cinv test_yuka_lmaxT4000.yaml noT3

~/gmv/new/get_plms_mh_crossilc.py $qe r $sim1 mh_cinv test_yuka_lmaxT3500.yaml noT3
~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 r mh_cinv test_yuka_lmaxT3500.yaml noT3
~/gmv/new/get_plms_mh_crossilc.py $qe r $sim1 mh_cinv test_yuka_lmaxT4000.yaml noT3
~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 r mh_cinv test_yuka_lmaxT4000.yaml noT3

~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim1 mh_cinv test_yuka_lmaxT3500.yaml noT3
~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim2 mh_cinv test_yuka_lmaxT3500.yaml noT3
~/gmv/new/get_plms_mh_crossilc.py $qe $sim2 $sim1 mh_cinv test_yuka_lmaxT3500.yaml noT3
~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim1 mh_cmbonly_phi1_tqu1tqu2_cinv test_yuka_lmaxT3500.yaml noT3
~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim1 mh_cmbonly_phi1_tqu2tqu1_cinv test_yuka_lmaxT3500.yaml noT3
~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim2 mh_cmbonly_cinv test_yuka_lmaxT3500.yaml noT3
~/gmv/new/get_plms_mh_crossilc.py $qe $sim2 $sim1 mh_cmbonly_cinv test_yuka_lmaxT3500.yaml noT3

~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim1 mh_cinv test_yuka_lmaxT4000.yaml noT3
~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim2 mh_cinv test_yuka_lmaxT4000.yaml noT3
~/gmv/new/get_plms_mh_crossilc.py $qe $sim2 $sim1 mh_cinv test_yuka_lmaxT4000.yaml noT3
~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim1 mh_cmbonly_phi1_tqu1tqu2_cinv test_yuka_lmaxT4000.yaml noT3
~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim1 mh_cmbonly_phi1_tqu2tqu1_cinv test_yuka_lmaxT4000.yaml noT3
~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim2 mh_cmbonly_cinv test_yuka_lmaxT4000.yaml noT3
~/gmv/new/get_plms_mh_crossilc.py $qe $sim2 $sim1 mh_cmbonly_cinv test_yuka_lmaxT4000.yaml noT3

~/gmv/new/get_plms_mh_crossilc.py $qe r $sim1 crossilc_twoseds_cinv test_yuka_lmaxT3500.yaml noT3
~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 r crossilc_twoseds_cinv test_yuka_lmaxT3500.yaml noT3
~/gmv/new/get_plms_mh_crossilc.py $qe r $sim1 crossilc_twoseds_cinv test_yuka_lmaxT4000.yaml noT3
~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 r crossilc_twoseds_cinv test_yuka_lmaxT4000.yaml noT3

~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim1 crossilc_twoseds_cinv test_yuka_lmaxT3500.yaml noT3
~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim2 crossilc_twoseds_cinv test_yuka_lmaxT3500.yaml noT3
~/gmv/new/get_plms_mh_crossilc.py $qe $sim2 $sim1 crossilc_twoseds_cinv test_yuka_lmaxT3500.yaml noT3
~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim1 crossilc_twoseds_cmbonly_phi1_tqu1tqu2_cinv test_yuka_lmaxT3500.yaml noT3
~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim1 crossilc_twoseds_cmbonly_phi1_tqu2tqu1_cinv test_yuka_lmaxT3500.yaml noT3
~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim2 crossilc_twoseds_cmbonly_cinv test_yuka_lmaxT3500.yaml noT3
~/gmv/new/get_plms_mh_crossilc.py $qe $sim2 $sim1 crossilc_twoseds_cmbonly_cinv test_yuka_lmaxT3500.yaml noT3

~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim1 crossilc_twoseds_cinv test_yuka_lmaxT4000.yaml noT3
~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim2 crossilc_twoseds_cinv test_yuka_lmaxT4000.yaml noT3
~/gmv/new/get_plms_mh_crossilc.py $qe $sim2 $sim1 crossilc_twoseds_cinv test_yuka_lmaxT4000.yaml noT3
~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim1 crossilc_twoseds_cmbonly_phi1_tqu1tqu2_cinv test_yuka_lmaxT4000.yaml noT3
~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim1 crossilc_twoseds_cmbonly_phi1_tqu2tqu1_cinv test_yuka_lmaxT4000.yaml noT3
~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim2 crossilc_twoseds_cmbonly_cinv test_yuka_lmaxT4000.yaml noT3
~/gmv/new/get_plms_mh_crossilc.py $qe $sim2 $sim1 crossilc_twoseds_cmbonly_cinv test_yuka_lmaxT4000.yaml noT3
