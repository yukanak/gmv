#!/bin/bash
#SBATCH --job-name=get_plms_mh_crossilc
#SBATCH --time=2:30:00
#SBATCH --array=2001-3000
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --partition=kipac

#ests=(T1T2 T2T1 TE ET EE TB BT EB BE)
#qe=${ests[$SLURM_ARRAY_TASK_ID%9-1]}
##sim1=$(((SLURM_ARRAY_TASK_ID-1)/9+1))
#sim1=$(((SLURM_ARRAY_TASK_ID-1+1999)/9+1))
#sim2=$(($sim1+1))

#ests=(T1T2 T2T1 T2E1 E2T1 E2E1)
#qe=${ests[$SLURM_ARRAY_TASK_ID%5-1]}
#sim1=$(((SLURM_ARRAY_TASK_ID-1)/5+1))
#sim2=$(($sim1+1))

#sim1=$SLURM_ARRAY_TASK_ID
#sim2=$(($sim1+1))

ests=(T1T2 T2T1 TE T2E1 ET E2T1 EE E2E1 TB BT EB BE)
qe=${ests[$SLURM_ARRAY_TASK_ID%12-1]}
sim1=$(((SLURM_ARRAY_TASK_ID-1)/12+1))
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

#~/gmv/new/get_plms_mh_crossilc.py $qe r $sim1 mh_cinv test_yuka_lmaxT3500.yaml noT3
#~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 r mh_cinv test_yuka_lmaxT3500.yaml noT3
#~/gmv/new/get_plms_mh_crossilc.py $qe r $sim1 mh_cinv test_yuka_lmaxT4000.yaml noT3
#~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 r mh_cinv test_yuka_lmaxT4000.yaml noT3

#~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim1 mh_cinv test_yuka_lmaxT3500.yaml noT3
#~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim2 mh_cinv test_yuka_lmaxT3500.yaml noT3
#~/gmv/new/get_plms_mh_crossilc.py $qe $sim2 $sim1 mh_cinv test_yuka_lmaxT3500.yaml noT3
#~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim1 mh_cmbonly_phi1_tqu1tqu2_cinv test_yuka_lmaxT3500.yaml noT3
#~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim1 mh_cmbonly_phi1_tqu2tqu1_cinv test_yuka_lmaxT3500.yaml noT3
#~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim2 mh_cmbonly_cinv test_yuka_lmaxT3500.yaml noT3
#~/gmv/new/get_plms_mh_crossilc.py $qe $sim2 $sim1 mh_cmbonly_cinv test_yuka_lmaxT3500.yaml noT3

#~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim1 mh_cinv test_yuka_lmaxT4000.yaml noT3
#~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim2 mh_cinv test_yuka_lmaxT4000.yaml noT3
#~/gmv/new/get_plms_mh_crossilc.py $qe $sim2 $sim1 mh_cinv test_yuka_lmaxT4000.yaml noT3
#~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim1 mh_cmbonly_phi1_tqu1tqu2_cinv test_yuka_lmaxT4000.yaml noT3
#~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim1 mh_cmbonly_phi1_tqu2tqu1_cinv test_yuka_lmaxT4000.yaml noT3
#~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim2 mh_cmbonly_cinv test_yuka_lmaxT4000.yaml noT3
#~/gmv/new/get_plms_mh_crossilc.py $qe $sim2 $sim1 mh_cmbonly_cinv test_yuka_lmaxT4000.yaml noT3

#~/gmv/new/get_plms_mh_crossilc.py $qe r $sim1 crossilc_twoseds_cinv test_yuka_lmaxT3500.yaml noT3
#~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 r crossilc_twoseds_cinv test_yuka_lmaxT3500.yaml noT3
#~/gmv/new/get_plms_mh_crossilc.py $qe r $sim1 crossilc_twoseds_cinv test_yuka_lmaxT4000.yaml noT3
#~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 r crossilc_twoseds_cinv test_yuka_lmaxT4000.yaml noT3

#~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim1 crossilc_twoseds_cinv test_yuka_lmaxT3500.yaml noT3
#~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim2 crossilc_twoseds_cinv test_yuka_lmaxT3500.yaml noT3
#~/gmv/new/get_plms_mh_crossilc.py $qe $sim2 $sim1 crossilc_twoseds_cinv test_yuka_lmaxT3500.yaml noT3
#~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim1 crossilc_twoseds_cmbonly_phi1_tqu1tqu2_cinv test_yuka_lmaxT3500.yaml noT3
#~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim1 crossilc_twoseds_cmbonly_phi1_tqu2tqu1_cinv test_yuka_lmaxT3500.yaml noT3
#~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim2 crossilc_twoseds_cmbonly_cinv test_yuka_lmaxT3500.yaml noT3
#~/gmv/new/get_plms_mh_crossilc.py $qe $sim2 $sim1 crossilc_twoseds_cmbonly_cinv test_yuka_lmaxT3500.yaml noT3
#
#~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim1 crossilc_twoseds_cinv test_yuka_lmaxT4000.yaml noT3
#~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim2 crossilc_twoseds_cinv test_yuka_lmaxT4000.yaml noT3
#~/gmv/new/get_plms_mh_crossilc.py $qe $sim2 $sim1 crossilc_twoseds_cinv test_yuka_lmaxT4000.yaml noT3
#~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim1 crossilc_twoseds_cmbonly_phi1_tqu1tqu2_cinv test_yuka_lmaxT4000.yaml noT3
#~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim1 crossilc_twoseds_cmbonly_phi1_tqu2tqu1_cinv test_yuka_lmaxT4000.yaml noT3
#~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim2 crossilc_twoseds_cmbonly_cinv test_yuka_lmaxT4000.yaml noT3
#~/gmv/new/get_plms_mh_crossilc.py $qe $sim2 $sim1 crossilc_twoseds_cmbonly_cinv test_yuka_lmaxT4000.yaml noT3

# Below two blocks are for Eq. 45-49, 12 ests
#~/gmv/new/get_plms_mh_crossilc_12ests.py all r $sim1 mh test_yuka_lmaxT3500.yaml noT3
#~/gmv/new/get_plms_mh_crossilc_12ests.py all $sim1 r mh test_yuka_lmaxT3500.yaml noT3
#~/gmv/new/get_plms_mh_crossilc_12ests.py all $sim1 $sim1 mh test_yuka_lmaxT3500.yaml noT3
#~/gmv/new/get_plms_mh_crossilc_12ests.py all $sim1 $sim2 mh test_yuka_lmaxT3500.yaml noT3
#~/gmv/new/get_plms_mh_crossilc_12ests.py all $sim2 $sim1 mh test_yuka_lmaxT3500.yaml noT3
#~/gmv/new/get_plms_mh_crossilc_12ests.py all $sim1 $sim1 mh_cmbonly_phi1_tqu1tqu2 test_yuka_lmaxT3500.yaml noT3
#~/gmv/new/get_plms_mh_crossilc_12ests.py all $sim1 $sim1 mh_cmbonly_phi1_tqu2tqu1 test_yuka_lmaxT3500.yaml noT3
#~/gmv/new/get_plms_mh_crossilc_12ests.py all $sim1 $sim2 mh_cmbonly test_yuka_lmaxT3500.yaml noT3
#~/gmv/new/get_plms_mh_crossilc_12ests.py all $sim2 $sim1 mh_cmbonly test_yuka_lmaxT3500.yaml noT3

#~/gmv/new/get_plms_mh_crossilc_12ests.py all r $sim1 crossilc_twoseds test_yuka_lmaxT3500.yaml noT3
#~/gmv/new/get_plms_mh_crossilc_12ests.py all $sim1 r crossilc_twoseds test_yuka_lmaxT3500.yaml noT3
#~/gmv/new/get_plms_mh_crossilc_12ests.py all $sim1 $sim1 crossilc_twoseds test_yuka_lmaxT3500.yaml noT3
#~/gmv/new/get_plms_mh_crossilc_12ests.py all $sim1 $sim2 crossilc_twoseds test_yuka_lmaxT3500.yaml noT3
#~/gmv/new/get_plms_mh_crossilc_12ests.py all $sim2 $sim1 crossilc_twoseds test_yuka_lmaxT3500.yaml noT3
#~/gmv/new/get_plms_mh_crossilc_12ests.py all $sim1 $sim1 crossilc_twoseds_cmbonly_phi1_tqu1tqu2 test_yuka_lmaxT3500.yaml noT3
#~/gmv/new/get_plms_mh_crossilc_12ests.py all $sim1 $sim1 crossilc_twoseds_cmbonly_phi1_tqu2tqu1 test_yuka_lmaxT3500.yaml noT3
#~/gmv/new/get_plms_mh_crossilc_12ests.py all $sim1 $sim2 crossilc_twoseds_cmbonly test_yuka_lmaxT3500.yaml noT3
#~/gmv/new/get_plms_mh_crossilc_12ests.py all $sim2 $sim1 crossilc_twoseds_cmbonly test_yuka_lmaxT3500.yaml noT3

# Below two blocks for with T3
#~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim1 mh_cinv test_yuka_lmaxT3500.yaml withT3
#~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim2 mh_cinv test_yuka_lmaxT3500.yaml withT3
#~/gmv/new/get_plms_mh_crossilc.py $qe $sim2 $sim1 mh_cinv test_yuka_lmaxT3500.yaml withT3
#~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim1 mh_cmbonly_phi1_tqu1tqu2_cinv test_yuka_lmaxT3500.yaml withT3
#~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim1 mh_cmbonly_phi1_tqu2tqu1_cinv test_yuka_lmaxT3500.yaml withT3
#~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim2 mh_cmbonly_cinv test_yuka_lmaxT3500.yaml withT3
#~/gmv/new/get_plms_mh_crossilc.py $qe $sim2 $sim1 mh_cmbonly_cinv test_yuka_lmaxT3500.yaml withT3
#~/gmv/new/get_plms_mh_crossilc.py $qe r $sim1 mh_cinv test_yuka_lmaxT3500.yaml withT3
#~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 r mh_cinv test_yuka_lmaxT3500.yaml withT3
#
#~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim1 crossilc_twoseds_cinv test_yuka_lmaxT3500.yaml withT3
#~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim2 crossilc_twoseds_cinv test_yuka_lmaxT3500.yaml withT3
#~/gmv/new/get_plms_mh_crossilc.py $qe $sim2 $sim1 crossilc_twoseds_cinv test_yuka_lmaxT3500.yaml withT3
#~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim1 crossilc_twoseds_cmbonly_phi1_tqu1tqu2_cinv test_yuka_lmaxT3500.yaml withT3
#~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim1 crossilc_twoseds_cmbonly_phi1_tqu2tqu1_cinv test_yuka_lmaxT3500.yaml withT3
#~/gmv/new/get_plms_mh_crossilc.py $qe $sim1 $sim2 crossilc_twoseds_cmbonly_cinv test_yuka_lmaxT3500.yaml withT3
#~/gmv/new/get_plms_mh_crossilc.py $qe $sim2 $sim1 crossilc_twoseds_cmbonly_cinv test_yuka_lmaxT3500.yaml withT3
#~/gmv/new/get_plms_mh_crossilc.py $qe r $sim1 crossilc_twoseds_cinv test_yuka_lmaxT3500.yaml withT3

# WEBSKY
~/gmv/pipeline/get_plms_mh_crossilc_12ests_websky.py $qe r $sim1 mh_cinv test_yuka_lmaxT3500.yaml noT3
~/gmv/pipeline/get_plms_mh_crossilc_12ests_websky.py $qe $sim1 r mh_cinv test_yuka_lmaxT3500.yaml noT3
~/gmv/pipeline/get_plms_mh_crossilc_12ests_websky.py $qe r $sim1 mh_cinv test_yuka_lmaxT4000.yaml noT3
~/gmv/pipeline/get_plms_mh_crossilc_12ests_websky.py $qe $sim1 r mh_cinv test_yuka_lmaxT4000.yaml noT3
~/gmv/pipeline/get_plms_mh_crossilc_12ests_websky.py $qe r $sim1 crossilc_onesed_cinv test_yuka_lmaxT3500.yaml noT3
~/gmv/pipeline/get_plms_mh_crossilc_12ests_websky.py $qe $sim1 r crossilc_onesed_cinv test_yuka_lmaxT3500.yaml noT3
~/gmv/pipeline/get_plms_mh_crossilc_12ests_websky.py $qe r $sim1 crossilc_onesed_cinv test_yuka_lmaxT4000.yaml noT3
~/gmv/pipeline/get_plms_mh_crossilc_12ests_websky.py $qe $sim1 r crossilc_onesed_cinv test_yuka_lmaxT4000.yaml noT3
