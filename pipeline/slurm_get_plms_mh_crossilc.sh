#!/bin/bash
#SBATCH --job-name=get_plms_mh_crossilc
#SBATCH --time=6:00:00
#SBATCH --array=2001-3000
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --partition=kipac

#sim1=$SLURM_ARRAY_TASK_ID
#sim2=$(($sim1+1))

ests=(T1T2 T2T1 TE T2E1 ET E2T1 EE E2E1 TB BT EB BE)
qe=${ests[$SLURM_ARRAY_TASK_ID%12-1]}
sim1=$(((SLURM_ARRAY_TASK_ID-1)/12+1))
sim2=$(($sim1+1))

export OMP_NUM_THREADS=12

# Ran below just to get the total Cls to average, for each lmaxT and each foreground mitigation method
# Then used average_totalcls.py to average
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py T1T2 $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID mh_cinv test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py T1T2 $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID mh_cinv test_yuka_lmaxT4000.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py T1T2 $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID crossilc_onesed_cinv test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py T1T2 $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID crossilc_onesed_cinv test_yuka_lmaxT4000.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py T1T2 $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID crossilc_twoseds_cinv test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py T1T2 $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID crossilc_twoseds_cinv test_yuka_lmaxT4000.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py T1T2 $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID mh_cinv test_yuka_lmaxT3500.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py T1T2 $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID mh_cinv test_yuka_lmaxT4000.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py T1T2 $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID crossilc_onesed_cinv test_yuka_lmaxT3500.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py T1T2 $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID crossilc_onesed_cinv test_yuka_lmaxT4000.yaml websky noT3

#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim1 mh_cinv test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim2 mh_cinv test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim2 $sim1 mh_cinv test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim1 mh_cmbonly_phi1_tqu1tqu2_cinv test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim1 mh_cmbonly_phi1_tqu2tqu1_cinv test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim2 mh_cmbonly_cinv test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim2 $sim1 mh_cmbonly_cinv test_yuka_lmaxT3500.yaml agora noT3
#
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim1 mh_cinv test_yuka_lmaxT4000.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim2 mh_cinv test_yuka_lmaxT4000.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim2 $sim1 mh_cinv test_yuka_lmaxT4000.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim1 mh_cmbonly_phi1_tqu1tqu2_cinv test_yuka_lmaxT4000.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim1 mh_cmbonly_phi1_tqu2tqu1_cinv test_yuka_lmaxT4000.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim2 mh_cmbonly_cinv test_yuka_lmaxT4000.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim2 $sim1 mh_cmbonly_cinv test_yuka_lmaxT4000.yaml agora noT3
#
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim1 crossilc_onesed_cinv test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim2 crossilc_onesed_cinv test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim2 $sim1 crossilc_onesed_cinv test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim1 crossilc_onesed_cmbonly_phi1_tqu1tqu2_cinv test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim1 crossilc_onesed_cmbonly_phi1_tqu2tqu1_cinv test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim2 crossilc_onesed_cmbonly_cinv test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim2 $sim1 crossilc_onesed_cmbonly_cinv test_yuka_lmaxT3500.yaml agora noT3
#
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim1 crossilc_onesed_cinv test_yuka_lmaxT4000.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim2 crossilc_onesed_cinv test_yuka_lmaxT4000.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim2 $sim1 crossilc_onesed_cinv test_yuka_lmaxT4000.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim1 crossilc_onesed_cmbonly_phi1_tqu1tqu2_cinv test_yuka_lmaxT4000.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim1 crossilc_onesed_cmbonly_phi1_tqu2tqu1_cinv test_yuka_lmaxT4000.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim2 crossilc_onesed_cmbonly_cinv test_yuka_lmaxT4000.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim2 $sim1 crossilc_onesed_cmbonly_cinv test_yuka_lmaxT4000.yaml agora noT3
#
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim1 crossilc_twoseds_cinv test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim2 crossilc_twoseds_cinv test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim2 $sim1 crossilc_twoseds_cinv test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim1 crossilc_twoseds_cmbonly_phi1_tqu1tqu2_cinv test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim1 crossilc_twoseds_cmbonly_phi1_tqu2tqu1_cinv test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim2 crossilc_twoseds_cmbonly_cinv test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim2 $sim1 crossilc_twoseds_cmbonly_cinv test_yuka_lmaxT3500.yaml agora noT3
#
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim1 crossilc_twoseds_cinv test_yuka_lmaxT4000.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim2 crossilc_twoseds_cinv test_yuka_lmaxT4000.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim2 $sim1 crossilc_twoseds_cinv test_yuka_lmaxT4000.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim1 crossilc_twoseds_cmbonly_phi1_tqu1tqu2_cinv test_yuka_lmaxT4000.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim1 crossilc_twoseds_cmbonly_phi1_tqu2tqu1_cinv test_yuka_lmaxT4000.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim2 crossilc_twoseds_cmbonly_cinv test_yuka_lmaxT4000.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim2 $sim1 crossilc_twoseds_cmbonly_cinv test_yuka_lmaxT4000.yaml agora noT3
#
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim1 mh_cinv test_yuka_lmaxT3500.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim2 mh_cinv test_yuka_lmaxT3500.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim2 $sim1 mh_cinv test_yuka_lmaxT3500.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim1 mh_cmbonly_phi1_tqu1tqu2_cinv test_yuka_lmaxT3500.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim1 mh_cmbonly_phi1_tqu2tqu1_cinv test_yuka_lmaxT3500.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim2 mh_cmbonly_cinv test_yuka_lmaxT3500.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim2 $sim1 mh_cmbonly_cinv test_yuka_lmaxT3500.yaml websky noT3
#
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim1 mh_cinv test_yuka_lmaxT4000.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim2 mh_cinv test_yuka_lmaxT4000.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim2 $sim1 mh_cinv test_yuka_lmaxT4000.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim1 mh_cmbonly_phi1_tqu1tqu2_cinv test_yuka_lmaxT4000.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim1 mh_cmbonly_phi1_tqu2tqu1_cinv test_yuka_lmaxT4000.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim2 mh_cmbonly_cinv test_yuka_lmaxT4000.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim2 $sim1 mh_cmbonly_cinv test_yuka_lmaxT4000.yaml websky noT3
#
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim1 crossilc_onesed_cinv test_yuka_lmaxT3500.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim2 crossilc_onesed_cinv test_yuka_lmaxT3500.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim2 $sim1 crossilc_onesed_cinv test_yuka_lmaxT3500.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim1 crossilc_onesed_cmbonly_phi1_tqu1tqu2_cinv test_yuka_lmaxT3500.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim1 crossilc_onesed_cmbonly_phi1_tqu2tqu1_cinv test_yuka_lmaxT3500.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim2 crossilc_onesed_cmbonly_cinv test_yuka_lmaxT3500.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim2 $sim1 crossilc_onesed_cmbonly_cinv test_yuka_lmaxT3500.yaml websky noT3
#
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim1 crossilc_onesed_cinv test_yuka_lmaxT4000.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim2 crossilc_onesed_cinv test_yuka_lmaxT4000.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim2 $sim1 crossilc_onesed_cinv test_yuka_lmaxT4000.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim1 crossilc_onesed_cmbonly_phi1_tqu1tqu2_cinv test_yuka_lmaxT4000.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim1 crossilc_onesed_cmbonly_phi1_tqu2tqu1_cinv test_yuka_lmaxT4000.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 $sim2 crossilc_onesed_cmbonly_cinv test_yuka_lmaxT4000.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim2 $sim1 crossilc_onesed_cmbonly_cinv test_yuka_lmaxT4000.yaml websky noT3

# For RDN0
~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe r $sim1 mh_cinv test_yuka_lmaxT3500.yaml agora noT3
~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 r mh_cinv test_yuka_lmaxT3500.yaml agora noT3
~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe r $sim1 mh_cinv test_yuka_lmaxT4000.yaml agora noT3
~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 r mh_cinv test_yuka_lmaxT4000.yaml agora noT3
~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe r $sim1 crossilc_onesed_cinv test_yuka_lmaxT3500.yaml agora noT3
~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 r crossilc_onesed_cinv test_yuka_lmaxT3500.yaml agora noT3
~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe r $sim1 crossilc_onesed_cinv test_yuka_lmaxT4000.yaml agora noT3
~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 r crossilc_onesed_cinv test_yuka_lmaxT4000.yaml agora noT3
~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe r $sim1 crossilc_twoseds_cinv test_yuka_lmaxT3500.yaml agora noT3
~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 r crossilc_twoseds_cinv test_yuka_lmaxT3500.yaml agora noT3
~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe r $sim1 crossilc_twoseds_cinv test_yuka_lmaxT4000.yaml agora noT3
~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 r crossilc_twoseds_cinv test_yuka_lmaxT4000.yaml agora noT3
~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe r $sim1 mh_cinv test_yuka_lmaxT3500.yaml websky noT3
~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 r mh_cinv test_yuka_lmaxT3500.yaml websky noT3
~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe r $sim1 mh_cinv test_yuka_lmaxT4000.yaml websky noT3
~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 r mh_cinv test_yuka_lmaxT4000.yaml websky noT3
~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe r $sim1 crossilc_onesed_cinv test_yuka_lmaxT3500.yaml websky noT3
~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 r crossilc_onesed_cinv test_yuka_lmaxT3500.yaml websky noT3
~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe r $sim1 crossilc_onesed_cinv test_yuka_lmaxT4000.yaml websky noT3
~/gmv/pipeline/get_plms_mh_crossilc_12ests.py $qe $sim1 r crossilc_onesed_cinv test_yuka_lmaxT4000.yaml websky noT3

