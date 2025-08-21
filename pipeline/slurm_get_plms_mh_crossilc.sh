#!/bin/bash
#SBATCH --job-name=get_plms_mh_crossilc
#SBATCH --time=15:00:00
#SBATCH --array=1-250
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --partition=kipac

sim1=$SLURM_ARRAY_TASK_ID
sim2=$(($sim1+1))

#ests=(T1T2 T2T1 TE T2E1 ET E2T1 EE E2E1 TB T2B1 BT B2T1 EB E2B1 BE B2E1)
#qe=${ests[$SLURM_ARRAY_TASK_ID%16-1]}
#sim1=$(((SLURM_ARRAY_TASK_ID-1)/16+1))
#sim2=$(($sim1+1))

export OMP_NUM_THREADS=12

# Ran below just to get the total Cls to average, for each lmaxT and each foreground mitigation method
# Then used average_totalcls.py to average
#~/gmv/pipeline/get_plms_mh_crossilc.py T1T2 $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID mh_cinv test_yuka.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py T1T2 $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID mh_cinv test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py T1T2 $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID mh_cinv test_yuka_lmaxT4000.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py T1T2 $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID crossilc_onesed_cinv test_yuka.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py T1T2 $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID crossilc_onesed_cinv test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py T1T2 $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID crossilc_onesed_cinv test_yuka_lmaxT4000.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py T1T2 $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID mh_cinv test_yuka_lmaxT3500.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py T1T2 $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID mh_cinv test_yuka_lmaxT4000.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py T1T2 $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID crossilc_onesed_cinv test_yuka_lmaxT3500.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py T1T2 $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID crossilc_onesed_cinv test_yuka_lmaxT4000.yaml websky noT3

# AGORA
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim1 mh_cinv test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim2 mh_cinv test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim2 $sim1 mh_cinv test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim1 mh_cmbonly_phi1_tqu1tqu2_cinv test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim1 mh_cmbonly_phi1_tqu2tqu1_cinv test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim2 mh_cmbonly_cinv test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim2 $sim1 mh_cmbonly_cinv test_yuka_lmaxT3500.yaml agora noT3
#
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim1 mh_cinv test_yuka_lmaxT4000.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim2 mh_cinv test_yuka_lmaxT4000.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim2 $sim1 mh_cinv test_yuka_lmaxT4000.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim1 mh_cmbonly_phi1_tqu1tqu2_cinv test_yuka_lmaxT4000.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim1 mh_cmbonly_phi1_tqu2tqu1_cinv test_yuka_lmaxT4000.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim2 mh_cmbonly_cinv test_yuka_lmaxT4000.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim2 $sim1 mh_cmbonly_cinv test_yuka_lmaxT4000.yaml agora noT3
#
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim1 crossilc_onesed_cinv test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim2 crossilc_onesed_cinv test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim2 $sim1 crossilc_onesed_cinv test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim1 crossilc_onesed_cmbonly_phi1_tqu1tqu2_cinv test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim1 crossilc_onesed_cmbonly_phi1_tqu2tqu1_cinv test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim2 crossilc_onesed_cmbonly_cinv test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim2 $sim1 crossilc_onesed_cmbonly_cinv test_yuka_lmaxT3500.yaml agora noT3
#
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim1 crossilc_onesed_cinv test_yuka_lmaxT4000.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim2 crossilc_onesed_cinv test_yuka_lmaxT4000.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim2 $sim1 crossilc_onesed_cinv test_yuka_lmaxT4000.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim1 crossilc_onesed_cmbonly_phi1_tqu1tqu2_cinv test_yuka_lmaxT4000.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim1 crossilc_onesed_cmbonly_phi1_tqu2tqu1_cinv test_yuka_lmaxT4000.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim2 crossilc_onesed_cmbonly_cinv test_yuka_lmaxT4000.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim2 $sim1 crossilc_onesed_cmbonly_cinv test_yuka_lmaxT4000.yaml agora noT3

# AGORA SQE
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim1 mh test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim2 mh test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim2 $sim1 mh test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim1 mh_cmbonly_phi1_tqu1tqu2 test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim1 mh_cmbonly_phi1_tqu2tqu1 test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim2 mh_cmbonly test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim2 $sim1 mh_cmbonly test_yuka_lmaxT3500.yaml agora noT3
#
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim1 crossilc_onesed test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim2 crossilc_onesed test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim2 $sim1 crossilc_onesed test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim1 crossilc_onesed_cmbonly_phi1_tqu1tqu2 test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim1 crossilc_onesed_cmbonly_phi1_tqu2tqu1 test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim2 crossilc_onesed_cmbonly test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim2 $sim1 crossilc_onesed_cmbonly test_yuka_lmaxT3500.yaml agora noT3

# AGORA GMV ALT
~/gmv/pipeline/get_plms_mh_crossilc.py all $sim1 $sim1 mh test_yuka_lmaxT3500.yaml agora noT3
~/gmv/pipeline/get_plms_mh_crossilc.py all $sim1 $sim2 mh test_yuka_lmaxT3500.yaml agora noT3
~/gmv/pipeline/get_plms_mh_crossilc.py all $sim2 $sim1 mh test_yuka_lmaxT3500.yaml agora noT3
~/gmv/pipeline/get_plms_mh_crossilc.py all $sim1 $sim1 mh_cmbonly_phi1_tqu1tqu2 test_yuka_lmaxT3500.yaml agora noT3
~/gmv/pipeline/get_plms_mh_crossilc.py all $sim1 $sim1 mh_cmbonly_phi1_tqu2tqu1 test_yuka_lmaxT3500.yaml agora noT3
~/gmv/pipeline/get_plms_mh_crossilc.py all $sim1 $sim2 mh_cmbonly test_yuka_lmaxT3500.yaml agora noT3
~/gmv/pipeline/get_plms_mh_crossilc.py all $sim2 $sim1 mh_cmbonly test_yuka_lmaxT3500.yaml agora noT3

# WEBSKY
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim1 mh_cinv test_yuka_lmaxT3500.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim2 mh_cinv test_yuka_lmaxT3500.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim2 $sim1 mh_cinv test_yuka_lmaxT3500.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim1 mh_cmbonly_phi1_tqu1tqu2_cinv test_yuka_lmaxT3500.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim1 mh_cmbonly_phi1_tqu2tqu1_cinv test_yuka_lmaxT3500.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim2 mh_cmbonly_cinv test_yuka_lmaxT3500.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim2 $sim1 mh_cmbonly_cinv test_yuka_lmaxT3500.yaml websky noT3
#
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim1 mh_cinv test_yuka_lmaxT4000.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim2 mh_cinv test_yuka_lmaxT4000.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim2 $sim1 mh_cinv test_yuka_lmaxT4000.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim1 mh_cmbonly_phi1_tqu1tqu2_cinv test_yuka_lmaxT4000.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim1 mh_cmbonly_phi1_tqu2tqu1_cinv test_yuka_lmaxT4000.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim2 mh_cmbonly_cinv test_yuka_lmaxT4000.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim2 $sim1 mh_cmbonly_cinv test_yuka_lmaxT4000.yaml websky noT3
#
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim1 crossilc_onesed_cinv test_yuka_lmaxT3500.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim2 crossilc_onesed_cinv test_yuka_lmaxT3500.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim2 $sim1 crossilc_onesed_cinv test_yuka_lmaxT3500.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim1 crossilc_onesed_cmbonly_phi1_tqu1tqu2_cinv test_yuka_lmaxT3500.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim1 crossilc_onesed_cmbonly_phi1_tqu2tqu1_cinv test_yuka_lmaxT3500.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim2 crossilc_onesed_cmbonly_cinv test_yuka_lmaxT3500.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim2 $sim1 crossilc_onesed_cmbonly_cinv test_yuka_lmaxT3500.yaml websky noT3
#
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim1 crossilc_onesed_cinv test_yuka_lmaxT4000.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim2 crossilc_onesed_cinv test_yuka_lmaxT4000.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim2 $sim1 crossilc_onesed_cinv test_yuka_lmaxT4000.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim1 crossilc_onesed_cmbonly_phi1_tqu1tqu2_cinv test_yuka_lmaxT4000.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim1 crossilc_onesed_cmbonly_phi1_tqu2tqu1_cinv test_yuka_lmaxT4000.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 $sim2 crossilc_onesed_cmbonly_cinv test_yuka_lmaxT4000.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim2 $sim1 crossilc_onesed_cmbonly_cinv test_yuka_lmaxT4000.yaml websky noT3

# For RDN0
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe r $sim1 mh_cinv test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 r mh_cinv test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe r $sim1 mh_cinv test_yuka_lmaxT4000.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 r mh_cinv test_yuka_lmaxT4000.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe r $sim1 crossilc_onesed_cinv test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 r crossilc_onesed_cinv test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe r $sim1 crossilc_onesed_cinv test_yuka_lmaxT4000.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 r crossilc_onesed_cinv test_yuka_lmaxT4000.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe r $sim1 crossilc_twoseds_cinv test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 r crossilc_twoseds_cinv test_yuka_lmaxT3500.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe r $sim1 crossilc_twoseds_cinv test_yuka_lmaxT4000.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 r crossilc_twoseds_cinv test_yuka_lmaxT4000.yaml agora noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe r $sim1 mh_cinv test_yuka_lmaxT3500.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 r mh_cinv test_yuka_lmaxT3500.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe r $sim1 mh_cinv test_yuka_lmaxT4000.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 r mh_cinv test_yuka_lmaxT4000.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe r $sim1 crossilc_onesed_cinv test_yuka_lmaxT3500.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 r crossilc_onesed_cinv test_yuka_lmaxT3500.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe r $sim1 crossilc_onesed_cinv test_yuka_lmaxT4000.yaml websky noT3
#~/gmv/pipeline/get_plms_mh_crossilc.py $qe $sim1 r crossilc_onesed_cinv test_yuka_lmaxT4000.yaml websky noT3

