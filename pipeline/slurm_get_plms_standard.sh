#!/bin/bash
#SBATCH --job-name=get_plms_standard
#SBATCH --time=1:00:00
#SBATCH --array=1-250
#SBATCH --cpus-per-task=12
#SBATCH --mem=50G
#SBATCH --partition=kipac

ests=(TT TE ET EE TB BT EB BE)
qe=${ests[$SLURM_ARRAY_TASK_ID%8-1]}
sim1=$(((SLURM_ARRAY_TASK_ID-1)/8+1))
sim2=$(($sim1+1))

#sim1=$SLURM_ARRAY_TASK_ID
#sim2=$(($sim1+1))

export OMP_NUM_THREADS=12

# If no averaged totalcls yet: just run what's commented out below to get the total Cls to average, for each lmaxT
# Once you have the total Cls for each input map, use average_totalcls.py to average
~/gmv/pipeline/get_plms_standard.py TT $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID standard_cinv test_yuka.yaml websky
~/gmv/pipeline/get_plms_standard.py TT $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID standard_cinv test_yuka_lmaxT3500.yaml websky
~/gmv/pipeline/get_plms_standard.py TT $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID standard_cinv test_yuka_lmaxT4000.yaml websky
#~/gmv/pipeline/get_plms_standard.py TT $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID standard_cinv test_yuka.yaml agora
#~/gmv/pipeline/get_plms_standard.py TT $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID standard_cinv test_yuka_lmaxT3500.yaml agora
#~/gmv/pipeline/get_plms_standard.py TT $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID standard_cinv test_yuka_lmaxT4000.yaml agora

# AGORA
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim1 standard_cinv test_yuka.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim2 standard_cinv test_yuka.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe $sim2 $sim1 standard_cinv test_yuka.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim1 standard_cmbonly_phi1_tqu1tqu2_cinv test_yuka.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim1 standard_cmbonly_phi1_tqu2tqu1_cinv test_yuka.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim2 standard_cmbonly_cinv test_yuka.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe $sim2 $sim1 standard_cmbonly_cinv test_yuka.yaml agora
#
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim1 standard_cinv test_yuka_lmaxT3500.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim2 standard_cinv test_yuka_lmaxT3500.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe $sim2 $sim1 standard_cinv test_yuka_lmaxT3500.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim1 standard_cmbonly_phi1_tqu1tqu2_cinv test_yuka_lmaxT3500.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim1 standard_cmbonly_phi1_tqu2tqu1_cinv test_yuka_lmaxT3500.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim2 standard_cmbonly_cinv test_yuka_lmaxT3500.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe $sim2 $sim1 standard_cmbonly_cinv test_yuka_lmaxT3500.yaml agora
#
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim1 standard_cinv test_yuka_lmaxT4000.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim2 standard_cinv test_yuka_lmaxT4000.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe $sim2 $sim1 standard_cinv test_yuka_lmaxT4000.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim1 standard_cmbonly_phi1_tqu1tqu2_cinv test_yuka_lmaxT4000.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim1 standard_cmbonly_phi1_tqu2tqu1_cinv test_yuka_lmaxT4000.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim2 standard_cmbonly_cinv test_yuka_lmaxT4000.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe $sim2 $sim1 standard_cmbonly_cinv test_yuka_lmaxT4000.yaml agora

# WEBSKY
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim1 standard_cinv test_yuka.yaml websky
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim2 standard_cinv test_yuka.yaml websky
#~/gmv/pipeline/get_plms_standard.py $qe $sim2 $sim1 standard_cinv test_yuka.yaml websky
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim1 standard_cmbonly_phi1_tqu1tqu2_cinv test_yuka.yaml websky
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim1 standard_cmbonly_phi1_tqu2tqu1_cinv test_yuka.yaml websky
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim2 standard_cmbonly_cinv test_yuka.yaml websky
#~/gmv/pipeline/get_plms_standard.py $qe $sim2 $sim1 standard_cmbonly_cinv test_yuka.yaml websky
#
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim1 standard_cinv test_yuka_lmaxT3500.yaml websky
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim2 standard_cinv test_yuka_lmaxT3500.yaml websky
#~/gmv/pipeline/get_plms_standard.py $qe $sim2 $sim1 standard_cinv test_yuka_lmaxT3500.yaml websky
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim1 standard_cmbonly_phi1_tqu1tqu2_cinv test_yuka_lmaxT3500.yaml websky
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim1 standard_cmbonly_phi1_tqu2tqu1_cinv test_yuka_lmaxT3500.yaml websky
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim2 standard_cmbonly_cinv test_yuka_lmaxT3500.yaml websky
#~/gmv/pipeline/get_plms_standard.py $qe $sim2 $sim1 standard_cmbonly_cinv test_yuka_lmaxT3500.yaml websky
#
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim1 standard_cinv test_yuka_lmaxT4000.yaml websky
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim2 standard_cinv test_yuka_lmaxT4000.yaml websky
#~/gmv/pipeline/get_plms_standard.py $qe $sim2 $sim1 standard_cinv test_yuka_lmaxT4000.yaml websky
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim1 standard_cmbonly_phi1_tqu1tqu2_cinv test_yuka_lmaxT4000.yaml websky
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim1 standard_cmbonly_phi1_tqu2tqu1_cinv test_yuka_lmaxT4000.yaml websky
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim2 standard_cmbonly_cinv test_yuka_lmaxT4000.yaml websky
#~/gmv/pipeline/get_plms_standard.py $qe $sim2 $sim1 standard_cmbonly_cinv test_yuka_lmaxT4000.yaml websky

# SQE
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim1 standard test_yuka.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim2 standard test_yuka.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe $sim2 $sim1 standard test_yuka.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim1 standard_cmbonly_phi1_tqu1tqu2 test_yuka.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim1 standard_cmbonly_phi1_tqu2tqu1 test_yuka.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim2 standard_cmbonly test_yuka.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe $sim2 $sim1 standard_cmbonly test_yuka.yaml agora
#
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim1 standard test_yuka_lmaxT3500.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim2 standard test_yuka_lmaxT3500.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe $sim2 $sim1 standard test_yuka_lmaxT3500.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim1 standard_cmbonly_phi1_tqu1tqu2 test_yuka_lmaxT3500.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim1 standard_cmbonly_phi1_tqu2tqu1 test_yuka_lmaxT3500.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim2 standard_cmbonly test_yuka_lmaxT3500.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe $sim2 $sim1 standard_cmbonly test_yuka_lmaxT3500.yaml agora
#
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim1 standard test_yuka_lmaxT4000.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim2 standard test_yuka_lmaxT4000.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe $sim2 $sim1 standard test_yuka_lmaxT4000.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim1 standard_cmbonly_phi1_tqu1tqu2 test_yuka_lmaxT4000.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim1 standard_cmbonly_phi1_tqu2tqu1 test_yuka_lmaxT4000.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim2 standard_cmbonly test_yuka_lmaxT4000.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe $sim2 $sim1 standard_cmbonly test_yuka_lmaxT4000.yaml agora

# GMV, NOT cinv
#~/gmv/pipeline/get_plms_standard.py all $sim1 $sim1 standard test_yuka_lmaxT3500.yaml agora
#~/gmv/pipeline/get_plms_standard.py all $sim1 $sim2 standard test_yuka_lmaxT3500.yaml agora
#~/gmv/pipeline/get_plms_standard.py all $sim2 $sim1 standard test_yuka_lmaxT3500.yaml agora
#~/gmv/pipeline/get_plms_standard.py all $sim1 $sim1 standard_cmbonly_phi1_tqu1tqu2 test_yuka_lmaxT3500.yaml agora
#~/gmv/pipeline/get_plms_standard.py all $sim1 $sim1 standard_cmbonly_phi1_tqu2tqu1 test_yuka_lmaxT3500.yaml agora
#~/gmv/pipeline/get_plms_standard.py all $sim1 $sim2 standard_cmbonly test_yuka_lmaxT3500.yaml agora
#~/gmv/pipeline/get_plms_standard.py all $sim2 $sim1 standard_cmbonly test_yuka_lmaxT3500.yaml agora

# For RDN0
#~/gmv/pipeline/get_plms_standard.py $qe r $sim1 standard_cinv test_yuka.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 r standard_cinv test_yuka.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe r $sim1 standard_cinv test_yuka_lmaxT3500.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 r standard_cinv test_yuka_lmaxT3500.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe r $sim1 standard_cinv test_yuka_lmaxT4000.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 r standard_cinv test_yuka_lmaxT4000.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe r $sim1 standard_cinv test_yuka.yaml websky
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 r standard_cinv test_yuka.yaml websky
#~/gmv/pipeline/get_plms_standard.py $qe r $sim1 standard_cinv test_yuka_lmaxT3500.yaml websky
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 r standard_cinv test_yuka_lmaxT3500.yaml websky
#~/gmv/pipeline/get_plms_standard.py $qe r $sim1 standard_cinv test_yuka_lmaxT4000.yaml websky
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 r standard_cinv test_yuka_lmaxT4000.yaml websky
#~/gmv/pipeline/get_plms_standard.py $qe r $sim1 standard test_yuka.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 r standard test_yuka.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe r $sim1 standard test_yuka_lmaxT3500.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 r standard test_yuka_lmaxT3500.yaml agora

## For FAKE RDN0
#~/gmv/pipeline/get_plms_standard.py $qe 1 $sim1 standard_cinv test_yuka_lmaxT3500.yaml agora
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 1 standard_cinv test_yuka_lmaxT3500.yaml agora

# For debug: CMB-only reconstruction for Websky
#~/gmv/pipeline/get_plms_standard.py $qe $sim1 $sim1 standard_cmbonly_cinv test_yuka_lmaxT3500.yaml websky

