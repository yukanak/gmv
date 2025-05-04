#!/bin/bash
#SBATCH --job-name=get_plms_agora
#SBATCH --time=0:30:00
#SBATCH --array=1-12
#SBATCH --cpus-per-task=12
#SBATCH --mem=50G
#SBATCH --partition=kipac

#ests=(TT TE ET EE TB BT EB BE)
#qe=${ests[$SLURM_ARRAY_TASK_ID%8-1]}

ests=(T1T2 T2T1 TE T2E1 ET E2T1 EE E2E1 TB BT EB BE)
qe=${ests[$SLURM_ARRAY_TASK_ID%12-1]}

export OMP_NUM_THREADS=12

#~/gmv/pipeline/get_plms_agora.py $qe agora_standard_cinv test_yuka.yaml
#~/gmv/pipeline/get_plms_agora.py $qe agora_standard_cinv test_yuka_lmaxT3500.yaml
#~/gmv/pipeline/get_plms_agora.py $qe agora_standard_cinv test_yuka_lmaxT4000.yaml

#~/gmv/pipeline/get_plms_agora.py $qe agora_mh_cinv test_yuka_lmaxT3500.yaml noT3
#~/gmv/pipeline/get_plms_agora.py $qe agora_mh_cinv test_yuka_lmaxT4000.yaml noT3
#
#~/gmv/pipeline/get_plms_agora.py $qe agora_crossilc_twoseds_cinv test_yuka_lmaxT3500.yaml noT3
#~/gmv/pipeline/get_plms_agora.py $qe agora_crossilc_twoseds_cinv test_yuka_lmaxT4000.yaml noT3
#
#~/gmv/pipeline/get_plms_agora.py $qe agora_crossilc_onesed_cinv test_yuka_lmaxT3500.yaml noT3
#~/gmv/pipeline/get_plms_agora.py $qe agora_crossilc_onesed_cinv test_yuka_lmaxT4000.yaml noT3

#~/gmv/pipeline/get_plms_agora.py all agora_standard test_yuka_lmaxT3500.yaml
#~/gmv/pipeline/get_plms_agora.py all agora_mh test_yuka_lmaxT3500.yaml noT3
#~/gmv/pipeline/get_plms_agora.py all agora_crossilc_twoseds test_yuka_lmaxT3500.yaml noT3

#~/gmv/pipeline/get_plms_agora_12ests.py $qe agora_standard test_yuka.yaml
#~/gmv/pipeline/get_plms_agora_12ests.py $qe agora_standard test_yuka_lmaxT3500.yaml

# WEBSKY
#~/gmv/pipeline/get_plms_websky_12ests.py $qe websky_standard_cinv test_yuka.yaml
#~/gmv/pipeline/get_plms_websky_12ests.py $qe websky_standard_cinv test_yuka_lmaxT3500.yaml
#~/gmv/pipeline/get_plms_websky_12ests.py $qe websky_standard_cinv test_yuka_lmaxT4000.yaml
#~/gmv/pipeline/get_plms_websky_12ests.py $qe websky_standard test_yuka.yaml
#~/gmv/pipeline/get_plms_websky_12ests.py $qe websky_standard test_yuka_lmaxT3500.yaml
~/gmv/pipeline/get_plms_websky_12ests.py $qe websky_mh_cinv test_yuka_lmaxT3500.yaml noT3
~/gmv/pipeline/get_plms_websky_12ests.py $qe websky_mh_cinv test_yuka_lmaxT4000.yaml noT3
~/gmv/pipeline/get_plms_websky_12ests.py $qe websky_crossilc_onesed_cinv test_yuka_lmaxT3500.yaml noT3
~/gmv/pipeline/get_plms_websky_12ests.py $qe websky_crossilc_onesed_cinv test_yuka_lmaxT4000.yaml noT3

