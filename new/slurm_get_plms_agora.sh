#!/bin/bash
#SBATCH --job-name=get_plms_agora
#SBATCH --time=3:00:00
#SBATCH --array=1-9
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --partition=kipac

#ests=(TT TE ET EE TB BT EB BE)
#qe=${ests[$SLURM_ARRAY_TASK_ID%8-1]}

ests=(T1T2 T2T1 TE ET EE TB BT EB BE)
qe=${ests[$SLURM_ARRAY_TASK_ID%9-1]}

export OMP_NUM_THREADS=12

#~/gmv/new/get_plms_agora.py $qe agora_standard_cinv test_yuka.yaml
#~/gmv/new/get_plms_agora.py $qe agora_standard_cinv test_yuka_lmaxT3500.yaml
#~/gmv/new/get_plms_agora.py $qe agora_standard_cinv test_yuka_lmaxT4000.yaml

~/gmv/new/get_plms_agora.py $qe agora_mh_cinv test_yuka_lmaxT3500.yaml noT3
~/gmv/new/get_plms_agora.py $qe agora_mh_cinv test_yuka_lmaxT4000.yaml noT3

~/gmv/new/get_plms_agora.py $qe agora_crossilc_twoseds_cinv test_yuka_lmaxT3500.yaml noT3
~/gmv/new/get_plms_agora.py $qe agora_crossilc_twoseds_cinv test_yuka_lmaxT4000.yaml noT3

~/gmv/new/get_plms_agora.py $qe agora_crossilc_onesed_cinv test_yuka_lmaxT3500.yaml noT3
~/gmv/new/get_plms_agora.py $qe agora_crossilc_onesed_cinv test_yuka_lmaxT4000.yaml noT3
