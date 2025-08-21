#!/bin/bash
#SBATCH --job-name=get_plms_agora
#SBATCH --time=02:00:00
#SBATCH --array=1-16
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --partition=kipac

#ests=(TT TE ET EE TB BT EB BE)
#qe=${ests[$SLURM_ARRAY_TASK_ID%8-1]}

ests=(T1T2 T2T1 TE T2E1 ET E2T1 EE E2E1 TB T2B1 BT B2T1 EB E2B1 BE B2E1)
qe=${ests[$SLURM_ARRAY_TASK_ID%16-1]}

export OMP_NUM_THREADS=12

# STANDARD GMV
#~/gmv/pipeline/get_plms_agora.py $qe standard_cinv test_yuka.yaml agora
#~/gmv/pipeline/get_plms_agora.py $qe standard_cinv test_yuka_lmaxT3500.yaml agora
#~/gmv/pipeline/get_plms_agora.py $qe standard_cinv test_yuka_lmaxT4000.yaml agora

#~/gmv/pipeline/get_plms_agora.py $qe standard_cinv test_yuka.yaml websky
#~/gmv/pipeline/get_plms_agora.py $qe standard_cinv test_yuka_lmaxT3500.yaml websky
#~/gmv/pipeline/get_plms_agora.py $qe standard_cinv test_yuka_lmaxT4000.yaml websky

# STANDARD SQE
#~/gmv/pipeline/get_plms_agora.py $qe standard test_yuka.yaml agora
#~/gmv/pipeline/get_plms_agora.py $qe standard test_yuka_lmaxT3500.yaml agora
#~/gmv/pipeline/get_plms_agora.py $qe standard test_yuka_lmaxT4000.yaml agora

# MH
~/gmv/pipeline/get_plms_agora.py $qe mh_cinv test_yuka_lmaxT3500.yaml agora noT3
~/gmv/pipeline/get_plms_agora.py $qe mh_cinv test_yuka_lmaxT4000.yaml agora noT3
#~/gmv/pipeline/get_plms_agora.py $qe mh_cinv test_yuka_lmaxT3500.yaml websky noT3
#~/gmv/pipeline/get_plms_agora.py $qe mh_cinv test_yuka_lmaxT4000.yaml websky noT3

# CROSS-ILC
~/gmv/pipeline/get_plms_agora.py $qe crossilc_onesed_cinv test_yuka_lmaxT3500.yaml agora noT3
~/gmv/pipeline/get_plms_agora.py $qe crossilc_onesed_cinv test_yuka_lmaxT4000.yaml agora noT3
#~/gmv/pipeline/get_plms_agora.py $qe crossilc_onesed_cinv test_yuka_lmaxT3500.yaml websky noT3
#~/gmv/pipeline/get_plms_agora.py $qe crossilc_onesed_cinv test_yuka_lmaxT4000.yaml websky noT3

# MH SQE
~/gmv/pipeline/get_plms_agora.py $qe mh test_yuka_lmaxT3500.yaml agora noT3

# CROSS-ILC SQE
~/gmv/pipeline/get_plms_agora.py $qe crossilc_onesed test_yuka_lmaxT3500.yaml agora noT3
