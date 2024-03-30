#!/bin/bash
#SBATCH --job-name=get_plms_agora
#SBATCH --time=1:00:00
#SBATCH --array=1-1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --partition=kipac

ests=(TT TE ET EE TB BT EB BE all TTEETE TBEB)
qe=${ests[$SLURM_ARRAY_TASK_ID%11-1]}
#ests=(TT TE ET EE TB BT EB BE TTprf all TTEETE TBEB TTEETEprf)
#qe=${ests[$SLURM_ARRAY_TASK_ID%13-1]}
#ests=(T1T2 T2T1 TE ET EE TB BT EB BE all TTEETE TBEB)
#qe=${ests[$SLURM_ARRAY_TASK_ID%12-1]}
#ests=(all TTEETE TBEB)
#qe=${ests[$SLURM_ARRAY_TASK_ID%3-1]}

export OMP_NUM_THREADS=12

~/gmv/tests/get_plms_agora.py TTEETEprf agora_profhrd test_yuka.yaml
#~/gmv/tests/get_plms_agora.py $qe agora_standard test_yuka_lmaxT4000.yaml
#~/gmv/tests/get_plms_agora.py $qe agora_standard test_yuka_lmaxT3500.yaml
#~/gmv/tests/get_plms_agora.py $qe agora_profhrd test_yuka_lmaxT4000.yaml
#~/gmv/tests/get_plms_agora.py $qe agora_profhrd test_yuka_lmaxT3500.yaml
#~/gmv/tests/get_plms_agora.py $qe agora_mh test_yuka_lmaxT4000.yaml
#~/gmv/tests/get_plms_agora.py $qe agora_crossilc_onesed test_yuka_lmaxT4000.yaml
#~/gmv/tests/get_plms_agora.py $qe agora_crossilc_twoseds test_yuka_lmaxT4000.yaml
