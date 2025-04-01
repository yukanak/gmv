#!/bin/bash
#SBATCH --job-name=get_plms_agora
#SBATCH --time=5:00:00
#SBATCH --array=1-9
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --partition=kipac

ests=(all TT TE ET EE TB BT EB BE)
qe=${ests[$SLURM_ARRAY_TASK_ID%9-1]}

#ests=(all TTEETE TBEB)
#qe=${ests[$SLURM_ARRAY_TASK_ID%3-1]}

#ests=(TT TE ET EE TB BT EB BE)
#qe=${ests[$SLURM_ARRAY_TASK_ID%8-1]}

#ests=(T1T2 T2T1 TE ET EE TB BT EB BE)
#qe=${ests[$SLURM_ARRAY_TASK_ID%9-1]}

#ests=(all TTEETE TBEB TT TE ET EE TB BT EB BE)
#qe=${ests[$SLURM_ARRAY_TASK_ID%11-1]}

#ests=(all TTEETE TBEB T1T2 T2T1 TE ET EE TB BT EB BE)
#qe=${ests[$SLURM_ARRAY_TASK_ID%12-1]}

export OMP_NUM_THREADS=12

~/gmv/tests/get_plms_agora_nolrad.py $qe agora_standard test_yuka.yaml
#~/gmv/tests/get_plms_agora_nolrad.py $qe agora_standard test_yuka_lmaxT4000.yaml
#~/gmv/tests/get_plms_agora_nolrad.py $qe agora_standard test_yuka_lmaxT3500.yaml

#~/gmv/tests/get_plms_agora.py $qe agora_standard test_yuka.yaml
#~/gmv/tests/get_plms_agora.py $qe agora_standard test_yuka_lmaxT4000.yaml
#~/gmv/tests/get_plms_agora.py $qe agora_standard test_yuka_lmaxT3500.yaml
#~/gmv/tests/get_plms_agora.py $qe agora_profhrd test_yuka.yaml
#~/gmv/tests/get_plms_agora.py $qe agora_profhrd test_yuka_lmaxT4000.yaml
#~/gmv/tests/get_plms_agora.py $qe agora_profhrd test_yuka_lmaxT3500.yaml

#~/gmv/tests/get_plms_agora.py $qe agora_mh test_yuka.yaml
#~/gmv/tests/get_plms_agora.py $qe agora_crossilc_onesed test_yuka.yaml
#~/gmv/tests/get_plms_agora.py $qe agora_crossilc_twoseds test_yuka.yaml
#~/gmv/tests/get_plms_agora.py $qe agora_mh test_yuka_lmaxT3500.yaml
#~/gmv/tests/get_plms_agora.py $qe agora_crossilc_onesed test_yuka_lmaxT3500.yaml
#~/gmv/tests/get_plms_agora.py $qe agora_crossilc_twoseds test_yuka_lmaxT3500.yaml
#~/gmv/tests/get_plms_agora.py $qe agora_mh test_yuka_lmaxT4000.yaml
#~/gmv/tests/get_plms_agora.py $qe agora_crossilc_onesed test_yuka_lmaxT4000.yaml
#~/gmv/tests/get_plms_agora.py $qe agora_crossilc_twoseds test_yuka_lmaxT4000.yaml

#~/gmv/tests/get_plms_agora_cinv.py $qe agora_standard test_yuka.yaml
#~/gmv/tests/get_plms_agora_cinv.py $qe agora_standard test_yuka_lmaxT3500.yaml
#~/gmv/tests/get_plms_agora_cinv.py $qe agora_standard test_yuka_lmaxT4000.yaml

#~/gmv/tests/get_plms_agora_cinv.py $qe agora_mh test_yuka.yaml
#~/gmv/tests/get_plms_agora_cinv.py $qe agora_mh test_yuka_lmaxT3500.yaml
#~/gmv/tests/get_plms_agora_cinv.py $qe agora_mh test_yuka_lmaxT4000.yaml
#~/gmv/tests/get_plms_agora_cinv.py $qe agora_crossilc_onesed test_yuka.yaml
#~/gmv/tests/get_plms_agora_cinv.py $qe agora_crossilc_onesed test_yuka_lmaxT3500.yaml
#~/gmv/tests/get_plms_agora_cinv.py $qe agora_crossilc_onesed test_yuka_lmaxT4000.yaml
#~/gmv/tests/get_plms_agora_cinv.py $qe agora_crossilc_twoseds test_yuka.yaml
#~/gmv/tests/get_plms_agora_cinv.py $qe agora_crossilc_twoseds test_yuka_lmaxT3500.yaml
#~/gmv/tests/get_plms_agora_cinv.py $qe agora_crossilc_twoseds test_yuka_lmaxT4000.yaml

#~/gmv/tests/get_plms_agora_alt.py $qe agora_standard_rotatedcmb test_yuka_lmaxT3500.yaml
#~/gmv/tests/get_plms_agora_alt.py $qe agora_standard_gaussianfg test_yuka_lmaxT3500.yaml
#~/gmv/tests/get_plms_agora_alt.py $qe agora_standard_rotatedcmb test_yuka_lmaxT4000.yaml
#~/gmv/tests/get_plms_agora_alt.py $qe agora_standard_gaussianfg test_yuka_lmaxT4000.yaml
#~/gmv/tests/get_plms_agora_alt.py $qe agora_mh_rotatedcmb test_yuka_lmaxT3500.yaml
#~/gmv/tests/get_plms_agora_alt.py $qe agora_mh_gaussianfg test_yuka_lmaxT3500.yaml
#~/gmv/tests/get_plms_agora_alt.py $qe agora_mh_rotatedcmb test_yuka_lmaxT4000.yaml
#~/gmv/tests/get_plms_agora_alt.py $qe agora_mh_gaussianfg test_yuka_lmaxT4000.yaml
#~/gmv/tests/get_plms_agora_alt.py $qe agora_crossilc_twoseds_rotatedcmb test_yuka_lmaxT3500.yaml
#~/gmv/tests/get_plms_agora_alt.py $qe agora_crossilc_twoseds_gaussianfg test_yuka_lmaxT3500.yaml
#~/gmv/tests/get_plms_agora_alt.py $qe agora_crossilc_twoseds_rotatedcmb test_yuka_lmaxT4000.yaml
#~/gmv/tests/get_plms_agora_alt.py $qe agora_crossilc_twoseds_gaussianfg test_yuka_lmaxT4000.yaml
#~/gmv/tests/get_plms_agora_alt.py $qe agora_standard_rotatedcmb test_yuka.yaml
#~/gmv/tests/get_plms_agora_alt.py $qe agora_standard_gaussianfg test_yuka.yaml
#~/gmv/tests/get_plms_agora_alt.py $qe agora_standard_gaussiancmb test_yuka.yaml
#~/gmv/tests/get_plms_agora_alt.py $qe agora_standard_rotatedcmb_gaussianfg test_yuka.yaml
#~/gmv/tests/get_plms_agora_alt.py $qe agora_standard_rotatedgaussiancmb test_yuka.yaml
#~/gmv/tests/get_plms_agora_alt.py $qe agora_standard_separated test_yuka.yaml
