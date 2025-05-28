#!/bin/bash
#SBATCH --job-name=random
#SBATCH --time=02:00:00
#SBATCH --array=1-1
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --partition=kipac

export OMP_NUM_THREADS=12

#python3 ~/gmv/pipeline/generate_gaussian_inputs.py
#python3 ~/gmv/pipeline/average_totalcls.py
#python3 ~/gmv/pipeline/test_n0_new_vs_old.py
#python3 ~/gmv/pipeline/check_agora_input_map.py
#python3 ~/gmv/pipeline/compare_n0.py
#python3 ~/gmv/pipeline/test_gaussian_inputs.py
#~/gmv/pipeline/get_plms_mh_crossilc_12ests.py BE 106 r crossilc_twoseds_cinv test_yuka_lmaxT3500.yaml agora noT3
#python3 /home/users/yukanaka/healqest/pipeline/spt3g_20192020/yuka_misc_scripts/check_cinv_maps.py
python3 ~/gmv/pipeline/convert_map2alm.py

