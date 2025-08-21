#!/bin/bash
#SBATCH --job-name=random
#SBATCH --time=05:00:00
#SBATCH --array=1-1
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --partition=kipac

export OMP_NUM_THREADS=12

#~/gmv/pipeline/get_plms_mh_crossilc.py BE 106 r crossilc_onesed_cinv test_yuka_lmaxT3500.yaml agora noT3
#python3 ~/gmv/pipeline/convert_map2alm.py
python3 ~/gmv/pipeline/generate_gaussian_inputs.py
#python3 ~/gmv/pipeline/average_totalcls.py
#python3 ~/gmv/pipeline/plot_ilc_weights.py
#python3 ~/gmv/pipeline/check_agora_input_map.py
#python3 ~/gmv/pipeline/compare_n0.py
#python3 ~/gmv/pipeline/test_gaussian_inputs_websky.py
#python3 /home/users/yukanaka/healqest/pipeline/spt3g_20192020/yuka_misc_scripts/check_cinv_maps.py
#python3 ~/gmv/so_noise_models/yuka_get_agora_fg_spec.py
#python3 /home/users/yukanaka/healqest/pipeline/spt3g_20192020/yuka_misc_scripts/pack_likelihood_products_yuuki.py
#python3 /home/users/yukanaka/healqest/pipeline/spt3g_20192020/yuka_misc_scripts/pack_likelihood_products_yuuki_SAN0.py

