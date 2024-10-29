#!/bin/bash
#SBATCH --job-name=temp
#SBATCH --time=1:00:00
#SBATCH --array=1-1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition=kipac

export OMP_NUM_THREADS=12

#python3 /scratch/users/yomori/lensingtest/test.py TT 1 1 1 2
#python3 ~/gmv/tests/compare_n0.py
#python3 ~/gmv/tests/plot_gmv_12ests_vs_9ests.py
python3 ~/gmv/tests/plot_debug_agora_alt.py
#python3 ~/gmv/tests/check_ptsrc_mask.py
#python3 get_plms_standard_cinv_rdn0.py TT r 1 standard test_yuka.yaml
