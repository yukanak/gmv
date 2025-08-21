#!/bin/bash
#SBATCH --job-name=analyze
#SBATCH --time=20:00:00
#SBATCH --array=1-1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition=kipac

export OMP_NUM_THREADS=12

#~/gmv/pipeline/analyze_standard.py
~/gmv/pipeline/analyze_mh_crossilc.py
#~/gmv/pipeline/analyze_agora.py
#~/gmv/pipeline/compare_n0.py
#~/gmv/pipeline/test_n0_new_vs_old.py
