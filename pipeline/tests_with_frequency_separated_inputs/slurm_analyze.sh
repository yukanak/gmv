#!/bin/bash
#SBATCH --job-name=analyze
#SBATCH --time=10:00:00
#SBATCH --array=1-1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition=kipac

export OMP_NUM_THREADS=12

#~/gmv/pipeline/analyze_standard.py
#~/gmv/pipeline/analyze_agora_12ests.py
~/gmv/pipeline/analyze_websky_12ests.py
#~/gmv/pipeline/compare_n0.py
#~/gmv/pipeline/analyze_mh_crossilc_12ests.py
