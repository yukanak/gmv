#!/bin/bash
#SBATCH --job-name=analyze
#SBATCH --time=20:00:00
#SBATCH --array=1-1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition=kipac

export OMP_NUM_THREADS=12

#~/gmv/new/analyze_standard.py
~/gmv/new/analyze_mh_crossilc.py
~/gmv/new/analyze_agora.py
