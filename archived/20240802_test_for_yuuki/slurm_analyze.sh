#!/bin/bash
#SBATCH --job-name=analyze
#SBATCH --time=4:00:00
#SBATCH --array=1-1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition=kipac

export OMP_NUM_THREADS=12

~/gmv/20240802_test_for_yuuki/analyze_yuuki.py
