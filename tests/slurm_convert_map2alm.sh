#!/bin/bash
#SBATCH --job-name=map2alm
#SBATCH --time=1:00:00
#SBATCH --array=1-1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition=kipac

export OMP_NUM_THREADS=12

~/gmv/tests/convert_map2alm.py
