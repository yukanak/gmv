#!/bin/bash
#SBATCH --job-name=analyze
#SBATCH --time=8:00:00
#SBATCH --array=1-1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --partition=kipac

export OMP_NUM_THREADS=12

~/gmv/tests/analyze_standard_cinv.py
#~/gmv/tests/analyze_agora_nolrad.py
#~/gmv/tests/analyze_agora_alt.py
#~/gmv/tests/analyze_cinv_9ests_alt_n0.py
#~/gmv/tests/analyze_crossilc.py
#~/gmv/tests/analyze_agora.py
