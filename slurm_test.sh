#!/bin/bash
#SBATCH --job-name=test
#SBATCH --time=12:00:00
#SBATCH --array=1-1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --partition=kipac

~/gmv/get_plms_test.py EB 1 qest_original_unl
