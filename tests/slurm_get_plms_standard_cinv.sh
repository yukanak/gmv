#!/bin/bash
#SBATCH --job-name=get_plms_standard_cinv
#SBATCH --time=2:00:00
#SBATCH --array=1-800
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --partition=kipac

ests=(TT TE ET EE TB BT EB BE)
qe=${ests[$SLURM_ARRAY_TASK_ID%8-1]}
sim1=$(((SLURM_ARRAY_TASK_ID-1)/8+1))
sim2=$(((SLURM_ARRAY_TASK_ID-1)/8+2))

export OMP_NUM_THREADS=12

#~/gmv/tests/get_plms_standard_cinv.py $qe $sim1 $sim1 standard test_yuka.yaml
#~/gmv/tests/get_plms_standard_cinv.py $qe $sim1 $sim2 standard test_yuka.yaml
#~/gmv/tests/get_plms_standard_cinv.py $qe $sim2 $sim1 standard test_yuka.yaml
~/gmv/tests/get_plms_standard_cinv.py $qe $sim1 $sim1 standard_cmbonly_phi1_tqu1tqu2 test_yuka.yaml
~/gmv/tests/get_plms_standard_cinv.py $qe $sim1 $sim1 standard_cmbonly_phi1_tqu2tqu1 test_yuka.yaml
#~/gmv/tests/get_plms_standard_cinv.py $qe $sim1 $sim2 standard_cmbonly test_yuka.yaml
#~/gmv/tests/get_plms_standard_cinv.py $qe $sim2 $sim1 standard_cmbonly test_yuka.yaml
