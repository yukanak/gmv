#!/bin/bash
#SBATCH --job-name=get_plms_standard
#SBATCH --time=1:00:00
#SBATCH --array=1-100
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --partition=kipac

ests=(TT TE ET EE TB BT EB BE all TTEETE TBEB)
qe=${ests[$SLURM_ARRAY_TASK_ID%11-1]}
sim1=$(((SLURM_ARRAY_TASK_ID-1)/11+1))
sim2=$(((SLURM_ARRAY_TASK_ID-1)/11+2))

export OMP_NUM_THREADS=12

~/gmv/tests/get_plms_standard.py TT $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID standard test_yuka_lmaxT3500.yaml
~/gmv/tests/get_plms_standard.py TT $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID standard test_yuka_lmaxT4000.yaml

#~/gmv/tests/get_plms_standard.py $qe $sim1 $sim1 standard
#~/gmv/tests/get_plms_standard.py $qe $sim1 $sim2 standard
#~/gmv/tests/get_plms_standard.py $qe $sim2 $sim1 standard
#~/gmv/tests/get_plms_standard.py $qe $sim1 $sim1 standard_cmbonly_phi1_tqu1tqu2
#~/gmv/tests/get_plms_standard.py $qe $sim1 $sim1 standard_cmbonly_phi1_tqu2tqu1
#~/gmv/tests/get_plms_standard.py $qe $sim1 $sim2 standard_cmbonly
#~/gmv/tests/get_plms_standard.py $qe $sim2 $sim1 standard_cmbonly
