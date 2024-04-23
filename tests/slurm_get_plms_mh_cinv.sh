#!/bin/bash
#SBATCH --job-name=get_plms_mh_cinv
#SBATCH --time=7:00:00
#SBATCH --array=1-900
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --partition=kipac

ests=(T1T2 T2T1 TE ET EE TB BT EB BE)
qe=${ests[$SLURM_ARRAY_TASK_ID%9-1]}
sim1=$(((SLURM_ARRAY_TASK_ID-1)/9+1))
sim2=$(((SLURM_ARRAY_TASK_ID-1)/9+2))

export OMP_NUM_THREADS=12

#~/gmv/tests/get_plms_mh.py T1T2 $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID mh test_yuka_lmaxT3500.yaml
#~/gmv/tests/get_plms_mh.py T1T2 $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID mh test_yuka_lmaxT4000.yaml

~/gmv/tests/get_plms_mh_cinv.py $qe $sim1 $sim1 mh test_yuka.yaml
~/gmv/tests/get_plms_mh_cinv.py $qe $sim1 $sim2 mh test_yuka.yaml
~/gmv/tests/get_plms_mh_cinv.py $qe $sim2 $sim1 mh test_yuka.yaml
~/gmv/tests/get_plms_mh_cinv.py $qe $sim1 $sim1 mh_cmbonly_phi1_tqu1tqu2 test_yuka.yaml
~/gmv/tests/get_plms_mh_cinv.py $qe $sim1 $sim1 mh_cmbonly_phi1_tqu2tqu1 test_yuka.yaml
~/gmv/tests/get_plms_mh_cinv.py $qe $sim1 $sim2 mh_cmbonly test_yuka.yaml
~/gmv/tests/get_plms_mh_cinv.py $qe $sim2 $sim1 mh_cmbonly test_yuka.yaml
