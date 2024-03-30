#!/bin/bash
#SBATCH --job-name=get_plms_mh
#SBATCH --time=7:00:00
#SBATCH --array=999-1200
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --partition=kipac

ests=(T1T2 T2T1 TE ET EE TB BT EB BE all TTEETE TBEB)
qe=${ests[$SLURM_ARRAY_TASK_ID%12-1]}
sim1=$(((SLURM_ARRAY_TASK_ID-1)/12+1))
sim2=$(((SLURM_ARRAY_TASK_ID-1)/12+2))

#ests=(all TTEETE TBEB)
#qe=${ests[$SLURM_ARRAY_TASK_ID%3-1]}
#sim1=$(((SLURM_ARRAY_TASK_ID-1)/3+1))
#sim2=$(((SLURM_ARRAY_TASK_ID-1)/3+2))

export OMP_NUM_THREADS=12

#~/gmv/tests/get_plms_mh.py T1T2 $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID mh test_yuka_lmaxT3500.yaml
#~/gmv/tests/get_plms_mh.py T1T2 $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID mh test_yuka_lmaxT4000.yaml

~/gmv/tests/get_plms_mh.py $qe $sim1 $sim1 mh test_yuka_lmaxT4000.yaml
~/gmv/tests/get_plms_mh.py $qe $sim1 $sim2 mh test_yuka_lmaxT4000.yaml
~/gmv/tests/get_plms_mh.py $qe $sim2 $sim1 mh test_yuka_lmaxT4000.yaml
~/gmv/tests/get_plms_mh.py $qe $sim1 $sim1 mh_cmbonly_phi1_tqu1tqu2 test_yuka_lmaxT4000.yaml
~/gmv/tests/get_plms_mh.py $qe $sim1 $sim1 mh_cmbonly_phi1_tqu2tqu1 test_yuka_lmaxT4000.yaml
~/gmv/tests/get_plms_mh.py $qe $sim1 $sim2 mh_cmbonly test_yuka_lmaxT4000.yaml
~/gmv/tests/get_plms_mh.py $qe $sim2 $sim1 mh_cmbonly test_yuka_lmaxT4000.yaml
