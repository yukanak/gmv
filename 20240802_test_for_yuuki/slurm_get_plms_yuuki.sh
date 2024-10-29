#!/bin/bash
#SBATCH --job-name=get_plms_yuuki
#SBATCH --time=1:00:00
#SBATCH --array=1-1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --partition=kipac

ests=(TT EE TE ET TB BT EB BE)
qe=${ests[$SLURM_ARRAY_TASK_ID%8-1]}
sim1=$(((SLURM_ARRAY_TASK_ID-1)/8+1))
sim2=$(((SLURM_ARRAY_TASK_ID-1)/8+2))

export OMP_NUM_THREADS=12

/home/users/yukanaka/gmv/20240802_test_for_yuuki/get_plms_yuuki.py TB 4 5 yuuki
/home/users/yukanaka/gmv/20240802_test_for_yuuki/get_plms_yuuki.py TB 70 71 yuuki
/home/users/yukanaka/gmv/20240802_test_for_yuuki/get_plms_yuuki.py TB 82 83 yuuki
/home/users/yukanaka/gmv/20240802_test_for_yuuki/get_plms_yuuki.py EB 84 85 yuuki
/home/users/yukanaka/gmv/20240802_test_for_yuuki/get_plms_yuuki.py EE 50 51 yuuki

#/home/users/yukanaka/gmv/20240802_test_for_yuuki/get_plms_yuuki.py $qe $sim1 $sim1 yuuki
#/home/users/yukanaka/gmv/20240802_test_for_yuuki/get_plms_yuuki.py $qe $sim1 $sim2 yuuki
#/home/users/yukanaka/gmv/20240802_test_for_yuuki/get_plms_yuuki.py $qe $sim2 $sim1 yuuki
#/home/users/yukanaka/gmv/20240802_test_for_yuuki/get_plms_yuuki.py $qe $sim1 $sim1 yuuki_cmbonly_phi1_tqu1tqu2
#/home/users/yukanaka/gmv/20240802_test_for_yuuki/get_plms_yuuki.py $qe $sim1 $sim1 yuuki_cmbonly_phi1_tqu2tqu1
#/home/users/yukanaka/gmv/20240802_test_for_yuuki/get_plms_yuuki.py $qe $sim1 $sim2 yuuki_cmbonly
#/home/users/yukanaka/gmv/20240802_test_for_yuuki/get_plms_yuuki.py $qe $sim2 $sim1 yuuki_cmbonly
