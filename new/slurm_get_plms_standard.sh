#!/bin/bash
#SBATCH --job-name=get_plms_standard
#SBATCH --time=5:00:00
#SBATCH --array=1000-1220
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --partition=kipac

ests=(TT TE ET EE TB BT EB BE)
qe=${ests[$SLURM_ARRAY_TASK_ID%8-1]}
#sim1=$(((SLURM_ARRAY_TASK_ID-1)/8+1))
sim1=$(((SLURM_ARRAY_TASK_ID-1)/8+100))
sim2=$(($sim1+1))

export OMP_NUM_THREADS=12

# If no averaged totalcls yet: just run what's commented out below to get the total Cls to average, for each lmaxT
# See get_plms_standard.py around lines 370 and on
# Once you have the total Cls for each input map, use average_totalcls.py to average
#~/gmv/new/get_plms_standard.py TT $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID standard_cinv test_yuka_lmaxT3500.yaml
#~/gmv/new/get_plms_standard.py TT $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID standard_cinv test_yuka_lmaxT4000.yaml

~/gmv/new/get_plms_standard.py $qe r $sim1 standard_cinv test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py $qe $sim1 r standard_cinv test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py $qe r $sim1 standard_cinv test_yuka_lmaxT4000.yaml
~/gmv/new/get_plms_standard.py $qe $sim1 r standard_cinv test_yuka_lmaxT4000.yaml
#~/gmv/new/get_plms_standard_temp.py $qe 1 $sim1 standard_cinv test_yuka.yaml
#~/gmv/new/get_plms_standard_temp.py $qe $sim1 1 standard_cinv test_yuka.yaml

~/gmv/new/get_plms_standard.py $qe $sim1 $sim1 standard_cinv test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py $qe $sim1 $sim2 standard_cinv test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py $qe $sim2 $sim1 standard_cinv test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py $qe $sim1 $sim1 standard_cmbonly_phi1_tqu1tqu2_cinv test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py $qe $sim1 $sim1 standard_cmbonly_phi1_tqu2tqu1_cinv test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py $qe $sim1 $sim2 standard_cmbonly_cinv test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py $qe $sim2 $sim1 standard_cmbonly_cinv test_yuka_lmaxT3500.yaml

~/gmv/new/get_plms_standard.py $qe $sim1 $sim1 standard_cinv test_yuka_lmaxT4000.yaml
~/gmv/new/get_plms_standard.py $qe $sim1 $sim2 standard_cinv test_yuka_lmaxT4000.yaml
~/gmv/new/get_plms_standard.py $qe $sim2 $sim1 standard_cinv test_yuka_lmaxT4000.yaml
~/gmv/new/get_plms_standard.py $qe $sim1 $sim1 standard_cmbonly_phi1_tqu1tqu2_cinv test_yuka_lmaxT4000.yaml
~/gmv/new/get_plms_standard.py $qe $sim1 $sim1 standard_cmbonly_phi1_tqu2tqu1_cinv test_yuka_lmaxT4000.yaml
~/gmv/new/get_plms_standard.py $qe $sim1 $sim2 standard_cmbonly_cinv test_yuka_lmaxT4000.yaml
~/gmv/new/get_plms_standard.py $qe $sim2 $sim1 standard_cmbonly_cinv test_yuka_lmaxT4000.yaml
