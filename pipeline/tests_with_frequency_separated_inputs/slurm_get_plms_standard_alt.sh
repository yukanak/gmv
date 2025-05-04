#!/bin/bash
#SBATCH --job-name=get_plms_standard
#SBATCH --time=6:30:00
#SBATCH --array=144-199
#SBATCH --cpus-per-task=12
#SBATCH --mem=50G
#SBATCH --partition=kipac

sim1=$SLURM_ARRAY_TASK_ID
sim2=$(($sim1+1))

export OMP_NUM_THREADS=12

# SQE
~/gmv/new/get_plms_standard.py TT $sim1 $sim1 standard test_yuka.yaml
~/gmv/new/get_plms_standard.py TT $sim1 $sim2 standard test_yuka.yaml
~/gmv/new/get_plms_standard.py TT $sim2 $sim1 standard test_yuka.yaml
~/gmv/new/get_plms_standard.py TT $sim1 $sim1 standard_cmbonly_phi1_tqu1tqu2 test_yuka.yaml
~/gmv/new/get_plms_standard.py TT $sim1 $sim1 standard_cmbonly_phi1_tqu2tqu1 test_yuka.yaml
~/gmv/new/get_plms_standard.py TT $sim1 $sim2 standard_cmbonly test_yuka.yaml
~/gmv/new/get_plms_standard.py TT $sim2 $sim1 standard_cmbonly test_yuka.yaml

~/gmv/new/get_plms_standard.py TT $sim1 $sim1 standard test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py TT $sim1 $sim2 standard test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py TT $sim2 $sim1 standard test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py TT $sim1 $sim1 standard_cmbonly_phi1_tqu1tqu2 test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py TT $sim1 $sim1 standard_cmbonly_phi1_tqu2tqu1 test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py TT $sim1 $sim2 standard_cmbonly test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py TT $sim2 $sim1 standard_cmbonly test_yuka_lmaxT3500.yaml

~/gmv/new/get_plms_standard.py TT r $sim1 standard test_yuka.yaml
~/gmv/new/get_plms_standard.py TT $sim1 r standard test_yuka.yaml
~/gmv/new/get_plms_standard.py TT r $sim1 standard test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py TT $sim1 r standard test_yuka_lmaxT3500.yaml

~/gmv/new/get_plms_standard.py TE $sim1 $sim1 standard test_yuka.yaml
~/gmv/new/get_plms_standard.py TE $sim1 $sim2 standard test_yuka.yaml
~/gmv/new/get_plms_standard.py TE $sim2 $sim1 standard test_yuka.yaml
~/gmv/new/get_plms_standard.py TE $sim1 $sim1 standard_cmbonly_phi1_tqu1tqu2 test_yuka.yaml
~/gmv/new/get_plms_standard.py TE $sim1 $sim1 standard_cmbonly_phi1_tqu2tqu1 test_yuka.yaml
~/gmv/new/get_plms_standard.py TE $sim1 $sim2 standard_cmbonly test_yuka.yaml
~/gmv/new/get_plms_standard.py TE $sim2 $sim1 standard_cmbonly test_yuka.yaml

~/gmv/new/get_plms_standard.py TE $sim1 $sim1 standard test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py TE $sim1 $sim2 standard test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py TE $sim2 $sim1 standard test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py TE $sim1 $sim1 standard_cmbonly_phi1_tqu1tqu2 test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py TE $sim1 $sim1 standard_cmbonly_phi1_tqu2tqu1 test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py TE $sim1 $sim2 standard_cmbonly test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py TE $sim2 $sim1 standard_cmbonly test_yuka_lmaxT3500.yaml

~/gmv/new/get_plms_standard.py TE r $sim1 standard test_yuka.yaml
~/gmv/new/get_plms_standard.py TE $sim1 r standard test_yuka.yaml
~/gmv/new/get_plms_standard.py TE r $sim1 standard test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py TE $sim1 r standard test_yuka_lmaxT3500.yaml

~/gmv/new/get_plms_standard.py ET $sim1 $sim1 standard test_yuka.yaml
~/gmv/new/get_plms_standard.py ET $sim1 $sim2 standard test_yuka.yaml
~/gmv/new/get_plms_standard.py ET $sim2 $sim1 standard test_yuka.yaml
~/gmv/new/get_plms_standard.py ET $sim1 $sim1 standard_cmbonly_phi1_tqu1tqu2 test_yuka.yaml
~/gmv/new/get_plms_standard.py ET $sim1 $sim1 standard_cmbonly_phi1_tqu2tqu1 test_yuka.yaml
~/gmv/new/get_plms_standard.py ET $sim1 $sim2 standard_cmbonly test_yuka.yaml
~/gmv/new/get_plms_standard.py ET $sim2 $sim1 standard_cmbonly test_yuka.yaml

~/gmv/new/get_plms_standard.py ET $sim1 $sim1 standard test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py ET $sim1 $sim2 standard test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py ET $sim2 $sim1 standard test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py ET $sim1 $sim1 standard_cmbonly_phi1_tqu1tqu2 test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py ET $sim1 $sim1 standard_cmbonly_phi1_tqu2tqu1 test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py ET $sim1 $sim2 standard_cmbonly test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py ET $sim2 $sim1 standard_cmbonly test_yuka_lmaxT3500.yaml

~/gmv/new/get_plms_standard.py ET r $sim1 standard test_yuka.yaml
~/gmv/new/get_plms_standard.py ET $sim1 r standard test_yuka.yaml
~/gmv/new/get_plms_standard.py ET r $sim1 standard test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py ET $sim1 r standard test_yuka_lmaxT3500.yaml

~/gmv/new/get_plms_standard.py EE $sim1 $sim1 standard test_yuka.yaml
~/gmv/new/get_plms_standard.py EE $sim1 $sim2 standard test_yuka.yaml
~/gmv/new/get_plms_standard.py EE $sim2 $sim1 standard test_yuka.yaml
~/gmv/new/get_plms_standard.py EE $sim1 $sim1 standard_cmbonly_phi1_tqu1tqu2 test_yuka.yaml
~/gmv/new/get_plms_standard.py EE $sim1 $sim1 standard_cmbonly_phi1_tqu2tqu1 test_yuka.yaml
~/gmv/new/get_plms_standard.py EE $sim1 $sim2 standard_cmbonly test_yuka.yaml
~/gmv/new/get_plms_standard.py EE $sim2 $sim1 standard_cmbonly test_yuka.yaml

~/gmv/new/get_plms_standard.py EE $sim1 $sim1 standard test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py EE $sim1 $sim2 standard test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py EE $sim2 $sim1 standard test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py EE $sim1 $sim1 standard_cmbonly_phi1_tqu1tqu2 test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py EE $sim1 $sim1 standard_cmbonly_phi1_tqu2tqu1 test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py EE $sim1 $sim2 standard_cmbonly test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py EE $sim2 $sim1 standard_cmbonly test_yuka_lmaxT3500.yaml

~/gmv/new/get_plms_standard.py EE r $sim1 standard test_yuka.yaml
~/gmv/new/get_plms_standard.py EE $sim1 r standard test_yuka.yaml
~/gmv/new/get_plms_standard.py EE r $sim1 standard test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py EE $sim1 r standard test_yuka_lmaxT3500.yaml

~/gmv/new/get_plms_standard.py TB $sim1 $sim1 standard test_yuka.yaml
~/gmv/new/get_plms_standard.py TB $sim1 $sim2 standard test_yuka.yaml
~/gmv/new/get_plms_standard.py TB $sim2 $sim1 standard test_yuka.yaml
~/gmv/new/get_plms_standard.py TB $sim1 $sim1 standard_cmbonly_phi1_tqu1tqu2 test_yuka.yaml
~/gmv/new/get_plms_standard.py TB $sim1 $sim1 standard_cmbonly_phi1_tqu2tqu1 test_yuka.yaml
~/gmv/new/get_plms_standard.py TB $sim1 $sim2 standard_cmbonly test_yuka.yaml
~/gmv/new/get_plms_standard.py TB $sim2 $sim1 standard_cmbonly test_yuka.yaml

~/gmv/new/get_plms_standard.py TB $sim1 $sim1 standard test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py TB $sim1 $sim2 standard test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py TB $sim2 $sim1 standard test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py TB $sim1 $sim1 standard_cmbonly_phi1_tqu1tqu2 test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py TB $sim1 $sim1 standard_cmbonly_phi1_tqu2tqu1 test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py TB $sim1 $sim2 standard_cmbonly test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py TB $sim2 $sim1 standard_cmbonly test_yuka_lmaxT3500.yaml

~/gmv/new/get_plms_standard.py TB r $sim1 standard test_yuka.yaml
~/gmv/new/get_plms_standard.py TB $sim1 r standard test_yuka.yaml
~/gmv/new/get_plms_standard.py TB r $sim1 standard test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py TB $sim1 r standard test_yuka_lmaxT3500.yaml

~/gmv/new/get_plms_standard.py BT $sim1 $sim1 standard test_yuka.yaml
~/gmv/new/get_plms_standard.py BT $sim1 $sim2 standard test_yuka.yaml
~/gmv/new/get_plms_standard.py BT $sim2 $sim1 standard test_yuka.yaml
~/gmv/new/get_plms_standard.py BT $sim1 $sim1 standard_cmbonly_phi1_tqu1tqu2 test_yuka.yaml
~/gmv/new/get_plms_standard.py BT $sim1 $sim1 standard_cmbonly_phi1_tqu2tqu1 test_yuka.yaml
~/gmv/new/get_plms_standard.py BT $sim1 $sim2 standard_cmbonly test_yuka.yaml
~/gmv/new/get_plms_standard.py BT $sim2 $sim1 standard_cmbonly test_yuka.yaml

~/gmv/new/get_plms_standard.py BT $sim1 $sim1 standard test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py BT $sim1 $sim2 standard test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py BT $sim2 $sim1 standard test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py BT $sim1 $sim1 standard_cmbonly_phi1_tqu1tqu2 test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py BT $sim1 $sim1 standard_cmbonly_phi1_tqu2tqu1 test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py BT $sim1 $sim2 standard_cmbonly test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py BT $sim2 $sim1 standard_cmbonly test_yuka_lmaxT3500.yaml

~/gmv/new/get_plms_standard.py BT r $sim1 standard test_yuka.yaml
~/gmv/new/get_plms_standard.py BT $sim1 r standard test_yuka.yaml
~/gmv/new/get_plms_standard.py BT r $sim1 standard test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py BT $sim1 r standard test_yuka_lmaxT3500.yaml

~/gmv/new/get_plms_standard.py EB $sim1 $sim1 standard test_yuka.yaml
~/gmv/new/get_plms_standard.py EB $sim1 $sim2 standard test_yuka.yaml
~/gmv/new/get_plms_standard.py EB $sim2 $sim1 standard test_yuka.yaml
~/gmv/new/get_plms_standard.py EB $sim1 $sim1 standard_cmbonly_phi1_tqu1tqu2 test_yuka.yaml
~/gmv/new/get_plms_standard.py EB $sim1 $sim1 standard_cmbonly_phi1_tqu2tqu1 test_yuka.yaml
~/gmv/new/get_plms_standard.py EB $sim1 $sim2 standard_cmbonly test_yuka.yaml
~/gmv/new/get_plms_standard.py EB $sim2 $sim1 standard_cmbonly test_yuka.yaml

~/gmv/new/get_plms_standard.py EB $sim1 $sim1 standard test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py EB $sim1 $sim2 standard test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py EB $sim2 $sim1 standard test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py EB $sim1 $sim1 standard_cmbonly_phi1_tqu1tqu2 test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py EB $sim1 $sim1 standard_cmbonly_phi1_tqu2tqu1 test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py EB $sim1 $sim2 standard_cmbonly test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py EB $sim2 $sim1 standard_cmbonly test_yuka_lmaxT3500.yaml

~/gmv/new/get_plms_standard.py EB r $sim1 standard test_yuka.yaml
~/gmv/new/get_plms_standard.py EB $sim1 r standard test_yuka.yaml
~/gmv/new/get_plms_standard.py EB r $sim1 standard test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py EB $sim1 r standard test_yuka_lmaxT3500.yaml

~/gmv/new/get_plms_standard.py BE $sim1 $sim1 standard test_yuka.yaml
~/gmv/new/get_plms_standard.py BE $sim1 $sim2 standard test_yuka.yaml
~/gmv/new/get_plms_standard.py BE $sim2 $sim1 standard test_yuka.yaml
~/gmv/new/get_plms_standard.py BE $sim1 $sim1 standard_cmbonly_phi1_tqu1tqu2 test_yuka.yaml
~/gmv/new/get_plms_standard.py BE $sim1 $sim1 standard_cmbonly_phi1_tqu2tqu1 test_yuka.yaml
~/gmv/new/get_plms_standard.py BE $sim1 $sim2 standard_cmbonly test_yuka.yaml
~/gmv/new/get_plms_standard.py BE $sim2 $sim1 standard_cmbonly test_yuka.yaml

~/gmv/new/get_plms_standard.py BE $sim1 $sim1 standard test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py BE $sim1 $sim2 standard test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py BE $sim2 $sim1 standard test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py BE $sim1 $sim1 standard_cmbonly_phi1_tqu1tqu2 test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py BE $sim1 $sim1 standard_cmbonly_phi1_tqu2tqu1 test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py BE $sim1 $sim2 standard_cmbonly test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py BE $sim2 $sim1 standard_cmbonly test_yuka_lmaxT3500.yaml

~/gmv/new/get_plms_standard.py BE r $sim1 standard test_yuka.yaml
~/gmv/new/get_plms_standard.py BE $sim1 r standard test_yuka.yaml
~/gmv/new/get_plms_standard.py BE r $sim1 standard test_yuka_lmaxT3500.yaml
~/gmv/new/get_plms_standard.py BE $sim1 r standard test_yuka_lmaxT3500.yaml

