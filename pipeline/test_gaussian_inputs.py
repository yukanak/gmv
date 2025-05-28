import numpy as np
import pickle
import healpy as hp
import camb
import os, sys
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import healqest_utils as utils
import matplotlib.pyplot as plt
import weights
import wignerd
import resp

def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

sim = 1
sim2 = 2
fg_model = 'agora'
append = 'mh'
config_file='test_yuka_lmaxT3500.yaml'
config = utils.parse_yaml(config_file)
lmax = config['lensrec']['Lmax']
lmin = config['lensrec']['lminT']
lmaxT = config['lensrec']['lmaxT']
lmaxP = config['lensrec']['lmaxP']
nside = config['lensrec']['nside']
dir_out = config['dir_out']
l = np.arange(0,lmax+1)
plm = np.load(dir_out+f'/plm_summed_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv_noT3.npy')
#plm_ij = np.load(dir_out+f'/plm_summed_healqest_gmv_seed1_{sim}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv_noT3.npy')
plm_ij = np.load(dir_out+f'/plm_EE_healqest_gmv_seed1_{sim}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_crossilc_twoseds_cinv_noT3.npy')
#plm_ij_2 = np.load(dir_out+f'/plm_summed_healqest_gmv_seed1_2_seed2_3_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv_noT3.npy')
plm_ij_2 = np.load(dir_out+f'/plm_EE_healqest_gmv_seed1_2_seed2_3_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_crossilc_twoseds_cinv_noT3.npy')
plm_ji = np.load(dir_out+f'/plm_summed_healqest_gmv_seed1_{sim2}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv_noT3.npy')
plm_2 = np.load(dir_out+f'/plm_summed_healqest_gmv_seed1_2_seed2_2_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv_noT3.npy')
plm_old = np.load('/oak/stanford/orgs/kipac/users/yukanaka/outputs_with_frequency_separated_inputs'+f'/plm_summed_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_noT3.npy')
#plm_ij_old = np.load('/oak/stanford/orgs/kipac/users/yukanaka/outputs_with_frequency_separated_inputs'+f'/plm_summed_healqest_gmv_seed1_{sim}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_noT3.npy')
plm_ij_old = np.load('/oak/stanford/orgs/kipac/users/yukanaka/outputs_with_frequency_separated_inputs'+f'/plm_EE_healqest_gmv_seed1_{sim}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_crossilc_twoseds_cinv_noT3.npy')
plm_ji_old = np.load('/oak/stanford/orgs/kipac/users/yukanaka/outputs_with_frequency_separated_inputs'+f'/plm_summed_healqest_gmv_seed1_{sim2}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_noT3.npy')
plm_old_2 = np.load('/oak/stanford/orgs/kipac/users/yukanaka/outputs_with_frequency_separated_inputs'+f'/plm_summed_healqest_gmv_seed1_2_seed2_2_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_noT3.npy')
clpp = hp.alm2cl(plm)
clpp_ij = hp.alm2cl(plm_ij)
clpp_ij_2 = hp.alm2cl(plm_ij_2)
clpp_ji = hp.alm2cl(plm_ji)
clpp_2 = hp.alm2cl(plm_2)
clpp_old = hp.alm2cl(plm_old)
clpp_ij_old = hp.alm2cl(plm_ij_old)
clpp_ji_old = hp.alm2cl(plm_ji_old)
clpp_old_2 = hp.alm2cl(plm_old_2)
# Theory spectrum
clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
clkk = slpp * (l*(l+1))**2/4

# Plot
plt.clf()
plt.axhline(y=1, color='k', linestyle='--')
plt.plot(l, moving_average(clpp_ij_2/clpp_ij), color='lightgreen', alpha=0.8, linestyle='-',label='Raw plm, MH, ij2 new / ij1 new')
plt.plot(l, moving_average(clpp_ij/clpp_ij_old), color='pink', alpha=0.8, linestyle='-',label='Raw plm, MH, ij new / old')
#plt.plot(l, moving_average(clpp_old_2/clpp_old), color='cornflowerblue', alpha=0.8, linestyle='-',label='Raw plm, MH, seed 2 old / seed 1 old')
#plt.plot(l, moving_average(clpp_2/clpp), color='lightgreen', alpha=0.8, linestyle='-',label='Raw plm, MH, seed 2 new / seed 1 new')
#plt.plot(l, moving_average(clpp/clpp_old), color='pink', alpha=0.8, linestyle='-',label='Raw plm, MH, seed 1, new / old')
#plt.plot(l, moving_average(clpp_2/clpp_old), color='bisque', alpha=0.8, linestyle='-',label='Raw plm, MH, seed 2 new / seed 1 old')
plt.xlabel('$\ell$')
plt.legend(loc='upper left', fontsize='small')
plt.xscale('log')
plt.xlim(10,lmax)
plt.ylim(0.8,1.2)
plt.savefig(dir_out+f'/figs/temp.png',bbox_inches='tight')



