#!/usr/bin/env python3
import numpy as np
import pickle
import healpy as hp
import camb
import os, sys
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import gmv_resp
import healqest_utils as utils
import matplotlib.pyplot as plt
import weights
import wignerd
import resp

sims=np.arange(99)+1
n0_n1_sims=np.arange(98)+1
config_file='test_yuka.yaml'
append='mh'
#append='crossilc_twoseds'
lbins=np.logspace(np.log10(50),np.log10(3000),20)
config = utils.parse_yaml(config_file)
lmax = config['lensrec']['Lmax']
lmin = config['lensrec']['lminT']
lmaxT = config['lensrec']['lmaxT']
lmaxP = config['lensrec']['lmaxP']
nside = config['lensrec']['nside']
dir_out = config['dir_out']
l = np.arange(0,lmax+1)
num = len(sims)
bin_centers = (lbins[:-1] + lbins[1:]) / 2
digitized = np.digitize(l, lbins)
u = None
withT3 = False
sim = 1

# Compare MH N0 for with vs without T3 for 12 estimators
filename = dir_out+'n0/n0_98simpairs_healqest_gmv_cinv_lmaxT3000_lmaxP4096_nside2048_standard_resp_from_sims.pkl'
n0_standard = pickle.load(open(filename,'rb'))
n0_standard_total = n0_standard['total'] * (l*(l+1))**2/4

# Compare MH N0 for with vs without T3 for 9 estimators
filename = dir_out+'n0/n0_98simpairs_healqest_gmv_noT3_lmaxT3000_lmaxP4096_nside2048_mh_resp_from_sims_9ests.pkl'
n0_eq4549_noT3 = pickle.load(open(filename,'rb'))
n0_eq4549_noT3_total = n0_eq4549_noT3['total'] * (l*(l+1))**2/4
filename = dir_out+'n0/n0_98simpairs_healqest_gmv_cinv_noT3_lmaxT3000_lmaxP4096_nside2048_mh_resp_from_sims_9ests.pkl'
n0_cinv_noT3 = pickle.load(open(filename,'rb'))
n0_cinv_noT3_total = n0_cinv_noT3['total'] * (l*(l+1))**2/4
n0_cinv_noT3_T1T2 = n0_cinv_noT3['T1T2'] * (l*(l+1))**2/4
n0_cinv_noT3_T2T1 = n0_cinv_noT3['T2T1'] * (l*(l+1))**2/4

filename = dir_out+'n0/n0_98simpairs_healqest_gmv_withT3_lmaxT3000_lmaxP4096_nside2048_mh_resp_from_sims_9ests_fixedweights.pkl'
n0_eq4549_withT3 = pickle.load(open(filename,'rb'))
n0_eq4549_withT3_total = n0_eq4549_withT3['total'] * (l*(l+1))**2/4
filename = dir_out+'n0/n0_98simpairs_healqest_gmv_cinv_withT3_lmaxT3000_lmaxP4096_nside2048_mh_resp_from_sims_9ests.pkl'
n0_cinv_withT3 = pickle.load(open(filename,'rb'))
n0_cinv_withT3_total = n0_cinv_withT3['total'] * (l*(l+1))**2/4

# Compare cross-ILC N0 for with vs without T3 for 9 estimators
#filename = dir_out+'n0/n0_98simpairs_healqest_gmv_noT3_lmaxT3000_lmaxP4096_nside2048_crossilc_twoseds_resp_from_sims_9ests.pkl'
#n0_eq4549_noT3 = pickle.load(open(filename,'rb'))
#n0_eq4549_noT3_total = n0_eq4549_noT3['total'] * (l*(l+1))**2/4
#filename = dir_out+'n0/n0_98simpairs_healqest_gmv_cinv_noT3_lmaxT3000_lmaxP4096_nside2048_crossilc_twoseds_resp_from_sims_9ests.pkl'
#n0_cinv_noT3 = pickle.load(open(filename,'rb'))
#n0_cinv_noT3_total = n0_cinv_noT3['total'] * (l*(l+1))**2/4

#filename = dir_out+'n0/n0_98simpairs_healqest_gmv_withT3_lmaxT3000_lmaxP4096_nside2048_crossilc_twoseds_resp_from_sims_9ests_fixedweights.pkl'
#n0_eq4549_withT3 = pickle.load(open(filename,'rb'))
#n0_eq4549_withT3_total = n0_eq4549_withT3['total'] * (l*(l+1))**2/4
#filename = dir_out+'n0/n0_98simpairs_healqest_gmv_cinv_withT3_lmaxT3000_lmaxP4096_nside2048_crossilc_twoseds_resp_from_sims_9ests.pkl'
#n0_cinv_withT3 = pickle.load(open(filename,'rb'))
#n0_cinv_withT3_total = n0_cinv_withT3['total'] * (l*(l+1))**2/4

# Compare MH N0 for with vs without T3 for 12 estimators
#filename = dir_out+'n0/n0_98simpairs_healqest_gmv_noT3_lmaxT3000_lmaxP4096_nside2048_mh_resp_from_sims_12ests.pkl'
#n0_eq4549_noT3 = pickle.load(open(filename,'rb'))
#n0_eq4549_noT3_total = n0_eq4549_noT3['total'] * (l*(l+1))**2/4
#filename = dir_out+'n0/n0_98simpairs_healqest_gmv_cinv_noT3_lmaxT3000_lmaxP4096_nside2048_mh_resp_from_sims_12ests.pkl'
#n0_cinv_noT3 = pickle.load(open(filename,'rb'))
#n0_cinv_noT3_total = n0_cinv_noT3['total'] * (l*(l+1))**2/4

#filename = dir_out+'n0/n0_98simpairs_healqest_gmv_withT3_lmaxT3000_lmaxP4096_nside2048_mh_resp_from_sims_12ests.pkl'
#n0_eq4549_withT3 = pickle.load(open(filename,'rb'))
#n0_eq4549_withT3_total = n0_eq4549_withT3['total'] * (l*(l+1))**2/4
#filename = dir_out+'n0/n0_98simpairs_healqest_gmv_cinv_withT3_lmaxT3000_lmaxP4096_nside2048_mh_resp_from_sims_12ests.pkl'
#n0_cinv_withT3 = pickle.load(open(filename,'rb'))
#n0_cinv_withT3_total = n0_cinv_withT3['total'] * (l*(l+1))**2/4

# Theory spectrum
clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
clkk = slpp * (l*(l+1))**2/4

# Plot
#plt.clf()
#plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')

#plt.plot(l, n0_eq4549_noT3_total, color='forestgreen', alpha=0.8, linestyle='-',label='Eq. 45-49, no T3')
#plt.plot(l, n0_cinv_noT3_total, color='darkorange', alpha=0.8, linestyle='-',label='Cinv-style, no T3')
#plt.plot(l, n0_eq4549_withT3_total, color='firebrick', alpha=0.8, linestyle='-',label='Eq. 45-49, with T3')
#plt.plot(l, n0_cinv_withT3_total, color='darkblue', alpha=0.8, linestyle='-',label='Cinv-style, with T3')

#plt.ylabel("$[\ell(\ell+1)]^2$$N_0$ / 4 $[\mu K^2]$")
#plt.xlabel('$\ell$')
#plt.title(f'GMV MH (9 Estimators) Reconstruction Noise Comparison, with vs without T3')
##plt.title(f'GMV Cross-ILC (9 Estimators) Reconstruction Noise Comparison, with vs without T3')
#plt.legend(loc='upper left', fontsize='small')
#plt.xscale('log')
#plt.yscale('log')
#plt.xlim(10,lmax)
#plt.savefig(dir_out+f'/figs/n0_comparison_gmv_mh_9ests_with_vs_without_T3.png',bbox_inches='tight')
##plt.savefig(dir_out+f'/figs/n0_comparison_gmv_crossilc_twoseds_9ests_with_vs_without_T3.png',bbox_inches='tight')

#plt.clf()
#ratio_eq4549 = n0_eq4549_noT3_total/n0_eq4549_withT3_total
#ratio_cinv = n0_cinv_noT3_total/n0_cinv_withT3_total
#plt.axhline(y=1, color='k', linestyle='--')
#plt.plot(l, ratio_eq4549, color='forestgreen', alpha=0.8, linestyle='-',label='Ratio Eq. 45-49: no T3 / with T3')
#plt.plot(l, ratio_cinv, color='darkorange', alpha=0.8, linestyle='-',label='Ratio cinv-style: no T3 / with T3')
#plt.xlabel('$\ell$')
#plt.title(f'GMV MH Reconstruction Noise Comparison, with vs without T3, 12 Estimators')
##plt.title(f'GMV MH Reconstruction Noise Comparison, with vs without T3, 9 Estimators')
#plt.legend(loc='upper left', fontsize='small')
#plt.xscale('log')
#plt.xlim(10,lmax)
##plt.savefig(dir_out+f'/figs/n0_comparison_gmv_mh_9ests_with_vs_without_T3_ratio.png',bbox_inches='tight')
##plt.savefig(dir_out+f'/figs/n0_comparison_gmv_crossilc_twoseds_9ests_with_vs_without_T3_ratio.png',bbox_inches='tight')
#plt.savefig(dir_out+f'/figs/n0_comparison_gmv_mh_12ests_with_vs_without_T3_ratio.png',bbox_inches='tight')

#plt.clf()
#ratio_eq4549_noT3 = n0_eq4549_noT3_total/n0_standard_total
#ratio_eq4549_withT3 = n0_eq4549_withT3_total/n0_standard_total
#ratio_cinv_noT3 = n0_cinv_noT3_total/n0_standard_total
#ratio_cinv_withT3 = n0_cinv_withT3_total/n0_standard_total
#plt.axhline(y=1, color='k', linestyle='--')
#plt.plot(l, ratio_eq4549_noT3, color='forestgreen', alpha=0.8, linestyle='-',label='Ratio Eq. 45-49 no T3 / standard')
#plt.plot(l, ratio_eq4549_withT3, color='darkblue', alpha=0.8, linestyle='-',label='Ratio Eq. 45-49 with T3 / standard')
#plt.plot(l, ratio_cinv_noT3, color='darkorange', alpha=0.8, linestyle='-',label='Ratio cinv-style no T3 / standard')
#plt.plot(l, ratio_cinv_withT3, color='firebrick', alpha=0.8, linestyle='-',label='Ratio cinv-style with T3 / standard')
#plt.xlabel('$\ell$')
#plt.title(f'GMV MH Reconstruction Noise Comparison, with vs without T3, 12 Estimators')
##plt.title(f'GMV MH Reconstruction Noise Comparison, with vs without T3, 9 Estimators')
#plt.legend(loc='upper left', fontsize='small')
#plt.xscale('log')
#plt.xlim(10,lmax)
#plt.savefig(dir_out+f'/figs/n0_comparison_gmv_mh_12ests_with_vs_without_T3_ratio_with_standard.png',bbox_inches='tight')
##plt.savefig(dir_out+f'/figs/n0_comparison_gmv_mh_9ests_with_vs_without_T3_ratio_with_standard.png',bbox_inches='tight')

plt.clf()
filename = dir_out+'n0/n0_98simpairs_healqest_gmv_cinv_noT3_lmaxT3000_lmaxP4096_nside2048_mh_resp_from_sims_9ests_inflatedT2.pkl'
n0_cinv_noT3 = pickle.load(open(filename,'rb'))
n0_cinv_noT3_T1T2 = n0_cinv_noT3['T1T2'] * (l*(l+1))**2/4
n0_cinv_noT3_T2T1 = n0_cinv_noT3['T2T1'] * (l*(l+1))**2/4
n0_ratio = n0_cinv_noT3_T1T2/n0_cinv_noT3_T2T1
#plm_T1T2 = np.load(dir_out+f'/plm_T1T2_healqest_gmv_seed1_1_seed2_1_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_mh_cinv_noT3.npy')
#plm_T2T1 = np.load(dir_out+f'/plm_T2T1_healqest_gmv_seed1_1_seed2_1_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_mh_cinv_noT3.npy')
plm_T1T2 = np.load(dir_out+f'/plm_T1T2_healqest_gmv_seed1_1_seed2_1_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_mh_cinv_noT3_inflatedT2.npy')
plm_T2T1 = np.load(dir_out+f'/plm_T2T1_healqest_gmv_seed1_1_seed2_1_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_mh_cinv_noT3_inflatedT2.npy')
clpp_T1T2 = hp.alm2cl(plm_T1T2)
clpp_T2T1 = hp.alm2cl(plm_T2T1)
clpp_ratio = clpp_T1T2/clpp_T2T1
#resp_T1T2 = np.load(dir_out+'/resp/sim_resp_gmv_cinv_estT1T2_lmaxT3000_lmaxP4096_lmin300_cltypelcmb_mh_noT3.npy')
#resp_T2T1 = np.load(dir_out+'/resp/sim_resp_gmv_cinv_estT2T1_lmaxT3000_lmaxP4096_lmin300_cltypelcmb_mh_noT3.npy')
resp_T1T2 = np.load(dir_out+'/resp/sim_resp_gmv_cinv_estT1T2_lmaxT3000_lmaxP4096_lmin300_cltypelcmb_mh_noT3_inflatedT2.npy')
resp_T2T1 = np.load(dir_out+'/resp/sim_resp_gmv_cinv_estT2T1_lmaxT3000_lmaxP4096_lmin300_cltypelcmb_mh_noT3_inflatedT2.npy')
resp_ratio = resp_T1T2/resp_T2T1
plt.axhline(y=1, color='k', linestyle='--')
plt.plot(l, n0_ratio, color='firebrick', alpha=0.8, linestyle='-',label='Ratio N0: T1T2 / T2T1')
plt.plot(l, clpp_ratio, color='darkblue', alpha=0.8, linestyle='--',label='Ratio raw clpp sim 1: T1T2 / T2T1')
plt.plot(l, resp_ratio, color='forestgreen', alpha=0.8, linestyle=':',label='Ratio sim resp: T1T2 / T2T1')
plt.xlabel('$\ell$')
plt.title(f'GMV MH Reconstruction Comparison')
plt.legend(loc='upper left', fontsize='small')
plt.xscale('log')
plt.xlim(10,lmax)
#plt.savefig(dir_out+f'/figs/reconstruction_comparison_gmv_mh_T1T2_vs_T2T1.png',bbox_inches='tight')
plt.savefig(dir_out+f'/figs/reconstruction_comparison_gmv_mh_inflatedT2_T1T2_vs_T2T1.png',bbox_inches='tight')

'''

# With 12 estimators, they match exactly, even when sim1 != sim2
sim1 = 1; sim2 = 1
plm_gmv_old_dict = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_noT3_12ests_cutweights_dict.npy',allow_pickle=True)
#gmv_old_dark_blue = plm_gmv_old_dict[()]['TE_GMV']
#gmv_old_light_blue = plm_gmv_old_dict[()]['ET_GMV']
gmv_old_dark_blue = plm_gmv_old_dict[()]['TE_GMV']+plm_gmv_old_dict[()]['T2E1_GMV']
gmv_old_light_blue = plm_gmv_old_dict[()]['ET_GMV']+plm_gmv_old_dict[()]['E2T1_GMV']

plm_gmv_cinv_T1T2_darkblue = np.load(dir_out+f'/plm_T1T2_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_noT3_cutweights_darkblue.npy')
plm_gmv_cinv_T2T1_darkblue = np.load(dir_out+f'/plm_T2T1_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_noT3_cutweights_darkblue.npy')
plm_gmv_cinv_T1T2_lightblue = np.load(dir_out+f'/plm_T1T2_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_noT3_cutweights_lightblue.npy')
plm_gmv_cinv_T2T1_lightblue = np.load(dir_out+f'/plm_T2T1_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_noT3_cutweights_lightblue.npy')
gmv_cinv_dark_blue = 0.5*(plm_gmv_cinv_T1T2_darkblue+plm_gmv_cinv_T2T1_darkblue)
gmv_cinv_light_blue = 0.5*(plm_gmv_cinv_T1T2_lightblue+plm_gmv_cinv_T2T1_lightblue)

cl_gmv_old_dark_blue = hp.alm2cl(gmv_old_dark_blue)
cl_gmv_cinv_dark_blue = hp.alm2cl(gmv_cinv_dark_blue)
cl_gmv_old_sum = hp.alm2cl(gmv_old_dark_blue+gmv_old_light_blue)
cl_gmv_cinv_sum = hp.alm2cl(gmv_cinv_dark_blue+gmv_cinv_light_blue)

plt.figure(0)
plt.clf()
plt.axhline(y=1, color='k', linestyle='--')
#plt.plot(l,cl_gmv_old_dark_blue/cl_gmv_cinv_dark_blue,color='darkblue',alpha=0.5,label=f"Ratio GMV dark blue term, sim1 = {sim1}, sim2 = {sim2}, with 9 estimators: Eq. 45-49 / cinv")
#plt.plot(l,cl_gmv_old_sum/cl_gmv_cinv_sum,color='firebrick',alpha=0.5,label=f"Ratio GMV dark blue + light blue term, sim1 = {sim1}, sim2 = {sim2}, with 9 estimators: Eq. 45-49 / cinv")
plt.plot(l,cl_gmv_old_dark_blue/cl_gmv_cinv_dark_blue,color='darkblue',alpha=0.5,label=f"Ratio GMV dark blue term, sim1 = {sim1}, sim2 = {sim2}, with 12 estimators: Eq. 45-49 / cinv")
plt.plot(l,cl_gmv_old_sum/cl_gmv_cinv_sum,color='firebrick',alpha=0.5,label=f"Ratio GMV dark blue + light blue term, sim1 = {sim1}, sim2 = {sim2}, with 12 estimators: Eq. 45-49 / cinv")
plt.xlabel('$\ell$')
plt.legend(loc='lower left', fontsize='x-small')
plt.xscale('log')
plt.xlim(10,lmax)
#plt.savefig(dir_out+f'/figs/gmv_check_terms_9est_sim1_{sim1}_sim2_{sim2}.png',bbox_inches='tight')
plt.savefig(dir_out+f'/figs/gmv_check_terms_12est_sim1_{sim1}_sim2_{sim2}.png',bbox_inches='tight')

sim1 = 1; sim2 = 50
plm_gmv_old_dict = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_noT3_12ests_cutweights_dict.npy',allow_pickle=True)
#gmv_old_dark_blue = plm_gmv_old_dict[()]['TE_GMV']
#gmv_old_light_blue = plm_gmv_old_dict[()]['ET_GMV']
gmv_old_dark_blue = plm_gmv_old_dict[()]['TE_GMV']+plm_gmv_old_dict[()]['T2E1_GMV']
gmv_old_light_blue = plm_gmv_old_dict[()]['ET_GMV']+plm_gmv_old_dict[()]['E2T1_GMV']

plm_gmv_cinv_T1T2_darkblue = np.load(dir_out+f'/plm_T1T2_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_noT3_cutweights_darkblue.npy')
plm_gmv_cinv_T2T1_darkblue = np.load(dir_out+f'/plm_T2T1_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_noT3_cutweights_darkblue.npy')
plm_gmv_cinv_T1T2_lightblue = np.load(dir_out+f'/plm_T1T2_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_noT3_cutweights_lightblue.npy')
plm_gmv_cinv_T2T1_lightblue = np.load(dir_out+f'/plm_T2T1_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_noT3_cutweights_lightblue.npy')
gmv_cinv_dark_blue = 0.5*(plm_gmv_cinv_T1T2_darkblue+plm_gmv_cinv_T2T1_darkblue)
gmv_cinv_light_blue = 0.5*(plm_gmv_cinv_T1T2_lightblue+plm_gmv_cinv_T2T1_lightblue)

cl_gmv_old_dark_blue = hp.alm2cl(gmv_old_dark_blue)
cl_gmv_cinv_dark_blue = hp.alm2cl(gmv_cinv_dark_blue)
cl_gmv_old_sum = hp.alm2cl(gmv_old_dark_blue+gmv_old_light_blue)
cl_gmv_cinv_sum = hp.alm2cl(gmv_cinv_dark_blue+gmv_cinv_light_blue)

plt.figure(0)
plt.clf()
plt.axhline(y=1, color='k', linestyle='--')
#plt.plot(l,cl_gmv_old_dark_blue/cl_gmv_cinv_dark_blue,color='darkblue',alpha=0.5,label=f"Ratio GMV dark blue term, sim1 = {sim1}, sim2 = {sim2}, with 9 estimators: Eq. 45-49 / cinv")
#plt.plot(l,cl_gmv_old_sum/cl_gmv_cinv_sum,color='firebrick',alpha=0.5,label=f"Ratio GMV dark blue + light blue term, sim1 = {sim1}, sim2 = {sim2}, with 9 estimators: Eq. 45-49 / cinv")
plt.plot(l,cl_gmv_old_dark_blue/cl_gmv_cinv_dark_blue,color='darkblue',alpha=0.5,label=f"Ratio GMV dark blue term, sim1 = {sim1}, sim2 = {sim2}, with 12 estimators: Eq. 45-49 / cinv")
plt.plot(l,cl_gmv_old_sum/cl_gmv_cinv_sum,color='firebrick',alpha=0.5,label=f"Ratio GMV dark blue + light blue term, sim1 = {sim1}, sim2 = {sim2}, with 12 estimators: Eq. 45-49 / cinv")
plt.xlabel('$\ell$')
plt.legend(loc='lower left', fontsize='x-small')
plt.xscale('log')
plt.xlim(10,lmax)
#plt.savefig(dir_out+f'/figs/gmv_check_terms_9est_sim1_{sim1}_sim2_{sim2}.png',bbox_inches='tight')
plt.savefig(dir_out+f'/figs/gmv_check_terms_12est_sim1_{sim1}_sim2_{sim2}.png',bbox_inches='tight')

##########

# Check if the terms cancel in the 9 estimator case with no T3 (dark+light blue cancels if sim1 = sim2)

sim1 = 1; sim2 = 1
totalcls_filename = dir_out+f'totalcls/totalcls_average_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_mh.npy'
w_tsz_null = np.loadtxt('ilc_weights/weights1d_TT_spt3g_cmbynull.dat')
w_Tmv = np.loadtxt('ilc_weights/weights1d_TT_spt3g_cmbmv.dat')
w_Emv = np.loadtxt('ilc_weights/weights1d_EE_spt3g_cmbmv.dat')
alm_cmb_sim1 = f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/tqu1/len/alms/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb1_seed{sim1}_alm_lmax{lmax}.fits'
alm_cmb_sim2 = f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/tqu1/len/alms/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb1_seed{sim2}_alm_lmax{lmax}.fits'
flm_150ghz_sim1 = f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_150ghz_seed{sim1}_alm_lmax{lmax}.fits'
flm_220ghz_sim1 = f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_220ghz_seed{sim1}_alm_lmax{lmax}.fits'
flm_95ghz_sim1 = f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_95ghz_seed{sim1}_alm_lmax{lmax}.fits'
flm_150ghz_sim2 = f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_150ghz_seed{sim2}_alm_lmax{lmax}.fits'
flm_220ghz_sim2 = f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_220ghz_seed{sim2}_alm_lmax{lmax}.fits'
flm_95ghz_sim2 = f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_95ghz_seed{sim2}_alm_lmax{lmax}.fits'
nlm1_090_filename = dir_out + f'nlm/nlm_090_lmax{lmax}_seed{sim1}.alm'
nlm1_150_filename = dir_out + f'nlm/nlm_150_lmax{lmax}_seed{sim1}.alm'
nlm1_220_filename = dir_out + f'nlm/nlm_220_lmax{lmax}_seed{sim1}.alm'
nlm2_090_filename = dir_out + f'nlm/nlm_090_lmax{lmax}_seed{sim2}.alm'
nlm2_150_filename = dir_out + f'nlm/nlm_150_lmax{lmax}_seed{sim2}.alm'
nlm2_220_filename = dir_out + f'nlm/nlm_220_lmax{lmax}_seed{sim2}.alm'

totalcls = np.load(totalcls_filename)
cltt1 = totalcls[:,4]; cltt2 = totalcls[:,5]; clttx = totalcls[:,6]; cltt3 = totalcls[:,0]; clee = totalcls[:,1]; clbb = totalcls[:,2]; clte = totalcls[:,3]

tlm1,elm1,blm1 = hp.read_alm(alm_cmb_sim1,hdu=[1,2,3])
tlm1_150 = tlm1.copy(); tlm1_220 = tlm1.copy(); tlm1_95 = tlm1.copy()
elm1_150 = elm1.copy(); elm1_220 = elm1.copy(); elm1_95 = elm1.copy()
tlm2,elm2,blm2 = hp.read_alm(alm_cmb_sim2,hdu=[1,2,3])
tlm2_150 = tlm2.copy(); tlm2_220 = tlm2.copy(); tlm2_95 = tlm2.copy()
elm2_150 = elm2.copy(); elm2_220 = elm2.copy(); elm2_95 = elm2.copy()

tflm1_150, eflm1_150, bflm1_150 = hp.read_alm(flm_150ghz_sim1,hdu=[1,2,3])
tflm1_220, eflm1_220, bflm1_220 = hp.read_alm(flm_220ghz_sim1,hdu=[1,2,3])
tflm1_95, eflm1_95, bflm1_95 = hp.read_alm(flm_95ghz_sim1,hdu=[1,2,3])
tlm1_150 += tflm1_150; tlm1_220 += tflm1_220; tlm1_95 += tflm1_95
elm1_150 += eflm1_150; elm1_220 += eflm1_220; elm1_95 += eflm1_95
tflm2_150, eflm2_150, bflm2_150 = hp.read_alm(flm_150ghz_sim2,hdu=[1,2,3])
tflm2_220, eflm2_220, bflm2_220 = hp.read_alm(flm_220ghz_sim2,hdu=[1,2,3])
tflm2_95, eflm2_95, bflm2_95 = hp.read_alm(flm_95ghz_sim2,hdu=[1,2,3])
tlm2_150 += tflm2_150; tlm2_220 += tflm2_220; tlm2_95 += tflm2_95
elm2_150 += eflm2_150; elm2_220 += eflm2_220; elm2_95 += eflm2_95

nlmt1_090,nlme1_090,nlmb1_090 = hp.read_alm(nlm1_090_filename,hdu=[1,2,3])
nlmt1_150,nlme1_150,nlmb1_150 = hp.read_alm(nlm1_150_filename,hdu=[1,2,3])
nlmt1_220,nlme1_220,nlmb1_220 = hp.read_alm(nlm1_220_filename,hdu=[1,2,3])
tlm1_150 += nlmt1_150; tlm1_220 += nlmt1_220; tlm1_95 += nlmt1_090
elm1_150 += nlme1_150; elm1_220 += nlme1_220; elm1_95 += nlme1_090
nlmt2_090,nlme2_090,nlmb2_090 = hp.read_alm(nlm2_090_filename,hdu=[1,2,3])
nlmt2_150,nlme2_150,nlmb2_150 = hp.read_alm(nlm2_150_filename,hdu=[1,2,3])
nlmt2_220,nlme2_220,nlmb2_220 = hp.read_alm(nlm2_220_filename,hdu=[1,2,3])
tlm2_150 += nlmt2_150; tlm2_220 += nlmt2_220; tlm2_95 += nlmt2_090
elm2_150 += nlme2_150; elm2_220 += nlme2_220; elm2_95 += nlme2_090

tlm1_mv = hp.almxfl(tlm1_95,w_Tmv[0][:lmax+1]) + hp.almxfl(tlm1_150,w_Tmv[1][:lmax+1]) + hp.almxfl(tlm1_220,w_Tmv[2][:lmax+1])
tlm1_tszn = hp.almxfl(tlm1_95,w_tsz_null[0][:lmax+1]) + hp.almxfl(tlm1_150,w_tsz_null[1][:lmax+1]) + hp.almxfl(tlm1_220,w_tsz_null[2][:lmax+1])
elm1 = hp.almxfl(elm1_95,w_Emv[0][:lmax+1]) + hp.almxfl(elm1_150,w_Emv[1][:lmax+1]) + hp.almxfl(elm1_220,w_Emv[2][:lmax+1])
tlm2_mv = hp.almxfl(tlm2_95,w_Tmv[0][:lmax+1]) + hp.almxfl(tlm2_150,w_Tmv[1][:lmax+1]) + hp.almxfl(tlm2_220,w_Tmv[2][:lmax+1])
tlm2_tszn = hp.almxfl(tlm2_95,w_tsz_null[0][:lmax+1]) + hp.almxfl(tlm2_150,w_tsz_null[1][:lmax+1]) + hp.almxfl(tlm2_220,w_tsz_null[2][:lmax+1])
elm2 = hp.almxfl(elm2_95,w_Emv[0][:lmax+1]) + hp.almxfl(elm2_150,w_Emv[1][:lmax+1]) + hp.almxfl(elm2_220,w_Emv[2][:lmax+1])

l_1 = 1000
l_2 = 500
#gmv_old_dark_blue = clee[l_1]*clte[l_2]*tlm1_mv*elm2
#gmv_cinv_dark_blue = clee[l_1]*clte[l_2]*0.5*(tlm1_mv*elm2 + tlm1_tszn*elm2)
#gmv_old_light_blue = clte[l_1]*clee[l_2]*elm1*tlm2_tszn
#gmv_cinv_light_blue = clte[l_1]*clee[l_2]*0.5*(elm1*tlm2_tszn + elm1*tlm2_mv)
gmv_old_dark_blue = tlm1_mv*elm2
gmv_cinv_dark_blue = 0.5*(tlm1_mv*elm2 + tlm1_tszn*elm2)
gmv_old_light_blue = elm1*tlm2_tszn
gmv_cinv_light_blue = 0.5*(elm1*tlm2_tszn + elm1*tlm2_mv)

print(gmv_old_dark_blue)
print(gmv_cinv_dark_blue)
print(gmv_old_dark_blue+gmv_old_light_blue)
print(gmv_cinv_dark_blue+gmv_cinv_light_blue)

cl_gmv_old_dark_blue = hp.alm2cl(gmv_old_dark_blue)
cl_gmv_cinv_dark_blue = hp.alm2cl(gmv_cinv_dark_blue)
cl_gmv_old_sum = hp.alm2cl(gmv_old_dark_blue+gmv_old_light_blue)
cl_gmv_cinv_sum = hp.alm2cl(gmv_cinv_dark_blue+gmv_cinv_light_blue)

print(cl_gmv_old_dark_blue)
print(cl_gmv_cinv_dark_blue)
print(cl_gmv_old_sum)
print(cl_gmv_cinv_sum)

plt.figure(0)
plt.clf()
plt.axhline(y=1, color='k', linestyle='--')
plt.plot(l,cl_gmv_old_dark_blue/cl_gmv_cinv_dark_blue,color='darkblue',alpha=0.5,label=f"Ratio GMV dark blue term, sim1 = {sim1}, sim2 = {sim2}, with 9 estimators: Eq. 45-49 / cinv")
plt.plot(l,cl_gmv_old_sum/cl_gmv_cinv_sum,color='firebrick',alpha=0.5,label=f"Ratio GMV dark blue + light blue term, sim1 = {sim1}, sim2 = {sim2}, with 9 estimators: Eq. 45-49 / cinv")
plt.xlabel('$\ell$')
plt.legend(loc='lower left', fontsize='x-small')
plt.xscale('log')
plt.xlim(10,lmax)
plt.savefig(dir_out+f'/figs/gmv_check_terms_9est_sim1_{sim1}_sim2_{sim2}.png',bbox_inches='tight')

# But with sim2 != sim1, doesn't match

sim1 = 1; sim2 = 50
totalcls_filename = dir_out+f'totalcls/totalcls_average_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_mh.npy'
w_tsz_null = np.loadtxt('ilc_weights/weights1d_TT_spt3g_cmbynull.dat')
w_Tmv = np.loadtxt('ilc_weights/weights1d_TT_spt3g_cmbmv.dat')
w_Emv = np.loadtxt('ilc_weights/weights1d_EE_spt3g_cmbmv.dat')
alm_cmb_sim1 = f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/tqu1/len/alms/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb1_seed{sim1}_alm_lmax{lmax}.fits'
alm_cmb_sim2 = f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/tqu1/len/alms/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb1_seed{sim2}_alm_lmax{lmax}.fits'
flm_150ghz_sim1 = f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_150ghz_seed{sim1}_alm_lmax{lmax}.fits'
flm_220ghz_sim1 = f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_220ghz_seed{sim1}_alm_lmax{lmax}.fits'
flm_95ghz_sim1 = f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_95ghz_seed{sim1}_alm_lmax{lmax}.fits'
flm_150ghz_sim2 = f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_150ghz_seed{sim2}_alm_lmax{lmax}.fits'
flm_220ghz_sim2 = f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_220ghz_seed{sim2}_alm_lmax{lmax}.fits'
flm_95ghz_sim2 = f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_95ghz_seed{sim2}_alm_lmax{lmax}.fits'
nlm1_090_filename = dir_out + f'nlm/nlm_090_lmax{lmax}_seed{sim1}.alm'
nlm1_150_filename = dir_out + f'nlm/nlm_150_lmax{lmax}_seed{sim1}.alm'
nlm1_220_filename = dir_out + f'nlm/nlm_220_lmax{lmax}_seed{sim1}.alm'
nlm2_090_filename = dir_out + f'nlm/nlm_090_lmax{lmax}_seed{sim2}.alm'
nlm2_150_filename = dir_out + f'nlm/nlm_150_lmax{lmax}_seed{sim2}.alm'
nlm2_220_filename = dir_out + f'nlm/nlm_220_lmax{lmax}_seed{sim2}.alm'

totalcls = np.load(totalcls_filename)
cltt1 = totalcls[:,4]; cltt2 = totalcls[:,5]; clttx = totalcls[:,6]; cltt3 = totalcls[:,0]; clee = totalcls[:,1]; clbb = totalcls[:,2]; clte = totalcls[:,3]

tlm1,elm1,blm1 = hp.read_alm(alm_cmb_sim1,hdu=[1,2,3])
tlm1_150 = tlm1.copy(); tlm1_220 = tlm1.copy(); tlm1_95 = tlm1.copy()
elm1_150 = elm1.copy(); elm1_220 = elm1.copy(); elm1_95 = elm1.copy()
tlm2,elm2,blm2 = hp.read_alm(alm_cmb_sim2,hdu=[1,2,3])
tlm2_150 = tlm2.copy(); tlm2_220 = tlm2.copy(); tlm2_95 = tlm2.copy()
elm2_150 = elm2.copy(); elm2_220 = elm2.copy(); elm2_95 = elm2.copy()

tflm1_150, eflm1_150, bflm1_150 = hp.read_alm(flm_150ghz_sim1,hdu=[1,2,3])
tflm1_220, eflm1_220, bflm1_220 = hp.read_alm(flm_220ghz_sim1,hdu=[1,2,3])
tflm1_95, eflm1_95, bflm1_95 = hp.read_alm(flm_95ghz_sim1,hdu=[1,2,3])
tlm1_150 += tflm1_150; tlm1_220 += tflm1_220; tlm1_95 += tflm1_95
elm1_150 += eflm1_150; elm1_220 += eflm1_220; elm1_95 += eflm1_95
tflm2_150, eflm2_150, bflm2_150 = hp.read_alm(flm_150ghz_sim2,hdu=[1,2,3])
tflm2_220, eflm2_220, bflm2_220 = hp.read_alm(flm_220ghz_sim2,hdu=[1,2,3])
tflm2_95, eflm2_95, bflm2_95 = hp.read_alm(flm_95ghz_sim2,hdu=[1,2,3])
tlm2_150 += tflm2_150; tlm2_220 += tflm2_220; tlm2_95 += tflm2_95
elm2_150 += eflm2_150; elm2_220 += eflm2_220; elm2_95 += eflm2_95

nlmt1_090,nlme1_090,nlmb1_090 = hp.read_alm(nlm1_090_filename,hdu=[1,2,3])
nlmt1_150,nlme1_150,nlmb1_150 = hp.read_alm(nlm1_150_filename,hdu=[1,2,3])
nlmt1_220,nlme1_220,nlmb1_220 = hp.read_alm(nlm1_220_filename,hdu=[1,2,3])
tlm1_150 += nlmt1_150; tlm1_220 += nlmt1_220; tlm1_95 += nlmt1_090
elm1_150 += nlme1_150; elm1_220 += nlme1_220; elm1_95 += nlme1_090
nlmt2_090,nlme2_090,nlmb2_090 = hp.read_alm(nlm2_090_filename,hdu=[1,2,3])
nlmt2_150,nlme2_150,nlmb2_150 = hp.read_alm(nlm2_150_filename,hdu=[1,2,3])
nlmt2_220,nlme2_220,nlmb2_220 = hp.read_alm(nlm2_220_filename,hdu=[1,2,3])
tlm2_150 += nlmt2_150; tlm2_220 += nlmt2_220; tlm2_95 += nlmt2_090
elm2_150 += nlme2_150; elm2_220 += nlme2_220; elm2_95 += nlme2_090

tlm1_mv = hp.almxfl(tlm1_95,w_Tmv[0][:lmax+1]) + hp.almxfl(tlm1_150,w_Tmv[1][:lmax+1]) + hp.almxfl(tlm1_220,w_Tmv[2][:lmax+1])
tlm1_tszn = hp.almxfl(tlm1_95,w_tsz_null[0][:lmax+1]) + hp.almxfl(tlm1_150,w_tsz_null[1][:lmax+1]) + hp.almxfl(tlm1_220,w_tsz_null[2][:lmax+1])
elm1 = hp.almxfl(elm1_95,w_Emv[0][:lmax+1]) + hp.almxfl(elm1_150,w_Emv[1][:lmax+1]) + hp.almxfl(elm1_220,w_Emv[2][:lmax+1])
tlm2_mv = hp.almxfl(tlm2_95,w_Tmv[0][:lmax+1]) + hp.almxfl(tlm2_150,w_Tmv[1][:lmax+1]) + hp.almxfl(tlm2_220,w_Tmv[2][:lmax+1])
tlm2_tszn = hp.almxfl(tlm2_95,w_tsz_null[0][:lmax+1]) + hp.almxfl(tlm2_150,w_tsz_null[1][:lmax+1]) + hp.almxfl(tlm2_220,w_tsz_null[2][:lmax+1])
elm2 = hp.almxfl(elm2_95,w_Emv[0][:lmax+1]) + hp.almxfl(elm2_150,w_Emv[1][:lmax+1]) + hp.almxfl(elm2_220,w_Emv[2][:lmax+1])

l_1 = 1000
l_2 = 500
#gmv_old_dark_blue = clee[l_1]*clte[l_2]*tlm1_mv*elm2
#gmv_cinv_dark_blue = clee[l_1]*clte[l_2]*0.5*(tlm1_mv*elm2 + tlm1_tszn*elm2)
#gmv_old_light_blue = clte[l_1]*clee[l_2]*elm1*tlm2_tszn
#gmv_cinv_light_blue = clte[l_1]*clee[l_2]*0.5*(elm1*tlm2_tszn + elm1*tlm2_mv)
gmv_old_dark_blue = tlm1_mv*elm2
gmv_cinv_dark_blue = 0.5*(tlm1_mv*elm2 + tlm1_tszn*elm2)
gmv_old_light_blue = elm1*tlm2_tszn
gmv_cinv_light_blue = 0.5*(elm1*tlm2_tszn + elm1*tlm2_mv)

cl_gmv_old_dark_blue = hp.alm2cl(gmv_old_dark_blue)
cl_gmv_cinv_dark_blue = hp.alm2cl(gmv_cinv_dark_blue)
cl_gmv_old_sum = hp.alm2cl(gmv_old_dark_blue+gmv_old_light_blue)
cl_gmv_cinv_sum = hp.alm2cl(gmv_cinv_dark_blue+gmv_cinv_light_blue)

plt.figure(0)
plt.clf()
plt.axhline(y=1, color='k', linestyle='--')
plt.plot(l,cl_gmv_old_dark_blue/cl_gmv_cinv_dark_blue,color='darkblue',alpha=0.5,label=f"Ratio GMV dark blue term, sim1 = {sim1}, sim2 = {sim2}, with 9 estimators: Eq. 45-49 / cinv")
plt.plot(l,cl_gmv_old_sum/cl_gmv_cinv_sum,color='firebrick',alpha=0.5,label=f"Ratio GMV dark blue + light blue term, sim1 = {sim1}, sim2 = {sim2}, with 9 estimators: Eq. 45-49 / cinv")
plt.xlabel('$\ell$')
plt.legend(loc='lower left', fontsize='x-small')
plt.xscale('log')
plt.xlim(10,lmax)
plt.savefig(dir_out+f'/figs/gmv_check_terms_9est_sim1_{sim1}_sim2_{sim2}.png',bbox_inches='tight')

##########

# Try inflating noise in tSZ T map, 9 estimators should see more of a difference (it doesn't)

# Load GMV plms
plm_gmv = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_noT3_inflatedT2.npy')
#plm_gmv_TTEETE = np.load(dir_out+f'/plm_TTEETE_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_noT3_inflatedT2.npy')
#plm_gmv_TBEB = np.load(dir_out+f'/plm_TBEB_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_noT3_inflatedT2.npy')
plm_gmv_12ests = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_noT3_12ests_inflatedT2.npy')
plm_gmv_withT3 = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_inflatedT2.npy')

# Load cinv-style GMV plms
ests = ['T1T2', 'T2T1', 'EE', 'TE', 'ET', 'TB', 'BT', 'EB', 'BE']
plms = np.zeros((len(np.load(dir_out+f'/plm_T1T2_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_noT3_inflatedT2.npy')),len(ests)), dtype=np.complex_)
for i, est in enumerate(ests):
    plms[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_noT3_inflatedT2.npy')
plm = 0.5*plms[:,0]+0.5*plms[:,1]+np.sum(plms[:,2:], axis=1)
plm_TTEETE = 0.5*plms[:,0]+0.5*plms[:,1]+np.sum(plms[:,2:5], axis=1)
plm_TBEB = np.sum(plms[:,5:], axis=1)
plms_withT3 = np.zeros((len(np.load(dir_out+f'/plm_T1T2_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_inflatedT2.npy')),len(ests)), dtype=np.complex_)
for i, est in enumerate(ests):
    plms_withT3[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_inflatedT2.npy')
plm_withT3 = 0.5*plms_withT3[:,0]+0.5*plms_withT3[:,1]+np.sum(plms_withT3[:,2:], axis=1)
plm_TTEETE_withT3 = 0.5*plms_withT3[:,0]+0.5*plms_withT3[:,1]+np.sum(plms_withT3[:,2:5], axis=1)
plm_TBEB_withT3 = np.sum(plms_withT3[:,5:], axis=1)
ests = ['T1T2', 'T2T1', 'EE', 'E2E1', 'TE', 'T2E1', 'ET', 'E2T1', 'TB', 'BT', 'EB', 'BE']
plms_12ests = np.zeros((len(np.load(dir_out+f'/plm_T1T2_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_noT3_inflatedT2.npy')),len(ests)), dtype=np.complex_)
for i, est in enumerate(ests):
    plms_12ests[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_noT3_inflatedT2.npy')
plm_12ests = 0.5*np.sum(plms_12ests[:,:8], axis=1)+np.sum(plms_12ests[:,8:], axis=1)

clpp_old_all = hp.alm2cl(plm_gmv)
clpp_cinv_all = hp.alm2cl(plm)
clpp_old_all_12ests = hp.alm2cl(plm_gmv_12ests)
clpp_cinv_all_12ests = hp.alm2cl(plm_12ests)
clpp_old_all_withT3 = hp.alm2cl(plm_gmv_withT3)
clpp_cinv_all_withT3 = hp.alm2cl(plm_withT3)
#clpp_old_TTEETE = hp.alm2cl(plm_gmv_TTEETE)
#clpp_cinv_TTEETE = hp.alm2cl(plm_TTEETE)
#clpp_old_TBEB = hp.alm2cl(plm_gmv_TBEB)
#clpp_cinv_TBEB = hp.alm2cl(plm_TBEB)

plt.figure(0)
plt.clf()
plt.axhline(y=1, color='k', linestyle='--')
plt.plot(l,clpp_old_all/clpp_cinv_all,color='darkblue',alpha=0.5,label="Ratio GMV all ests clpp sim 1 with 9 estimators: Eq. 45-49 / cinv")
plt.plot(l,clpp_old_all_12ests/clpp_cinv_all_12ests,color='firebrick',alpha=0.5,label="Ratio GMV all ests clpp sim 1 with 12 estimators: Eq. 45-49 / cinv")
#plt.plot(l,clpp_old_all_withT3/clpp_cinv_all_withT3,color='forestgreen',alpha=0.5,label="Ratio GMV all ests clpp sim 1 with 9 estimators and with T3: Eq. 45-49 / cinv")
plt.xlabel('$\ell$')
plt.legend(loc='lower left', fontsize='x-small')
plt.xscale('log')
plt.xlim(10,lmax)
plt.savefig(dir_out+f'/figs/gmv_inflatedT2.png',bbox_inches='tight')

##########

# Sanity check raw plm comparison for no T3, 9 estimators
ests = ['T1T2', 'T2T1', 'EE', 'TE', 'ET', 'TB', 'BT', 'EB', 'BE']
# Load GMV plms
sim = 1
plm_gmv = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_noT3.npy')
plm_gmv_TTEETE = np.load(dir_out+f'/plm_TTEETE_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_noT3.npy')
plm_gmv_TBEB = np.load(dir_out+f'/plm_TBEB_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_noT3.npy')

# Load cinv-style GMV plms
#sim = 50
sim = 1
plms = np.zeros((len(np.load(dir_out+f'/plm_T1T2_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_noT3.npy')),len(ests)), dtype=np.complex_)
for i, est in enumerate(ests):
    plms[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_noT3.npy')
plm = 0.5*plms[:,0]+0.5*plms[:,1]+np.sum(plms[:,2:], axis=1)
plm_TTEETE = 0.5*plms[:,0]+0.5*plms[:,1]+np.sum(plms[:,2:5], axis=1)
plm_TBEB = np.sum(plms[:,5:], axis=1)

clpp_old_all = hp.alm2cl(plm_gmv)
clpp_cinv_all = hp.alm2cl(plm)
clpp_old_TTEETE = hp.alm2cl(plm_gmv_TTEETE)
clpp_cinv_TTEETE = hp.alm2cl(plm_TTEETE)
clpp_old_TBEB = hp.alm2cl(plm_gmv_TBEB)
clpp_cinv_TBEB = hp.alm2cl(plm_TBEB)

# Actually... They're the same
plt.figure(0)
plt.clf()
plt.axhline(y=1, color='k', linestyle='--')
plt.plot(l,clpp_old_all/clpp_cinv_all,color='darkblue',alpha=0.5,label="Ratio GMV all ests clpp sim 1 with 9 estimators: Eq. 45-49 / cinv")
#plt.plot(l,clpp_old_TTEETE/clpp_cinv_TTEETE,color='firebrick',alpha=0.5,label="Ratio GMV TTEETE ests clpp sim 1 with 9 estimators: Eq. 45-49 / cinv")
#plt.plot(l,clpp_old_TBEB/clpp_cinv_TBEB,color='forestgreen',alpha=0.5,label="Ratio GMV TBEB ests clpp sim 1 with 9 estimators: Eq. 45-49 / cinv")
plt.xlabel('$\ell$')
plt.legend(loc='lower left', fontsize='x-small')
plt.xscale('log')
plt.xlim(10,lmax)
plt.savefig(dir_out+f'/figs/gmv_9ests.png',bbox_inches='tight')

##########

# Sanity check that 12 estimators for both methods are NOT identical, and that the issue comes from TTEETE not TBEB
ests = ['T1T2', 'T2T1', 'EE', 'E2E1', 'TE', 'T2E1', 'ET', 'E2T1', 'TB', 'BT', 'EB', 'BE']
plm_gmv = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_noT3_12ests.npy')
plm_gmv_TTEETE = np.load(dir_out+f'/plm_TTEETE_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_noT3_12ests.npy')
plm_gmv_TBEB = np.load(dir_out+f'/plm_TBEB_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_noT3_12ests.npy')
# Load cinv-style GMV plms
plms = np.zeros((len(np.load(dir_out+f'/plm_T1T2_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_noT3.npy')),len(ests)), dtype=np.complex_)
for i, est in enumerate(ests):
    plms[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_noT3.npy')
plm = 0.5*np.sum(plms[:,:8], axis=1)+np.sum(plms[:,8:], axis=1)
plm_TTEETE = 0.5*np.sum(plms[:,:8], axis=1)
plm_TBEB = np.sum(plms[:,8:], axis=1)

clpp_old_all = hp.alm2cl(plm_gmv)
clpp_cinv_all = hp.alm2cl(plm)
clpp_old_TTEETE = hp.alm2cl(plm_gmv_TTEETE)
clpp_cinv_TTEETE = hp.alm2cl(plm_TTEETE)
clpp_old_TBEB = hp.alm2cl(plm_gmv_TBEB)
clpp_cinv_TBEB = hp.alm2cl(plm_TBEB)

# Actually... They're the same
plt.figure(0)
plt.clf()
plt.axhline(y=1, color='k', linestyle='--')
plt.plot(l,clpp_old_all/clpp_cinv_all,color='darkblue',alpha=0.5,label="Ratio GMV all ests clpp sim 1 with 12 estimators: Eq. 45-49 / cinv")
plt.plot(l,clpp_old_TTEETE/clpp_cinv_TTEETE,color='firebrick',alpha=0.5,label="Ratio GMV TTEETE ests clpp sim 1 with 12 estimators: Eq. 45-49 / cinv")
plt.plot(l,clpp_old_TBEB/clpp_cinv_TBEB,color='forestgreen',alpha=0.5,label="Ratio GMV TBEB ests clpp sim 1 with 12 estimators: Eq. 45-49 / cinv")
plt.xlabel('$\ell$')
plt.legend(loc='lower left', fontsize='x-small')
plt.xscale('log')
plt.xlim(10,lmax)
plt.savefig(dir_out+f'/figs/gmv_12ests.png',bbox_inches='tight')

##########

plm_gmv_old_dict = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_noT3_12ests_nulledclte_dict.npy',allow_pickle=True)
plm_gmv_old_T1T2 = plm_gmv_old_dict[()]['TT_GMV']
plm_gmv_old_T2T1 = plm_gmv_old_dict[()]['T2T1_GMV']
plm_gmv_old_TE = plm_gmv_old_dict[()]['TE_GMV']
plm_gmv_cinv_T1T2 = np.load(dir_out+f'/plm_T1T2_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_noT3_nulledclte.npy')
plm_gmv_cinv_T2T1 = np.load(dir_out+f'/plm_T2T1_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_noT3_nulledclte.npy')
plm_gmv_cinv_TE = np.load(dir_out+f'/plm_TE_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_noT3_nulledclte.npy')
#np.array_equal(plm_gmv_old_T1T2,plm_gmv_cinv_T1T2) # False
#where_different = (~np.equal(plm_gmv_old_T1T2,plm_gmv_cinv_T1T2)).astype(int)
#indices = np.flatnonzero(where_different) # array([      1,       2,       3, ..., 8394750, 8394751, 8394752])
#np.shape(indices) # 8382622
#np.shape(plm_gmv_old_T1T2) # 8394753
#plm_gmv_old_T1T2[1] # (-4189.229605473229+0j)
#plm_gmv_cinv_T1T2[1] # (-4189.229605478817+0j)

clpp_old_T1T2 = hp.alm2cl(plm_gmv_old_T1T2)
clpp_cinv_T1T2 = hp.alm2cl(plm_gmv_cinv_T1T2)
clpp_old_T2T1 = hp.alm2cl(plm_gmv_old_T2T1)
clpp_cinv_T2T1 = hp.alm2cl(plm_gmv_cinv_T2T1)
clpp_old_TE = hp.alm2cl(plm_gmv_old_TE)
clpp_cinv_TE = hp.alm2cl(plm_gmv_cinv_TE)

plt.figure(0)
plt.clf()
plt.axhline(y=1, color='k', linestyle='--')
plt.plot(l,clpp_old_T1T2/clpp_cinv_T1T2,color='darkblue',alpha=0.5,label="Ratio GMV T1T2 clpp sim 1 with nulled clte: Eq. 45-49 / cinv")
plt.plot(l,clpp_old_T2T1/clpp_cinv_T2T1,color='firebrick',alpha=0.5,label="Ratio GMV T2T1 clpp sim 1 with nulled clte: Eq. 45-49 / cinv")
plt.plot(l,clpp_old_TE/clpp_cinv_TE,color='forestgreen',alpha=0.5,label="Ratio GMV TE clpp sim 1 with nulled clte: Eq. 45-49 / cinv")
plt.xlabel('$\ell$')
plt.legend(loc='lower left', fontsize='x-small')
plt.xscale('log')
plt.xlim(10,lmax)
plt.savefig(dir_out+f'/figs/gmv_nulledclte.png',bbox_inches='tight')

##########

sim = 1

plm_gmv_12 = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_noT3_12ests.npy')
plm_gmv_9 = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_noT3.npy')
#np.array_equal(plm_gmv_12,plm_gmv_9) # False
#where_different = (~np.equal(plm_gmv_12,plm_gmv_9)).astype(int)
#indices = np.flatnonzero(where_different) # [      3       4       7 ... 8394749 8394750 8394751]
#np.shape(indices) # 5960041
#np.shape(plm_gmv_12) # 8394753
#plm_gmv_9[4] # (46661.215689182325+0j)
#plm_gmv_12[4] # (46661.21568918232+0j)
clpp_12 = hp.alm2cl(plm_gmv_12)
clpp_9 = hp.alm2cl(plm_gmv_9)

resp_gmv_12 = np.load(dir_out+f'/resp/sim_resp_gmv_estall_12ests_lmaxT3000_lmaxP4096_lmin300_cltypelcmb_mh_noT3.npy')
resp_gmv_9 = np.load(dir_out+f'/resp/sim_resp_gmv_estall_lmaxT3000_lmaxP4096_lmin300_cltypelcmb_mh_noT3.npy')
inv_resp_gmv_12 = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_12[1:] = 1./(resp_gmv_12)[1:]
inv_resp_gmv_9 = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_9[1:] = 1./(resp_gmv_9)[1:]

plt.figure(0)
plt.clf()
plt.axhline(y=1, color='k', linestyle='--')
plt.plot(l,clpp_12/clpp_9,color='darkblue',alpha=0.5,label="Ratio GMV Eq. 45-49 clpp sim 1: 12 estimators / 9 estimators")
plt.plot(l,inv_resp_gmv_12/inv_resp_gmv_9,color='firebrick',alpha=0.5,label="Ratio GMV Eq. 45-49 1/R: 12 estimators / 9 estimators")
plt.xlabel('$\ell$')
plt.legend(loc='lower left', fontsize='x-small')
plt.xscale('log')
plt.xlim(10,lmax)
plt.savefig(dir_out+f'/figs/gmv_12ests_vs_9ests.png',bbox_inches='tight')
'''
