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
import scipy.signal as signal
from scipy.signal import lfilter
import matplotlib.colors as mcolors

def analyze():
    '''
    Compare with N0/N1 subtraction.
    Baselining NO T3 for MH, cross-ILC. The "withT3" case is not implemented yet since "noT3" is hard-coded in when getting the N0, etc.
    '''
    lmax = 4096
    l = np.arange(0,lmax+1)
    lbins = np.logspace(np.log10(50),np.log10(3000),20)
    bin_centers = (lbins[:-1] + lbins[1:]) / 2
    digitized = np.digitize(l, lbins)
    # Input kappa
    klm = hp.read_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_spt3g_input_klm_lmax{lmax}.fits')
    input_clkk = hp.alm2cl(klm,klm,lmax=lmax)
    binned_input_clkk = np.array([input_clkk[digitized == i].mean() for i in range(1, len(lbins))])

    # First, lmaxT = 3000 cases, NOT cinv-style
    config_file='test_yuka.yaml'
    config = utils.parse_yaml(config_file)
    append_list = ['agora_standard', 'agora_mh', 'agora_crossilc_twoseds', 'agora_crossilc_onesed']
    #binned_bias_gmv_3000 = get_lensing_bias(config,append_list,cinv=False,sqe=False,sims=np.arange(250)+1,n0_n1_sims=np.arange(249)+1)
    #binned_bias_sqe_3000 = get_lensing_bias(config,append_list,cinv=False,sqe=True,sims=np.arange(250)+1,n0_n1_sims=np.arange(249)+1)

    # lmaxT = 3000, cinv
    config_file='test_yuka.yaml'
    config = utils.parse_yaml(config_file)
    append_list = ['agora_standard']
    binned_bias_gmv_3000_cinv = get_lensing_bias(config,append_list,cinv=True,sqe=False,sims=np.arange(250)+1,n0_n1_sims=np.arange(249)+1)

    # lmaxT = 3500, cinv
    config_file='test_yuka_lmaxT3500.yaml'
    config = utils.parse_yaml(config_file)
    append_list = ['agora_standard', 'agora_mh', 'agora_crossilc_twoseds']
    binned_bias_gmv_3500_cinv = get_lensing_bias(config,append_list,cinv=True,sqe=False,sims=np.arange(250)+1,n0_n1_sims=np.arange(249)+1)

    # lmaxT = 4000, cinv
    config_file='test_yuka_lmaxT4000.yaml'
    config = utils.parse_yaml(config_file)
    append_list = ['agora_standard', 'agora_mh', 'agora_crossilc_twoseds']
    binned_bias_gmv_4000_cinv = get_lensing_bias(config,append_list,cinv=True,sqe=False,sims=np.arange(250)+1,n0_n1_sims=np.arange(249)+1)

    # lmaxT = 3000, rdn0 and cinv
    config_file='test_yuka.yaml'
    config = utils.parse_yaml(config_file)
    append_list = ['agora_standard']
    binned_bias_gmv_3000_cinv_rdn0 = get_lensing_bias(config,append_list,cinv=True,rdn0=True,sqe=False,sims=np.arange(250)+1,n0_n1_sims=np.arange(249)+1)

    # lmaxT = 3500, rdn0 and cinv
    config_file='test_yuka_lmaxT3500.yaml'
    config = utils.parse_yaml(config_file)
    append_list = ['agora_standard', 'agora_mh', 'agora_crossilc_twoseds']
    binned_bias_gmv_3500_cinv_rdn0 = get_lensing_bias(config,append_list,cinv=True,rdn0=True,sqe=False,sims=np.arange(250)+1,n0_n1_sims=np.arange(249)+1)

    # lmaxT = 4000, rdn0 and cinv
    config_file='test_yuka_lmaxT4000.yaml'
    config = utils.parse_yaml(config_file)
    append_list = ['agora_standard', 'agora_mh', 'agora_crossilc_twoseds']
    binned_bias_gmv_4000_cinv_rdn0 = get_lensing_bias(config,append_list,cinv=True,rdn0=True,sqe=False,sims=np.arange(250)+1,n0_n1_sims=np.arange(249)+1)

    dir_out = config['dir_out']

    # Plot
    plt.figure(0)
    plt.clf()
    plt.axhline(y=0, color='gray', alpha=0.5, linestyle='--')

    #plt.plot(bin_centers[:-2], binned_bias_gmv_3500[:-2,3]/binned_input_clkk[:-2], color='goldenrod', marker='o', linestyle='-', ms=3, alpha=0.8, label="Cross-ILC GMV (one component CIB)")
    plt.plot(bin_centers, binned_bias_gmv_3500_cinv_rdn0[:,2]/binned_input_clkk, color='darkorange', marker='o', linestyle='-', ms=3, alpha=0.8, label="Cross-ILC GMV, RDN0")
    plt.plot(bin_centers, binned_bias_gmv_3500_cinv_rdn0[:,1]/binned_input_clkk, color='forestgreen', marker='o', linestyle='-', ms=3, alpha=0.8, label="MH GMV, RDN0")
    #plt.plot(bin_centers[:-2], binned_bias_sqe_3500[:-2,0]/binned_input_clkk[:-2], color='firebrick', marker='o', linestyle='-', ms=3, alpha=0.8, label=f'Standard SQE')
    plt.plot(bin_centers, binned_bias_gmv_3500_cinv_rdn0[:,0]/binned_input_clkk, color='darkblue', marker='o', linestyle='-', ms=3, alpha=0.8, label="Standard GMV, RDN0")

    #plt.errorbar(bin_centers, binned_bias_gmv[:,2]/binned_input_clkk, yerr=binned_uncertainty_gmv[:,2]/binned_input_clkk, color='darkorange', marker='o', linestyle='-', ms=3, alpha=0.8, label="Cross-ILC GMV")
    #plt.errorbar(bin_centers, binned_bias_gmv[:,1]/binned_input_clkk, yerr=binned_uncertainty_gmv[:,1]/binned_input_clkk, color='forestgreen', marker='o', linestyle='-', ms=3, alpha=0.8, label="MH GMV")
    #plt.errorbar(bin_centers, binned_bias_sqe[:,0]/binned_input_clkk, yerr=binned_uncertainty_sqe[:,0]/binned_input_clkk, color='firebrick', marker='o', linestyle='-', ms=3, alpha=0.8, label=f'Standard SQE')
    #plt.errorbar(bin_centers, binned_bias_gmv[:,0]/binned_input_clkk, yerr=binned_uncertainty_gmv[:,0]/binned_input_clkk, color='darkblue', marker='o', linestyle='-', ms=3, alpha=0.8, label="Standard GMV")

    #plt.plot(bin_centers, binned_bias_gmv_3500_cinv[:,1]/binned_input_clkk, color='lightgreen', marker='o', linestyle='--', ms=3, alpha=0.8, label="MH GMV, Sim-Based N0")
    plt.plot(bin_centers, binned_bias_gmv_3500_cinv[:,0]/binned_input_clkk, color='cornflowerblue', marker='o', linestyle='--', ms=3, alpha=0.8, label="Standard GMV, Sim-Based N0")

    #errorbars_cinv = np.load(f'errorbars_cinv_standard_lmaxT3000.npy')
    #plt.errorbar(bin_centers[:-2],binned_bias_gmv_3000_cinv[:-2,0]/binned_input_clkk[:-2],yerr=errorbars_cinv[:-2],color='darkblue', alpha=0.5, marker='o', linestyle='None', ms=3, label="Standard GMV") 
    #plt.errorbar(bin_centers[:-2],binned_bias_gmv_3000_cinv_rdn0[:-2,0]/binned_input_clkk[:-2],yerr=errorbars_cinv[:-2],color='firebrick', alpha=0.5, marker='o', linestyle='None', ms=3, label="Standard GMV, RDN0") 

    #plt.ylabel("$C_\ell^{\kappa\kappa}$")
    plt.xlabel('$\ell$')
    plt.title(f'Lensing Bias from Agora Sim / Input Kappa Spectrum, Cinv-Style GMV, lmaxT = 3500')
    plt.legend(loc='upper left', fontsize='small')
    plt.xscale('log')
    plt.xlim(50,3001)
    #plt.xlim(10,lmax)
    plt.ylim(-0.2,0.2)
    #plt.savefig(dir_out+f'/figs/bias_total.png',bbox_inches='tight')
    plt.savefig(dir_out+f'/figs/bias_total_cinv.png',bbox_inches='tight')

    plt.clf()
    plt.axhline(y=0, color='gray', alpha=0.5, linestyle='--')

    plt.plot(bin_centers, binned_bias_gmv_3000_cinv_rdn0[:,0]/binned_input_clkk, color='darkblue', marker='o', linestyle=':', ms=3, alpha=0.8, label="Standard GMV, lmaxT = 3000")

    plt.plot(bin_centers, binned_bias_gmv_3500_cinv_rdn0[:,2]/binned_input_clkk, color='orange', marker='o', linestyle='--', ms=3, alpha=0.8, label="Cross-ILC GMV, lmaxT = 3500")
    plt.plot(bin_centers, binned_bias_gmv_4000_cinv_rdn0[:,2]/binned_input_clkk, color='bisque', marker='o', linestyle='-', ms=3, alpha=0.8, label="Cross-ILC GMV, lmaxT = 4000")

    plt.plot(bin_centers, binned_bias_gmv_3500_cinv_rdn0[:,1]/binned_input_clkk, color='mediumseagreen', marker='o', linestyle='--', ms=3, alpha=0.8, label="MH GMV, lmaxT = 3500")
    plt.plot(bin_centers, binned_bias_gmv_4000_cinv_rdn0[:,1]/binned_input_clkk, color='lightgreen', marker='o', linestyle='-', ms=3, alpha=0.8, label="MH GMV, lmaxT = 4000")

    plt.plot(bin_centers, binned_bias_gmv_3500_cinv_rdn0[:,0]/binned_input_clkk, color='cornflowerblue', marker='o', linestyle='--', ms=3, alpha=0.8, label="Standard GMV, lmaxT = 3500")
    plt.plot(bin_centers, binned_bias_gmv_4000_cinv_rdn0[:,0]/binned_input_clkk, color='lightsteelblue', marker='o', linestyle='-', ms=3, alpha=0.8, label="Standard GMV, lmaxT = 4000")

    plt.xlabel('$\ell$')
    plt.title(f'Lensing Bias from Agora Sim / Input Kappa Spectrum, Cinv-Style GMV, with RDN0')
    plt.legend(loc='upper left', fontsize='small')
    plt.xscale('log')
    plt.xlim(50,3001)
    plt.ylim(-0.2,0.2)
    plt.savefig(dir_out+f'/figs/bias_total_different_lmaxT.png',bbox_inches='tight')

'''
    plt.figure(1)
    plt.clf()
    plt.axhline(y=0, color='gray', alpha=0.5, linestyle='--')

    plt.plot(bin_centers, binned_bias_gmv[:,2]/binned_uncertainty_gmv[:,2], color='goldenrod', marker='o', linestyle='--', ms=3, alpha=0.8, label="Cross-ILC GMV (one component CIB)")
    plt.plot(bin_centers, binned_bias_gmv[:,3]/binned_uncertainty_gmv[:,3], color='darkorange', marker='o', linestyle='--', ms=3, alpha=0.8, label="Cross-ILC GMV (two component CIB)")
    plt.plot(bin_centers, binned_bias_gmv[:,1]/binned_uncertainty_gmv[:,1], color='forestgreen', marker='o', linestyle='--', ms=3, alpha=0.8, label="MH GMV")
    plt.plot(bin_centers, binned_bias_gmv[:,4]/binned_uncertainty_gmv[:,4], color='plum', marker='o', linestyle='--', ms=3, alpha=0.8, label="Profile Hardened GMV")
    plt.plot(bin_centers, binned_bias_sqe[:,0]/binned_uncertainty_sqe[:,0], color='firebrick', marker='o', linestyle='--', ms=3, alpha=0.8, label=f'Standard SQE')
    plt.plot(bin_centers, binned_bias_gmv[:,0]/binned_uncertainty_gmv[:,0], color='darkblue', marker='o', linestyle='--', ms=3, alpha=0.8, label="Standard GMV")

    #plt.ylabel("$C_\ell^{\kappa\kappa}$")
    plt.xlabel('$\ell$')
    plt.title(f'Bias / Uncertainty')
    plt.legend(loc='upper left', fontsize='x-small', bbox_to_anchor=(1,0.5))
    plt.xscale('log')
    plt.xlim(10,lmax)
    plt.ylim(-0.3,0.3)
    plt.savefig(dir_out+f'/figs/bias_over_uncertainty_total.png',bbox_inches='tight')
'''

def compare_n0():
    '''
    Compare N0 for different cases.
    '''
    lmax = 4096
    l = np.arange(0,lmax+1)
    lbins = np.logspace(np.log10(50),np.log10(3000),20)
    bin_centers = (lbins[:-1] + lbins[1:]) / 2
    digitized = np.digitize(l, lbins)
    # Input kappa
    klm = hp.read_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_spt3g_input_klm_lmax{lmax}.fits')
    input_clkk = hp.alm2cl(klm,klm,lmax=lmax)
    binned_input_clkk = np.array([input_clkk[digitized == i].mean() for i in range(1, len(lbins))])
    # Get output directory
    config_file = 'test_yuka_lmaxT3500.yaml'
    config = utils.parse_yaml(config_file)
    dir_out = config['dir_out']
    n0_n1_sims=np.arange(249)+1

    # lmaxT = 3500, RDN0 and cinv
    config_file='test_yuka_lmaxT3500.yaml'
    config = utils.parse_yaml(config_file)
    # Lensing bias
    append_list = ['agora_standard', 'agora_mh', 'agora_crossilc_twoseds']
    binned_bias_gmv_3500_cinv_rdn0 = get_lensing_bias(config,append_list,cinv=True,rdn0=True,sqe=False,sims=np.arange(250)+1,n0_n1_sims=np.arange(249)+1)
    # RDN0 for 9 estimators, no T3
    rdn0_lmaxT3500_cinv_standard = get_rdn0(sims=n0_n1_sims,qetype='gmv_cinv',config=config,append='standard')
    rdn0_lmaxT3500_cinv_standard *= (l*(l+1))**2/4
    binned_rdn0_lmaxT3500_cinv_standard = [rdn0_lmaxT3500_cinv_standard[digitized == i].mean() for i in range(1, len(lbins))]
    rdn0_lmaxT3500_cinv_mh_9ests = get_rdn0(sims=n0_n1_sims,qetype='gmv_cinv',config=config,append='mh')
    rdn0_lmaxT3500_cinv_mh_9ests *= (l*(l+1))**2/4
    binned_rdn0_lmaxT3500_cinv_mh_9ests = [rdn0_lmaxT3500_cinv_mh_9ests[digitized == i].mean() for i in range(1, len(lbins))]
    rdn0_lmaxT3500_cinv_crossilc_9ests = get_rdn0(sims=n0_n1_sims,qetype='gmv_cinv',config=config,append='crossilc_twoseds')
    rdn0_lmaxT3500_cinv_crossilc_9ests *= (l*(l+1))**2/4
    binned_rdn0_lmaxT3500_cinv_crossilc_9ests = [rdn0_lmaxT3500_cinv_crossilc_9ests[digitized == i].mean() for i in range(1, len(lbins))]

    # lmaxT = 4000, RDN0 and cinv
    config_file='test_yuka_lmaxT4000.yaml'
    config = utils.parse_yaml(config_file)
    # Lensing bias
    append_list = ['agora_standard', 'agora_mh', 'agora_crossilc_twoseds']
    binned_bias_gmv_4000_cinv_rdn0 = get_lensing_bias(config,append_list,cinv=True,rdn0=True,sqe=False,sims=np.arange(250)+1,n0_n1_sims=np.arange(249)+1)
    # RDN0 for 9 estimators, no T3
    rdn0_lmaxT4000_cinv_standard = get_rdn0(sims=n0_n1_sims,qetype='gmv_cinv',config=config,append='standard')
    rdn0_lmaxT4000_cinv_standard *= (l*(l+1))**2/4
    binned_rdn0_lmaxT4000_cinv_standard = [rdn0_lmaxT4000_cinv_standard[digitized == i].mean() for i in range(1, len(lbins))]
    rdn0_lmaxT4000_cinv_mh_9ests = get_rdn0(sims=n0_n1_sims,qetype='gmv_cinv',config=config,append='mh')
    rdn0_lmaxT4000_cinv_mh_9ests *= (l*(l+1))**2/4
    binned_rdn0_lmaxT4000_cinv_mh_9ests = [rdn0_lmaxT4000_cinv_mh_9ests[digitized == i].mean() for i in range(1, len(lbins))]
    rdn0_lmaxT4000_cinv_crossilc_9ests = get_rdn0(sims=n0_n1_sims,qetype='gmv_cinv',config=config,append='crossilc_twoseds')
    rdn0_lmaxT4000_cinv_crossilc_9ests *= (l*(l+1))**2/4
    binned_rdn0_lmaxT4000_cinv_crossilc_9ests = [rdn0_lmaxT4000_cinv_crossilc_9ests[digitized == i].mean() for i in range(1, len(lbins))]

    # Plot
    plt.figure(figsize=(10,6))
    plt.clf()
    #cm_blue = plt.cm.get_cmap('Blues')
    #cm_green = plt.cm.get_cmap('Greens')
    #cm_orange = plt.cm.get_cmap('Oranges')
    cm_blue = mcolors.LinearSegmentedColormap.from_list("cm_blue", ["#addfed", "#000080"])
    cm_green = mcolors.LinearSegmentedColormap.from_list("cm_green", ["#abdbbe", "#00441b"])
    cm_orange = mcolors.LinearSegmentedColormap.from_list("cm_orange", ["#ffe0c2", "#f5800f"])
    sc1 = plt.scatter(binned_bias_gmv_3500_cinv_rdn0[:,0]/binned_input_clkk, binned_rdn0_lmaxT3500_cinv_standard, c=bin_centers, cmap=cm_blue, marker='.', vmin=150, vmax=2700, label='Standard GMV (lmaxT = 3500)')
    sc2 = plt.scatter(binned_bias_gmv_3500_cinv_rdn0[:,1]/binned_input_clkk, binned_rdn0_lmaxT3500_cinv_mh_9ests, c=bin_centers, cmap=cm_green, marker='.', vmin=150, vmax=2700, label='MH GMV (lmaxT = 3500)')
    sc3 = plt.scatter(binned_bias_gmv_3500_cinv_rdn0[:,2]/binned_input_clkk, binned_rdn0_lmaxT3500_cinv_crossilc_9ests, c=bin_centers, cmap=cm_orange, marker='.', vmin=150, vmax=2700, label='Cross-ILC GMV (lmaxT = 3500)')
    plt.scatter(binned_bias_gmv_4000_cinv_rdn0[:,0]/binned_input_clkk, binned_rdn0_lmaxT4000_cinv_standard, cmap=cm_blue, c=bin_centers, marker='x', vmin=150, vmax=2700, label='Standard GMV (lmaxT = 4000)')
    plt.scatter(binned_bias_gmv_4000_cinv_rdn0[:,1]/binned_input_clkk, binned_rdn0_lmaxT4000_cinv_mh_9ests, cmap=cm_green, c=bin_centers, marker='x', vmin=150, vmax=2700, label='MH GMV (lmaxT = 4000)')
    plt.scatter(binned_bias_gmv_4000_cinv_rdn0[:,2]/binned_input_clkk, binned_rdn0_lmaxT4000_cinv_crossilc_9ests, c=bin_centers, cmap=cm_orange, marker='x', vmin=150, vmax=2700, label='Cross-ILC GMV (lmaxT = 4000)')

    plt.ylabel("$[\ell(\ell+1)]^2$$RDN_0$ / 4 $[\mu K^2]$")
    plt.xlabel('Lensing Bias / Input Kappa Spectrum')
    plt.title(f'Reconstruction Noise vs Lensing Bias, Cinv-Style GMV')
    plt.legend(loc='lower right', fontsize='small')
    plt.colorbar(sc1, label='L bins')
    plt.yscale('log')
    plt.xlim(-0.2,0.1)
    plt.ylim(1e-8,3e-7)
    plt.savefig(dir_out+f'/figs/n0_vs_bias_different_lmaxT.png',bbox_inches='tight')

    '''
    # lmaxT = 3500, RDN0, cinv-style MH, 12 estimators, no T3
    filename = dir_out+f'/n0/rdn0_249simpairs_healqest_gmv_cinv_noT3_lmaxT3500_lmaxP4096_nside2048_mh_resp_from_sims_12ests.pkl'
    rdn0_lmaxT3500_cinv_mh_12ests = pickle.load(open(filename,'rb')) # * (l*(l+1))**2/4

    # lmaxT = 3500, RDN0, cinv-style cross-ILC, 12 estimators, no T3
    filename = dir_out+f'/n0/rdn0_249simpairs_healqest_gmv_cinv_noT3_lmaxT3500_lmaxP4096_nside2048_crossilc_twoseds_resp_from_sims_12ests.pkl'
    rdn0_lmaxT3500_cinv_crossilc_12ests = pickle.load(open(filename,'rb')) # * (l*(l+1))**2/4

    rdn0_ratio_9ests_vs_12ests_mh_cinv_lmaxT3500 = rdn0_lmaxT3500_cinv_mh_9ests/rdn0_lmaxT3500_cinv_mh_12ests
    rdn0_ratio_9ests_vs_12ests_crossilc_cinv_lmaxT3500 = rdn0_lmaxT3500_cinv_crossilc_9ests/rdn0_lmaxT3500_cinv_crossilc_12ests

    plt.clf()
    plt.axhline(y=1, color='gray', alpha=0.5, linestyle='--')
    plt.plot(l, rdn0_ratio_9ests_vs_12ests_mh_cinv_lmaxT3500, color='forestgreen', alpha=0.8, linestyle='-',label='MH')
    plt.plot(l, rdn0_ratio_9ests_vs_12ests_crossilc_cinv_lmaxT3500, color='darkorange', alpha=0.8, linestyle='-',label='Cross-ILC')
    plt.title('RDN0 9 ests / 12 ests, lmaxT = 3500, Cinv-Style')
    plt.xlabel('$\ell$')
    plt.legend(loc='upper left', fontsize='small')
    plt.xscale('log')
    plt.xlim(50,3001)
    plt.ylim(0.8,1.2)
    plt.savefig(dir_out+f'/figs/rdn0_comparison_9ests_vs_12ests.png',bbox_inches='tight')
    '''

def get_lensing_bias(config, append_list, cinv=False, rdn0=False, sqe=False, sims=np.arange(250)+1, n0_n1_sims=np.arange(249)+1):
    '''
    Only does total, the TTEETE/TT-only part is deprecated. Also, profhrd doesn't work.
    '''
    lmax = config['lensrec']['Lmax']
    lmin = config['lensrec']['lminT']
    lmaxT = config['lensrec']['lmaxT']
    lmaxP = config['lensrec']['lmaxP']
    nside = config['lensrec']['nside']
    dir_out = config['dir_out']
    l = np.arange(0,lmax+1)
    lbins = np.logspace(np.log10(50),np.log10(3000),20)
    bin_centers = (lbins[:-1] + lbins[1:]) / 2
    digitized = np.digitize(l, lbins)
    profile_file='fg_profiles/TT_srini_mvilc_foreground_residuals.pkl'
    # Theory spectrum
    clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
    ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
    clkk = slpp * (l*(l+1))**2/4
    binned_clkk = [clkk[digitized == i].mean() for i in range(1, len(lbins))]
    # Input kappa
    klm = hp.read_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_spt3g_input_klm_lmax{lmax}.fits')
    input_clkk = hp.alm2cl(klm,klm,lmax=lmax)
    binned_input_clkk = [input_clkk[digitized == i].mean() for i in range(1, len(lbins))]

    # Bias
    bias = np.zeros((len(l),len(append_list)), dtype=np.complex_)
    binned_bias = np.zeros((len(bin_centers),len(append_list)), dtype=np.complex_)
    # Uncertainty saved from before
    binned_uncertainty = np.zeros((len(bin_centers),len(append_list)), dtype=np.complex_)
    # Cross with input
    cross = np.zeros((len(bin_centers),len(append_list)), dtype=np.complex_)

    for j, append in enumerate(append_list):
        append_alt = append[6:]
        print(f'Doing {append_alt}!')
        u = None

        if append == 'agora_standard':
            ests = ['TT', 'EE', 'TE', 'ET', 'TB', 'BT', 'EB', 'BE']
            if sqe or cinv:
                resps = np.zeros((len(l),len(ests)), dtype=np.complex_)
                inv_resps = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
                for i, est in enumerate(ests):
                    if sqe:
                        # Get SQE response
                        resps[:,i] = get_sim_response(est,config,gmv=False,append=append_alt,sims=sims,cinv=False)
                    elif cinv:
                        # GMV response
                        resps[:,i] = get_sim_response(est,config,gmv=True,cinv=True,append=append_alt,sims=sims)
                    inv_resps[1:,i] = 1/(resps)[1:,i]
                resp = np.sum(resps, axis=1)
            else:
                resp = get_sim_response('all',config,gmv=True,append=append_alt,sims=sims,cinv=False)
            inv_resp = np.zeros_like(l,dtype=np.complex_); inv_resp[1:] = 1/(resp)[1:]

        elif append == 'agora_mh' or append == 'agora_crossilc_onesed' or append == 'agora_crossilc_twoseds':
            ests = ['T1T2', 'T2T1', 'EE', 'TE', 'ET', 'TB', 'BT', 'EB', 'BE']
            if sqe or cinv:
                resps = np.zeros((len(l),len(ests)), dtype=np.complex_)
                inv_resps = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
                for i, est in enumerate(ests):
                    if sqe:
                        # Get SQE response
                        resps[:,i] = get_sim_response(est,config,gmv=False,append=append_alt,sims=sims,cinv=False)
                    elif cinv:
                        # GMV response
                        resps[:,i] = get_sim_response(est,config,gmv=True,cinv=True,append=append_alt,sims=sims)
                    inv_resps[1:,i] = 1/(resps)[1:,i]
                resp = 0.5*resps[:,0]+0.5*resps[:,1]+np.sum(resps[:,2:], axis=1)
            else:
                resp = get_sim_response('all',config,gmv=True,append=append_alt,sims=sims,cinv=False)
            inv_resp = np.zeros_like(l,dtype=np.complex_); inv_resp[1:] = 1/(resp)[1:]

        # Get N0 correction (remember these still need (l*(l+1))**2/4 factor to convert to kappa)
        if rdn0:
            if cinv:
                n0 = get_rdn0(sims=n0_n1_sims,qetype='gmv_cinv',config=config,append=append_alt)
            elif sqe:
                n0 = get_rdn0(sims=n0_n1_sims,qetype='sqe',config=config,append=append_alt)
            else:
                n0 = get_rdn0(sims=n0_n1_sims,qetype='gmv',config=config,append=append_alt)
            n0_total = n0 * (l*(l+1))**2/4
        else:
            if cinv:
                n0 = get_n0(sims=n0_n1_sims,qetype='gmv_cinv',config=config,append=append_alt)
            elif sqe:
                n0 = get_n0(sims=n0_n1_sims,qetype='sqe',config=config,append=append_alt)
            else:
                n0 = get_n0(sims=n0_n1_sims,qetype='gmv',config=config,append=append_alt)
            n0_total = n0['total'] * (l*(l+1))**2/4

        # N1
        if cinv:
            n1 = get_n1(sims=n0_n1_sims,qetype='gmv_cinv',config=config,append=append_alt)
        elif sqe:
            n1 = get_n1(sims=n0_n1_sims,qetype='sqe',config=config,append=append_alt)
        else:
            n1 = get_n1(sims=n0_n1_sims,qetype='gmv',config=config,append=append_alt)
        n1_total = n1['total'] * (l*(l+1))**2/4

        if cinv:
            if append == 'agora_mh' or append == 'agora_crossilc_onesed' or append == 'agora_crossilc_twoseds':
                # Load GMV plms, cinv-style
                plms = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_gmv_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_noT3.npy')),len(ests)), dtype=np.complex_)
                for i, est in enumerate(ests):
                    plms[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_noT3.npy')
                plm = 0.5*np.sum(plms[:,:2], axis=1)+np.sum(plms[:,2:], axis=1)
            else:
                # Load GMV plms, cinv-style
                plms = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_gmv_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv.npy')),len(ests)), dtype=np.complex_)
                for i, est in enumerate(ests):
                    # Commented out below: I'm testing the case without NG fg, should give zero for bias
                    #plms[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_1_seed2_1_lmaxT3000_lmaxP4096_nside2048_standard_cinv.npy')
                    plms[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv.npy')
                plm = np.sum(plms, axis=1)
        elif sqe:
            if append == 'agora_mh' or append == 'agora_crossilc_onesed' or append == 'agora_crossilc_twoseds':
                # Load SQE plms
                plms = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_noT3.npy')),len(ests)), dtype=np.complex_)
                for i, est in enumerate(ests):
                    plms[:,i] = np.load(dir_out+f'/plm_{est}_healqest_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_noT3.npy')
                plm = 0.5*plms[:,0]+0.5*plms[:,1]+np.sum(plms[:,2:], axis=1)
            else:
                # Load SQE plms
                plms = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')),len(ests)), dtype=np.complex_)
                for i, est in enumerate(ests):
                    plms[:,i] = np.load(dir_out+f'/plm_{est}_healqest_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
                plm = np.sum(plms, axis=1)
        else:
            # Load GMV plms, not cinv-style
            plm_gmv = np.load(dir_out+f'/plm_all_healqest_gmv_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')

        # Response correct
        plm_resp_corr = hp.almxfl(plm,inv_resp)

        # Get spectra
        auto = hp.alm2cl(plm_resp_corr, plm_resp_corr, lmax=lmax) * (l*(l+1))**2/4

        # Cross with input
        cross_unbinned = hp.alm2cl(klm, plm_resp_corr) * (l*(l+1))/2
        cross[:,j] = [cross_unbinned[digitized == i].mean() for i in range(1, len(lbins))]

        # N0 and N1 subtract
        auto_debiased = auto - n0_total - n1_total

        # Bin!
        binned_auto_debiased = [auto_debiased[digitized == i].mean() for i in range(1, len(lbins))]

        # Get bias
        bias[:,j] = auto_debiased - input_clkk
        #binned_bias[:,j] = [bias[:,j][digitized == i].mean() for i in range(1, len(lbins))]
        binned_bias[:,j] = np.array(binned_auto_debiased) - np.array(binned_input_clkk)

        # Get uncertainty
        #if cinv:
        #    uncertainty_cinv = np.load(dir_out+f'/agora_reconstruction/measurement_uncertainty_lmaxT{lmaxT}_{append_alt}_cinv.npy')[:,0]
        #    uncertainty_sqe = np.load(dir_out+f'/agora_reconstruction/measurement_uncertainty_lmaxT{lmaxT}_{append_alt}.npy')[:,1]
        #    binned_uncertainty_cinv[:,j] = np.load(dir_out+f'/agora_reconstruction/binned_measurement_uncertainty_lmaxT{lmaxT}_{append_alt}_cinv.npy')[:,0]
        #    binned_uncertainty_sqe[:,j] = np.load(dir_out+f'/agora_reconstruction/binned_measurement_uncertainty_lmaxT{lmaxT}_{append_alt}.npy')[:,1]
        #else:
        #    uncertainty_gmv = np.load(dir_out+f'/agora_reconstruction/measurement_uncertainty_lmaxT{lmaxT}_{append_alt}.npy')[:,0]
        #    uncertainty_sqe = np.load(dir_out+f'/agora_reconstruction/measurement_uncertainty_lmaxT{lmaxT}_{append_alt}.npy')[:,1]
        #    binned_uncertainty_gmv[:,j] = np.load(dir_out+f'/agora_reconstruction/binned_measurement_uncertainty_lmaxT{lmaxT}_{append_alt}.npy')[:,0]
        #    binned_uncertainty_sqe[:,j] = np.load(dir_out+f'/agora_reconstruction/binned_measurement_uncertainty_lmaxT{lmaxT}_{append_alt}.npy')[:,1]

        # Smooth the unbinned lines because it's noisy af...
        #B, A = signal.butter(3, 0.1, output='ba')
        #bias_over_uncertainty_gmv[:,j] = signal.filtfilt(B,A,bias_over_uncertainty_gmv[:,j])
        #bias_over_uncertainty_sqe[:,j] = signal.filtfilt(B,A,bias_over_uncertainty_sqe[:,j])
        #bias_over_uncertainty_gmv[:,j] = savitzky_golay(bias_over_uncertainty_gmv[:,j],101,3)
        #bias_over_uncertainty_sqe[:,j] = savitzky_golay(bias_over_uncertainty_sqe[:,j],101,3)
        #bias_over_uncertainty_gmv[:,j] = lfilter([1.0 / 150] * 150, 1, bias_over_uncertainty_gmv[:,j])
        #bias_over_uncertainty_sqe[:,j] = lfilter([1.0 / 150] * 150, 1, bias_over_uncertainty_sqe[:,j])

    return binned_bias

def get_rdn0(sims,qetype,config,append):
    '''
    Only returns total N0, not for each estimator.
    Doing no T3, with T3 not implemented.
    Argument qetype can be 'gmv', 'gmv_cinv', or 'sqe'.
    Not implemented for append == 'profhrd'.
    '''
    lmax = config['lensrec']['Lmax']
    lmin = config['lensrec']['lminT']
    lmaxT = config['lensrec']['lmaxT']
    lmaxP = config['lensrec']['lmaxP']
    nside = config['lensrec']['nside']
    cltype = config['lensrec']['cltype']
    sl = {ee:config['cls'][cltype][ee] for ee in config['cls'][cltype].keys()}
    dir_out = config['dir_out']
    l = np.arange(0,lmax+1)
    num = len(sims)

    if append=='crossilc_twoseds' or append=='crossilc_onesed' or append=='mh':
        filename = dir_out+f'/n0/rdn0_{num}simpairs_healqest_{qetype}_noT3_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_resp_from_sims_9ests.pkl'
    else:
        filename = dir_out+f'/n0/rdn0_{num}simpairs_healqest_{qetype}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_resp_from_sims.pkl'
        #filename = dir_out+f'/n0/fake_rdn0_{num}simpairs_healqest_{qetype}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_resp_from_sims.pkl'

    if os.path.isfile(filename):
        print(f'Getting RDN0: {filename}')
        rdn0 = pickle.load(open(filename,'rb'))

    elif qetype == 'gmv' or qetype == 'gmv_cinv' or qetype == 'sqe':
        if append == 'standard':
            ests = ['TT', 'EE', 'TE', 'ET', 'TB', 'BT', 'EB', 'BE']
        elif append == 'mh' or append == 'crossilc_onesed' or append == 'crossilc_twoseds':
            ests = ['T1T2', 'T2T1', 'EE', 'TE', 'ET', 'TB', 'BT', 'EB', 'BE']

        # Get response
        if qetype == 'gmv':
            resp = get_sim_response('all',config,gmv=True,append=append,sims=np.append(sims,num+1),cinv=False)
        elif qetype == 'gmv_cinv' or qetype == 'sqe':
            resps = np.zeros((len(l),len(ests)), dtype=np.complex_)
            inv_resps = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
            for i, est in enumerate(ests):
                if qetype == 'gmv_cinv':
                    resps[:,i] = get_sim_response(est,config,gmv=True,cinv=True,append=append,sims=np.append(sims,num+1))
                elif qetype == 'sqe':
                    resps[:,i] = get_sim_response(est,config,gmv=False,cinv=False,append=append,sims=np.append(sims,num+1))
                inv_resps[1:,i] = 1/(resps)[1:,i]
            if append == 'standard':
                resp = np.sum(resps, axis=1)
            elif append == 'mh' or append == 'crossilc_onesed' or append == 'crossilc_twoseds':
                resp = 0.5*np.sum(resps[:,:2], axis=1)+np.sum(resps[:,2:], axis=1)
        inv_resp = np.zeros(len(l),dtype=np.complex_); inv_resp[1:] = 1./(resp)[1:]

        # Get sim-based N0
        if qetype == 'gmv':
            n0 = get_n0(sims=sims,qetype='gmv',config=config,append=append)
        elif qetype == 'gmv_cinv':
            n0 = get_n0(sims=sims,qetype='gmv_cinv',config=config,append=append)
        elif qetype == 'sqe':
            n0 = get_n0(sims=sims,qetype='sqe',config=config,append=append)
        n0_total = n0['total']

        rdn0 = 0
        #sims = np.arange(248)+2; num = len(sims)
        for i, sim in enumerate(sims):
            if qetype == 'gmv':
                plm_ir = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
                plm_ri = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            elif qetype == 'gmv_cinv':
                if append == 'standard':
                    # Get ir sims
                    plms_ir = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_gmv_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv.npy')),len(ests)), dtype=np.complex_)
                    #plms_ir = np.zeros((len(np.load(dir_out+f'/outputs_temp/plm_EE_healqest_gmv_seed1_{sim}_seed2_1_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv.npy')),len(ests)), dtype=np.complex_)
                    for i, est in enumerate(ests):
                        plms_ir[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv.npy')
                        #plms_ir[:,i] = np.load(dir_out+f'/outputs_temp/plm_{est}_healqest_gmv_seed1_{sim}_seed2_1_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv.npy')
                    plm_ir = np.sum(plms_ir, axis=1)
                elif append == 'mh' or append == 'crossilc_onesed' or append == 'crossilc_twoseds':
                    # Get ir sims
                    plms_ir = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_gmv_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_noT3.npy')),len(ests)), dtype=np.complex_)
                    for i, est in enumerate(ests):
                        plms_ir[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_noT3.npy')
                    plm_ir = 0.5*np.sum(plms_ir[:,:2], axis=1)+np.sum(plms_ir[:,2:], axis=1)
                if append == 'standard':
                    # Get ri sims
                    plms_ri = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_gmv_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv.npy')),len(ests)), dtype=np.complex_)
                    #plms_ri = np.zeros((len(np.load(dir_out+f'/outputs_temp/plm_EE_healqest_gmv_seed1_1_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv.npy')),len(ests)), dtype=np.complex_)
                    for i, est in enumerate(ests):
                        plms_ri[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv.npy')
                        #plms_ri[:,i] = np.load(dir_out+f'/outputs_temp/plm_{est}_healqest_gmv_seed1_1_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv.npy')
                    plm_ri = np.sum(plms_ri, axis=1)
                elif append == 'mh' or append == 'crossilc_onesed' or append == 'crossilc_twoseds':
                    # Get ri sims
                    plms_ri = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_gmv_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_noT3.npy')),len(ests)), dtype=np.complex_)
                    for i, est in enumerate(ests):
                        plms_ri[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_noT3.npy')
                    plm_ri = 0.5*np.sum(plms_ri[:,:2], axis=1)+np.sum(plms_ri[:,2:], axis=1)
            elif qetype == 'sqe':
                if append == 'standard':
                    # Get ir sims
                    plms_ir = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')),len(ests)), dtype=np.complex_)
                    for i, est in enumerate(ests):
                        plms_ir[:,i] = np.load(dir_out+f'/plm_{est}_healqest_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
                    plm_ir = np.sum(plms_ir, axis=1)
                elif append == 'mh' or append == 'crossilc_onesed' or append == 'crossilc_twoseds':
                    # Get ir sims
                    plms_ir = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_noT3.npy')),len(ests)), dtype=np.complex_)
                    for i, est in enumerate(ests):
                        plms_ir[:,i] = np.load(dir_out+f'/plm_{est}_healqest_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_noT3.npy')
                    plm_ir = 0.5*np.sum(plms_ir[:,:2], axis=1)+np.sum(plms_ir[:,2:], axis=1)
                if append == 'standard':
                    # Get ri sims
                    plms_ri = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')),len(ests)), dtype=np.complex_)
                    for i, est in enumerate(ests):
                        plms_ri[:,i] = np.load(dir_out+f'/plm_{est}_healqest_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
                    plm_ri = np.sum(plms_ri, axis=1)
                elif append == 'mh' or append == 'crossilc_onesed' or append == 'crossilc_twoseds':
                    # Get ri sims
                    plms_ri = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_noT3.npy')),len(ests)), dtype=np.complex_)
                    for i, est in enumerate(ests):
                        plms_ri[:,i] = np.load(dir_out+f'/plm_{est}_healqest_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_noT3.npy')
                    plm_ri = 0.5*np.sum(plms_ri[:,:2], axis=1)+np.sum(plms_ri[:,2:], axis=1)

            if np.any(np.isnan(plm_ir)):
                print(f'Sim {sim} is bad!')
                num -= 1
                continue

            # Response correct
            plm_ir = hp.almxfl(plm_ir,inv_resp)
            plm_ri = hp.almxfl(plm_ri,inv_resp)

            # Get <irir>, <irri>, <riir>, <riri>
            irir = hp.alm2cl(plm_ir,plm_ir,lmax=lmax)
            irri = hp.alm2cl(plm_ir,plm_ri,lmax=lmax)
            riir = hp.alm2cl(plm_ri,plm_ir,lmax=lmax)
            riri = hp.alm2cl(plm_ri,plm_ri,lmax=lmax)

            rdn0 += irir+irri+riir+riri

        rdn0 /= num
        # Subtract the sim-based (<ijij>+<ijji>) N0 that I already have saved
        rdn0 -= n0_total
        with open(filename, 'wb') as f:
            pickle.dump(rdn0, f)

    else:
        print('Invalid argument qetype.')

    return rdn0

def get_n0(sims,qetype,config,append,cmbonly=False):
    '''
    Get N0 bias. qetype should be 'gmv' or 'sqe'.
    Returns dictionary containing keys for each estimator.
    '''
    lmax = config['lensrec']['Lmax']
    lmin = config['lensrec']['lminT']
    lmaxT = config['lensrec']['lmaxT']
    lmaxP = config['lensrec']['lmaxP']
    nside = config['lensrec']['nside']
    cltype = config['lensrec']['cltype']
    sl = {ee:config['cls'][cltype][ee] for ee in config['cls'][cltype].keys()}
    dir_out = config['dir_out']
    l = np.arange(0,lmax+1)
    num = len(sims)
    append_original = append
    if cmbonly:
        append += '_cmbonly'
    if append=='crossilc_twoseds' or append=='crossilc_onesed' or append=='mh':
        filename = dir_out+f'/n0/n0_{num}simpairs_healqest_{qetype}_noT3_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_resp_from_sims_9ests.pkl'
    else:
        filename = dir_out+f'/n0/n0_{num}simpairs_healqest_{qetype}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_resp_from_sims.pkl'
    if os.path.isfile(filename):
        n0 = pickle.load(open(filename,'rb'))
    else:
        print(f"File {filename} doesn't exist!")

    return n0

def get_n1(sims,qetype,config,append):
    '''
    Get N1 bias. qetype should be 'gmv' or 'sqe'.
    Returns dictionary containing keys 'total', 'TTEETE', and 'TBEB' for GMV.
    Similarly for SQE.
    '''
    lmax = config['lensrec']['Lmax']
    lmin = config['lensrec']['lminT']
    lmaxT = config['lensrec']['lmaxT']
    lmaxP = config['lensrec']['lmaxP']
    nside = config['lensrec']['nside']
    cltype = config['lensrec']['cltype']
    sl = {ee:config['cls'][cltype][ee] for ee in config['cls'][cltype].keys()}
    dir_out = config['dir_out']
    l = np.arange(0,lmax+1)
    num = len(sims)
    if append=='crossilc_twoseds' or append=='crossilc_onesed' or append=='mh':
        filename = dir_out+f'/n1/n1_{num}simpairs_healqest_{qetype}_noT3_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_resp_from_sims_9ests.pkl'
    else:
        filename = dir_out+f'/n1/n1_{num}simpairs_healqest_{qetype}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_resp_from_sims.pkl'
    if os.path.isfile(filename):
        n1 = pickle.load(open(filename,'rb'))
    else:
        print(f"File {filename} doesn't exist!")

    return n1

def get_sim_response(est,config,gmv,append,sims,filename=None,cinv=False):
    '''
    Make sure the sims are lensed, not unlensed!
    '''
    lmax = config['lensrec']['Lmax']
    lmaxT = config['lensrec']['lmaxT']
    lmaxP = config['lensrec']['lmaxP']
    lmin = config['lensrec']['lminT']
    nside = config['lensrec']['nside']
    cltype = config['lensrec']['cltype']
    dir_out = config['dir_out']
    l = np.arange(0,lmax+1)
    num = len(sims)
    if filename is None:
        fn = ''
        if gmv and cinv:
            fn += f'_gmv_cinv_est{est}'
        elif gmv:
            fn += f'_gmv_est{est}'
        else:
            fn += f'_sqe_est{est}'
        fn += f'_lmaxT{lmaxT}_lmaxP{lmaxP}_lmin{lmin}_cltype{cltype}_{append}'
        if append=='crossilc_twoseds' or append=='crossilc_onesed' or append=='mh':
            fn += '_noT3'
        filename = dir_out+f'/resp/sim_resp_{num}sims{fn}.npy'

    if os.path.isfile(filename):
        print('Loading from existing file!')
        sim_resp = np.load(filename)
    else:
        print(f"File {filename} doesn't exist!")
    return sim_resp

####################

#analyze()
compare_n0()
