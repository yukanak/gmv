#!/usr/bin/env python3
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

def compare_n0(config_file='test_yuka.yaml', cinv=True):
    '''
    Compare N0 for MH GMV (legs are tSZ-nulled map and MV-ILC map) vs
    cross-ILC GMV (legs are tSZ- and CIB- nulled maps) vs
    profile hardened GMV vs standard GMV.
    All using response from sims, NOT analytic response.
    cinv = False NOT IMPLEMENTED.
    '''
    config = utils.parse_yaml(config_file)
    lmax = config['lensrec']['Lmax']
    lmin = config['lensrec']['lminT']
    lmaxT = config['lensrec']['lmaxT']
    lmaxP = config['lensrec']['lmaxP']
    nside = config['lensrec']['nside']
    dir_out = config['dir_out']
    l = np.arange(0,lmax+1)
    if not cinv:
        print('NOT IMPLEMENTED')
        return None

    #=========================================================================#

    # Full sky, no masking
    # Sims are signal + Agora foregrounds + SPT3G 2019-2020 noise levels frequency correlated noise realizations generated from frequency separated noise spectra

    # Standard SQE
    # lmaxT = 3000
    filename = dir_out+f'/n0/rdn0_249simpairs_healqest_sqe_lmaxT3000_lmaxP4096_nside2048_agora_standard_resp_from_sims.pkl'
    n0_sqe_3000 = pickle.load(open(filename,'rb'))
    n0_sqe_3000_total = n0_sqe_3000 * (l*(l+1))**2/4
    # lmaxT = 3500
    filename = dir_out+f'/n0/rdn0_249simpairs_healqest_sqe_lmaxT3500_lmaxP4096_nside2048_agora_standard_resp_from_sims.pkl'
    n0_sqe_3500 = pickle.load(open(filename,'rb'))
    n0_sqe_3500_total = n0_sqe_3500 * (l*(l+1))**2/4

    # Standard GMV
    # lmaxT = 3000
    filename = dir_out+f'/n0/rdn0_249simpairs_healqest_gmv_cinv_lmaxT3000_lmaxP4096_nside2048_agora_standard_resp_from_sims.pkl'
    n0_standard_3000 = pickle.load(open(filename,'rb'))
    n0_standard_3000_total = n0_standard_3000 * (l*(l+1))**2/4
    # lmaxT = 3500
    filename = dir_out+f'/n0/rdn0_249simpairs_healqest_gmv_cinv_lmaxT3500_lmaxP4096_nside2048_agora_standard_resp_from_sims.pkl'
    n0_standard_3500 = pickle.load(open(filename,'rb'))
    n0_standard_3500_total = n0_standard_3500 * (l*(l+1))**2/4
    # lmaxT = 4000
    filename = dir_out+f'/n0/rdn0_249simpairs_healqest_gmv_cinv_lmaxT4000_lmaxP4096_nside2048_agora_standard_resp_from_sims.pkl'
    n0_standard_4000 = pickle.load(open(filename,'rb'))
    n0_standard_4000_total = n0_standard_4000 * (l*(l+1))**2/4

    # Standard GMV, WEBSKY
    # lmaxT = 3000
    filename = dir_out+f'/n0/rdn0_249simpairs_healqest_gmv_cinv_lmaxT3000_lmaxP4096_nside2048_websky_standard_resp_from_sims.pkl'
    n0_standard_3000_websky = pickle.load(open(filename,'rb'))
    n0_standard_3000_websky_total = n0_standard_3000_websky * (l*(l+1))**2/4
    # lmaxT = 3500
    filename = dir_out+f'/n0/rdn0_249simpairs_healqest_gmv_cinv_lmaxT3500_lmaxP4096_nside2048_websky_standard_resp_from_sims.pkl'
    n0_standard_3500_websky = pickle.load(open(filename,'rb'))
    n0_standard_3500_websky_total = n0_standard_3500_websky * (l*(l+1))**2/4
    # lmaxT = 4000
    filename = dir_out+f'/n0/rdn0_249simpairs_healqest_gmv_cinv_lmaxT4000_lmaxP4096_nside2048_websky_standard_resp_from_sims.pkl'
    n0_standard_4000_websky = pickle.load(open(filename,'rb'))
    n0_standard_4000_websky_total = n0_standard_4000_websky * (l*(l+1))**2/4

    # MH GMV
    # lmaxT = 3500
    filename = dir_out+f'/n0/rdn0_249simpairs_healqest_gmv_cinv_noT3_lmaxT3500_lmaxP4096_nside2048_agora_mh_resp_from_sims_12ests.pkl'
    n0_mh_3500 = pickle.load(open(filename,'rb'))
    n0_mh_3500_total = n0_mh_3500 * (l*(l+1))**2/4
    # lmaxT = 4000
    filename = dir_out+f'/n0/rdn0_249simpairs_healqest_gmv_cinv_noT3_lmaxT4000_lmaxP4096_nside2048_agora_mh_resp_from_sims_12ests.pkl'
    n0_mh_4000 = pickle.load(open(filename,'rb'))
    n0_mh_4000_total = n0_mh_4000 * (l*(l+1))**2/4

    # MH GMV, WEBSKY
    # lmaxT = 3500
    filename = dir_out+f'/n0/rdn0_249simpairs_healqest_gmv_cinv_noT3_lmaxT3500_lmaxP4096_nside2048_websky_mh_resp_from_sims_12ests.pkl'
    n0_mh_3500_websky = pickle.load(open(filename,'rb'))
    n0_mh_3500_websky_total = n0_mh_3500_websky * (l*(l+1))**2/4
    # lmaxT = 4000
    filename = dir_out+f'/n0/rdn0_249simpairs_healqest_gmv_cinv_noT3_lmaxT4000_lmaxP4096_nside2048_websky_mh_resp_from_sims_12ests.pkl'
    n0_mh_4000_websky = pickle.load(open(filename,'rb'))
    n0_mh_4000_websky_total = n0_mh_4000_websky * (l*(l+1))**2/4

    # Cross-ILC GMV, one component
    # lmaxT = 3500
    #filename = dir_out+f'/n0/rdn0_249simpairs_healqest_gmv_cinv_noT3_lmaxT3500_lmaxP4096_nside2048_agora_crossilc_onesed_resp_from_sims_12ests.pkl'
    #n0_crossilc_onesed_3500 = pickle.load(open(filename,'rb'))
    #n0_crossilc_onesed_3500_total = n0_crossilc_onesed_3500 * (l*(l+1))**2/4
    # lmaxT = 4000
    #filename = dir_out+f'/n0/rdn0_249simpairs_healqest_gmv_cinv_noT3_lmaxT4000_lmaxP4096_nside2048_agora_crossilc_onesed_resp_from_sims_12ests.pkl'
    #n0_crossilc_onesed_4000 = pickle.load(open(filename,'rb'))
    #n0_crossilc_onesed_4000_total = n0_crossilc_onesed_4000 * (l*(l+1))**2/4

    # Cross-ILC GMV, one component, WEBSKY
    # lmaxT = 3500
    filename = dir_out+f'/n0/rdn0_249simpairs_healqest_gmv_cinv_noT3_lmaxT3500_lmaxP4096_nside2048_websky_crossilc_onesed_resp_from_sims_12ests.pkl'
    n0_crossilc_onesed_3500_websky = pickle.load(open(filename,'rb'))
    n0_crossilc_onesed_3500_websky_total = n0_crossilc_onesed_websky_3500 * (l*(l+1))**2/4
    # lmaxT = 4000
    filename = dir_out+f'/n0/rdn0_249simpairs_healqest_gmv_cinv_noT3_lmaxT4000_lmaxP4096_nside2048_websky_crossilc_onesed_resp_from_sims_12ests.pkl'
    n0_crossilc_onesed_4000_websky = pickle.load(open(filename,'rb'))
    n0_crossilc_onesed_4000_websky_total = n0_crossilc_onesed_websky_4000 * (l*(l+1))**2/4

    # Cross-ILC GMV, two component
    # lmaxT = 3500
    filename = dir_out+f'/n0/rdn0_249simpairs_healqest_gmv_cinv_noT3_lmaxT3500_lmaxP4096_nside2048_agora_crossilc_twoseds_resp_from_sims_12ests.pkl'
    n0_crossilc_twoseds_3500 = pickle.load(open(filename,'rb'))
    n0_crossilc_twoseds_3500_total = n0_crossilc_twoseds_3500 * (l*(l+1))**2/4
    # lmaxT = 4000
    filename = dir_out+f'/n0/rdn0_249simpairs_healqest_gmv_cinv_noT3_lmaxT4000_lmaxP4096_nside2048_agora_crossilc_twoseds_resp_from_sims_12ests.pkl'
    n0_crossilc_twoseds_4000 = pickle.load(open(filename,'rb'))
    n0_crossilc_twoseds_4000_total = n0_crossilc_twoseds_4000 * (l*(l+1))**2/4

    # Profile hardened GMV

    # Theory spectrum
    clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
    ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
    clkk = slpp * (l*(l+1))**2/4

    # Plot
    plt.clf()
    plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')
    plt.plot(l, n0_mh_3500_total, color='forestgreen', alpha=0.8, linestyle='-',label='Gradient Cleaning GMV, Agora')
    #plt.plot(l, n0_crossilc_onesed_total, color='goldenrod', alpha=0.8, linestyle='-',label='Cross-ILC GMV (one component CIB), total')
    #plt.plot(l, n0_crossilc_twoseds_total, color='darkorange', alpha=0.8, linestyle='-',label='Cross-ILC GMV (two component CIB), total')
    plt.plot(l, n0_crossilc_twoseds_3500_total, color='darkorange', alpha=0.8, linestyle='-',label='Cross-ILC GMV, Agora')
    #plt.plot(l, n0_profhrd_3500_total, color='plum', alpha=0.8, linestyle='-',label='Profile Hardened GMV')
    plt.plot(l, n0_sqe_3500_total, color='firebrick', alpha=0.8, linestyle='-',label='Standard SQE, Agora')
    plt.plot(l, n0_standard_3500_total, color='darkblue', alpha=0.8, linestyle='-',label='Standard GMV, Agora')
    plt.plot(l, n0_mh_3500_websky_total, color='lightgreen', alpha=0.8, linestyle='--',label='Gradient Cleaning GMV, WebSky')
    plt.plot(l, n0_crossilc_onesed_3500_websky_total, color='bisque', alpha=0.8, linestyle='--',label='Cross-ILC GMV, WebSky')
    plt.plot(l, n0_standard_3500_websky_total, color='cornflowerblue', alpha=0.8, linestyle='--',label='Standard GMV, WebSky')
    plt.ylabel("$[\ell(\ell+1)]^2$$N_0$ / 4 $[\mu K^2]$")
    plt.xlabel('$\ell$')
    plt.title(f'GMV Reconstruction Noise Comparison, lmaxT = 3500')
    plt.legend(loc='upper left', fontsize='small')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,lmax)
    plt.ylim(1e-8,1e-7)
    plt.savefig(dir_out+f'/figs/n0_comparison_gmv_lmaxT3500.png',bbox_inches='tight')

    plt.clf()
    #ratio_n0_crossilc_onesed_3500_total = n0_crossilc_onesed_3500_total/n0_standard_3500_total
    ratio_n0_crossilc_twoseds_3500_total = n0_crossilc_twoseds_3500_total/n0_standard_3500_total
    ratio_n0_mh_3500_total = n0_mh_3500_total/n0_standard_3500_total
    #ratio_n0_profhrd_3500_total = n0_profhrd_3500_total/n0_standard_3500_total
    ratio_n0_sqe_3500_total = n0_sqe_3500_total/n0_standard_3500_total
    ratio_n0_crossilc_onesed_3500_websky_total = n0_crossilc_onesed_3500_websky_total/n0_standard_3500_websky_total
    ratio_n0_mh_3500_websky_total = n0_mh_3500_websky_total/n0_standard_3500_websky_total
    # Ratios with error bars
    plt.axhline(y=1, color='k', linestyle='--')
    plt.plot(l, ratio_n0_sqe_3500_total, color='firebrick', alpha=0.8, linestyle='-',label='Ratio Standard SQE / Standard GMV (Agora)')
    plt.plot(l, ratio_n0_mh_3500_total, color='forestgreen', alpha=0.8, linestyle='-',label='Ratio Gradient Cleaning GMV / Standard GMV (Agora)')
    plt.plot(l, ratio_n0_crossilc_twoseds_3500_total, color='darkorange', alpha=0.8, linestyle='-',label='Ratio Cross-ILC GMV / Standard GMV (Agora)')
    #plt.plot(l, ratio_n0_crossilc_onesed_3500_total,color='goldenrod', alpha=0.8, linestyle='-',label='Ratio Cross-ILC GMV (one component CIB) / Standard GMV')
    #plt.plot(l, ratio_n0_crossilc_twoseds_3500_total, color='darkorange', alpha=0.8, linestyle='-',label='Ratio Cross-ILC GMV (two component CIB) / Standard GMV')
    #plt.plot(l, ratio_n0_profhrd_3500_total, color='plum', alpha=0.8, linestyle='-',label='Ratio Profile Hardened GMV / Standard GMV')
    plt.plot(l, ratio_n0_mh_3500_websky_total, color='lightgreen', alpha=0.8, linestyle='--',label='Ratio Gradient Cleaning GMV / Standard GMV (WebSky)')
    plt.plot(l, ratio_n0_crossilc_onesed_3500_websky_total, color='bisque', alpha=0.8, linestyle='--',label='Ratio Cross-ILC GMV / Standard GMV (WebSky)')
    plt.xlabel('$\ell$')
    plt.title(f'GMV Reconstruction Noise Comparison, lmaxT = 3500')
    plt.legend(loc='upper left', fontsize='small')
    plt.xscale('log')
    plt.ylim(0.9,1.4)
    plt.xlim(10,lmax)
    plt.savefig(dir_out+f'/figs/n0_comparison_gmv_lmaxT3500_ratio.png',bbox_inches='tight')

    plt.clf()
    plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')
    # AGORA
    plt.plot(l, n0_sqe_3000_total, color='firebrick', alpha=0.8, linestyle='-',label='Standard SQE, lmaxT = 3000')
    plt.plot(l, n0_sqe_3500_total, color='pink', alpha=0.8, linestyle='-',label='Standard SQE, lmaxT = 3500')
    plt.plot(l, n0_standard_3000_total, color='darkblue', alpha=0.8, linestyle='-',label='Standard GMV, lmaxT = 3000')
    plt.plot(l, n0_standard_3500_total, color='cornflowerblue', alpha=0.8, linestyle='-',label='Standard GMV, lmaxT = 3500')
    plt.plot(l, n0_standard_4000_total, color='lightsteelblue', alpha=0.8, linestyle='-',label='Standard GMV, lmaxT = 4000')
    plt.plot(l, n0_mh_3500_total, color='forestgreen', alpha=0.8, linestyle='-',label='Gradient Cleaning GMV, lmaxT = 3500')
    plt.plot(l, n0_mh_4000_total, color='lightgreen', alpha=0.8, linestyle='-',label='Gradient Cleaning GMV, lmaxT = 4000')
    plt.plot(l, n0_crossilc_twoseds_3500_total, color='darkorange', alpha=0.8, linestyle='-',label='Cross-ILC GMV, lmaxT = 3500')
    plt.plot(l, n0_crossilc_twoseds_4000_total, color='bisque', alpha=0.8, linestyle='-',label='Cross-ILC GMV, lmaxT = 4000')
    plt.ylabel("$[\ell(\ell+1)]^2$$N_0$ / 4 $[\mu K^2]$")
    plt.xlabel('$\ell$')
    plt.title(f'GMV Reconstruction Noise Comparison (Agora)')
    plt.legend(loc='upper left', fontsize='small')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,lmax)
    plt.ylim(1e-8,1e-7)
    plt.savefig(dir_out+f'/figs/n0_comparison_gmv_different_lmaxT_agora.png',bbox_inches='tight')

    plt.clf()
    plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')
    # WEBSKY
    plt.plot(l, n0_standard_3000_websky_total, color='darkblue', alpha=0.8, linestyle='-',label='Standard GMV, lmaxT = 3000')
    plt.plot(l, n0_standard_3500_websky_total, color='cornflowerblue', alpha=0.8, linestyle='-',label='Standard GMV, lmaxT = 3500')
    plt.plot(l, n0_standard_4000_websky_total, color='lightsteelblue', alpha=0.8, linestyle='-',label='Standard GMV, lmaxT = 4000')
    plt.plot(l, n0_mh_3500_websky_total, color='forestgreen', alpha=0.8, linestyle='-',label='Gradient Cleaning GMV, lmaxT = 3500')
    plt.plot(l, n0_mh_4000_websky_total, color='lightgreen', alpha=0.8, linestyle='-',label='Gradient Cleaning GMV, lmaxT = 4000')
    plt.plot(l, n0_crossilc_onesed_3500_websky_total, color='darkorange', alpha=0.8, linestyle='-',label='Cross-ILC GMV, lmaxT = 3500')
    plt.plot(l, n0_crossilc_onesed_4000_websky_total, color='bisque', alpha=0.8, linestyle='-',label='Cross-ILC GMV, lmaxT = 4000')
    plt.ylabel("$[\ell(\ell+1)]^2$$N_0$ / 4 $[\mu K^2]$")
    plt.xlabel('$\ell$')
    plt.title(f'GMV Reconstruction Noise Comparison (WebSky)')
    plt.legend(loc='upper left', fontsize='small')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,lmax)
    plt.ylim(1e-8,1e-7)
    plt.savefig(dir_out+f'/figs/n0_comparison_gmv_different_lmaxT_websky.png',bbox_inches='tight')

    plt.clf()
    # AGORA
    #ratio_n0_crossilc_onesed_3500_total = n0_crossilc_onesed_3500_total/n0_standard_3500_total
    ratio_n0_crossilc_twoseds_3500_total = n0_crossilc_twoseds_3500_total/n0_standard_3500_total
    ratio_n0_crossilc_twoseds_4000_total = n0_crossilc_twoseds_4000_total/n0_standard_4000_total
    ratio_n0_mh_3500_total = n0_mh_3500_total/n0_standard_3500_total
    ratio_n0_mh_4000_total = n0_mh_4000_total/n0_standard_4000_total
    #ratio_n0_profhrd_3500_total = n0_profhrd_3500_total/n0_standard_3500_total
    ratio_n0_sqe_3000_total = n0_sqe_3000_total/n0_standard_3000_total
    ratio_n0_sqe_3500_total = n0_sqe_3500_total/n0_standard_3500_total
    # Ratios with error bars
    plt.axhline(y=1, color='k', linestyle='--')
    plt.plot(l, ratio_n0_sqe_3000_total, color='firebrick', alpha=0.8, linestyle='-',label='Ratio Standard SQE (lmaxT = 3000) / Standard GMV (lmaxT = 3000)')
    plt.plot(l, ratio_n0_sqe_3500_total, color='pink', alpha=0.8, linestyle='-',label='Ratio Standard SQE (lmaxT = 3500) / Standard GMV (lmaxT = 3500)')
    plt.plot(l, ratio_n0_mh_3500_total, color='forestgreen', alpha=0.8, linestyle='-',label='Ratio Gradient Cleaning GMV (lmaxT = 3500) / Standard GMV (lmaxT = 3500)')
    plt.plot(l, ratio_n0_mh_4000_total, color='lightgreen', alpha=0.8, linestyle='-',label='Ratio Gradient Cleaning GMV (lmaxT = 4000) / Standard GMV (lmaxT = 4000)')
    plt.plot(l, ratio_n0_crossilc_twoseds_3500_total, color='darkorange', alpha=0.8, linestyle='-',label='Ratio Cross-ILC GMV (lmaxT = 3500) / Standard GMV (lmaxT = 3500)')
    plt.plot(l, ratio_n0_crossilc_twoseds_4000_total, color='bisque', alpha=0.8, linestyle='-',label='Ratio Cross-ILC GMV (lmaxT = 4000) / Standard GMV (lmaxT = 4000)')
    plt.xlabel('$\ell$')
    plt.title(f'GMV Reconstruction Noise Comparison (Agora)')
    plt.legend(loc='lower left', fontsize='small')
    plt.xscale('log')
    plt.ylim(0.5,1.5)
    plt.xlim(10,lmax)
    plt.savefig(dir_out+f'/figs/n0_comparison_gmv_different_lmaxT_agora_ratio.png',bbox_inches='tight')

    plt.clf()
    # WEBSKY
    ratio_n0_crossilc_onesed_3500_websky_total = n0_crossilc_onesed_3500_websky_total/n0_standard_3500_websky_total
    ratio_n0_crossilc_onesed_4000_websky_total = n0_crossilc_onesed_4000_websky_total/n0_standard_4000_websky_total
    ratio_n0_mh_3500_websky_total = n0_mh_3500_websky_total/n0_standard_3500_websky_total
    ratio_n0_mh_4000_websky_total = n0_mh_4000_websky_total/n0_standard_4000_websky_total
    # Ratios with error bars
    plt.axhline(y=1, color='k', linestyle='--')
    plt.plot(l, ratio_n0_mh_3500_websky_total, color='forestgreen', alpha=0.8, linestyle='-',label='Ratio Gradient Cleaning GMV (lmaxT = 3500) / Standard GMV (lmaxT = 3500)')
    plt.plot(l, ratio_n0_mh_4000_websky_total, color='lightgreen', alpha=0.8, linestyle='-',label='Ratio Gradient Cleaning GMV (lmaxT = 4000) / Standard GMV (lmaxT = 4000)')
    plt.plot(l, ratio_n0_crossilc_onesed_3500_websky_total, color='darkorange', alpha=0.8, linestyle='-',label='Ratio Cross-ILC GMV (lmaxT = 3500)/ Standard GMV (lmaxT = 3500)')
    plt.plot(l, ratio_n0_crossilc_onesed_4000_websky_total, color='bisque', alpha=0.8, linestyle='-',label='Ratio Cross-ILC GMV (lmaxT = 4000) / Standard GMV (lmaxT = 4000)')
    plt.xlabel('$\ell$')
    plt.title(f'GMV Reconstruction Noise Comparison (WebSky)')
    plt.legend(loc='lower left', fontsize='small')
    plt.xscale('log')
    plt.ylim(0.5,1.5)
    plt.xlim(10,lmax)
    plt.savefig(dir_out+f'/figs/n0_comparison_gmv_different_lmaxT_websky_ratio.png',bbox_inches='tight')

compare_n0()
