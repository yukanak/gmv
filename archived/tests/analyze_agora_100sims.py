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

def analyze():
    '''
    Compare with N0/N1 subtraction.
    '''
    lmax = 4096
    l = np.arange(0,lmax+1)
    lbins = np.logspace(np.log10(50),np.log10(3000),20)
    bin_centers = (lbins[:-1] + lbins[1:]) / 2
    digitized = np.digitize(l, lbins)
    # Input kappa
    klm = hp.read_alm('/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_spt3g_input_klm_lmax4096.fits')
    input_clkk = hp.alm2cl(klm,klm,lmax=lmax)
    binned_input_clkk = np.array([input_clkk[digitized == i].mean() for i in range(1, len(lbins))])

    # First, lmaxT = 3000 cases
    config_file='test_yuka.yaml'
    config = utils.parse_yaml(config_file)
    #append_list = ['agora_standard', 'agora_mh', 'agora_crossilc_onesed', 'agora_crossilc_twoseds', 'agora_profhrd']
    append_list = ['agora_standard', 'agora_mh', 'agora_crossilc_twoseds']
    binned_bias_gmv_3000, binned_bias_sqe_3000 = get_lensing_bias(config,append_list,cinv=False)

    # lmaxT = 3000, cinv
    config_file='test_yuka.yaml'
    config = utils.parse_yaml(config_file)
    #append_list = ['agora_standard', 'agora_mh', 'agora_crossilc_onesed', 'agora_crossilc_twoseds', 'agora_profhrd']
    append_list = ['agora_standard', 'agora_mh', 'agora_crossilc_twoseds', 'agora_crossilc_onesed']
    binned_bias_gmv_3000_cinv, binned_bias_sqe_3000_cinv = get_lensing_bias(config,append_list,cinv=True)

    # lmaxT = 3500 cases, cinv
    config_file='test_yuka_lmaxT3500.yaml'
    config = utils.parse_yaml(config_file)
    append_list = ['agora_standard', 'agora_mh', 'agora_crossilc_twoseds', 'agora_crossilc_onesed']
    binned_bias_gmv_3500_cinv, binned_bias_sqe_3500_cinv = get_lensing_bias(config,append_list,cinv=True)

    # lmaxT = 4000 cases, cinv
    config_file='test_yuka_lmaxT4000.yaml'
    config = utils.parse_yaml(config_file)
    append_list = ['agora_standard', 'agora_mh', 'agora_crossilc_twoseds', 'agora_crossilc_onesed']
    binned_bias_gmv_4000_cinv, binned_bias_sqe_4000_cinv = get_lensing_bias(config,append_list,cinv=True)

    # lmaxT = 3000, rdn0 and cinv
    config_file='test_yuka.yaml'
    config = utils.parse_yaml(config_file)
    append_list = ['agora_standard']
    binned_bias_gmv_3000_rdn0, binned_bias_sqe_3000_rdn0 = get_lensing_bias(config,append_list,rdn0=True,cinv=True)

    dir_out = config['dir_out']

    # Plot
    plt.figure(0)
    plt.clf()
    #plt.plot(l, bias_gmv[:,2]/input_clkk, color='goldenrod', linestyle='-', label="Cross-ILC GMV (one component CIB)")
    #plt.plot(l, bias_gmv[:,3]/input_clkk, color='darkorange', linestyle='-', label="Cross-ILC GMV (two component CIB)")
    #plt.plot(l, bias_gmv[:,1]/input_clkk, color='forestgreen', linestyle='-', label="MH GMV")
    #plt.plot(l, bias_gmv[:,4]/input_clkk, color='plum', linestyle='-', label="Profile Hardened GMV")
    #plt.plot(l, bias_sqe[:,0]/input_clkk, color='firebrick', linestyle='-', label=f'Standard SQE')
    #plt.plot(l, bias_gmv[:,0]/input_clkk, color='darkblue', linestyle='-', label="Standard GMV")
    plt.axhline(y=0, color='gray', alpha=0.5, linestyle='--')

    #plt.plot(bin_centers[:-2], binned_bias_gmv_3000[:-2,]/binned_input_clkk[:-2], color='goldenrod', marker='o', linestyle='-', ms=3, alpha=0.8, label="Cross-ILC GMV (one component CIB)")
    #plt.plot(bin_centers[:-2], binned_bias_gmv_3000[:-2,]/binned_input_clkk[:-2], color='darkorange', marker='o', linestyle='-', ms=3, alpha=0.8, label="Cross-ILC GMV (two component CIB)")
    plt.plot(bin_centers[:-2], binned_bias_gmv_3000[:-2,2]/binned_input_clkk[:-2], color='darkorange', marker='o', linestyle='-', ms=3, alpha=0.8, label="Cross-ILC GMV")
    plt.plot(bin_centers[:-2], binned_bias_gmv_3000[:-2,1]/binned_input_clkk[:-2], color='forestgreen', marker='o', linestyle='-', ms=3, alpha=0.8, label="MH GMV")
    #plt.plot(bin_centers[:-2], binned_bias_gmv_3000[:-2,4]/binned_input_clkk[:-2], color='plum', marker='o', linestyle='-', ms=3, alpha=0.8, label="Profile Hardened GMV")
    plt.plot(bin_centers[:-2], binned_bias_sqe_3000[:-2,0]/binned_input_clkk[:-2], color='firebrick', marker='o', linestyle='-', ms=3, alpha=0.8, label=f'Standard SQE')
    plt.plot(bin_centers[:-2], binned_bias_gmv_3000[:-2,0]/binned_input_clkk[:-2], color='darkblue', marker='o', linestyle='-', ms=3, alpha=0.8, label="Standard GMV")

    #plt.errorbar(bin_centers, binned_bias_gmv[:,2]/binned_input_clkk, yerr=binned_uncertainty_gmv[:,2]/binned_input_clkk, color='goldenrod', marker='o', linestyle='-', ms=3, alpha=0.8, label="Cross-ILC GMV (one component CIB)")
    #plt.errorbar(bin_centers, binned_bias_gmv[:,3]/binned_input_clkk, yerr=binned_uncertainty_gmv[:,3]/binned_input_clkk, color='darkorange', marker='o', linestyle='-', ms=3, alpha=0.8, label="Cross-ILC GMV (two component CIB)")
    #plt.errorbar(bin_centers, binned_bias_gmv[:,1]/binned_input_clkk, yerr=binned_uncertainty_gmv[:,1]/binned_input_clkk, color='forestgreen', marker='o', linestyle='-', ms=3, alpha=0.8, label="MH GMV")
    #plt.errorbar(bin_centers, binned_bias_gmv[:,4]/binned_input_clkk, yerr=binned_uncertainty_gmv[:,4]/binned_input_clkk, color='plum', marker='o', linestyle='-', ms=3, alpha=0.8, label="Profile Hardened GMV")
    #plt.errorbar(bin_centers, binned_bias_sqe[:,0]/binned_input_clkk, yerr=binned_uncertainty_sqe[:,0]/binned_input_clkk, color='firebrick', marker='o', linestyle='-', ms=3, alpha=0.8, label=f'Standard SQE')
    #plt.errorbar(bin_centers, binned_bias_gmv[:,0]/binned_input_clkk, yerr=binned_uncertainty_gmv[:,0]/binned_input_clkk, color='darkblue', marker='o', linestyle='-', ms=3, alpha=0.8, label="Standard GMV")

    #plt.ylabel("$C_\ell^{\kappa\kappa}$")
    plt.xlabel('$\ell$')
    plt.title(f'Lensing Bias from Agora Sim / Input Kappa Spectrum')
    plt.legend(loc='upper left', fontsize='small')
    plt.xscale('log')
    plt.xlim(50,2001)
    #plt.xlim(50,3001)
    #plt.xlim(10,lmax)
    plt.ylim(-0.2,0.2)
    plt.savefig(dir_out+f'/figs/bias_total.png',bbox_inches='tight')

    plt.clf()
    plt.axhline(y=0, color='gray', alpha=0.5, linestyle='--')

    plt.plot(bin_centers[:-2], binned_bias_gmv_3000_cinv[:-2,2]/binned_input_clkk[:-2], color='darkorange', marker='o', linestyle='-', ms=3, alpha=0.8, label="Cross-ILC GMV")
    plt.plot(bin_centers[:-2], binned_bias_gmv_3000_cinv[:-2,1]/binned_input_clkk[:-2], color='forestgreen', marker='o', linestyle='-', ms=3, alpha=0.8, label="MH GMV")
    plt.plot(bin_centers[:-2], binned_bias_sqe_3000_cinv[:-2,0]/binned_input_clkk[:-2], color='firebrick', marker='o', linestyle='-', ms=3, alpha=0.8, label=f'Standard SQE')
    plt.plot(bin_centers[:-2], binned_bias_gmv_3000_cinv[:-2,0]/binned_input_clkk[:-2], color='darkblue', marker='o', linestyle='-', ms=3, alpha=0.8, label="Standard GMV")

    plt.xlabel('$\ell$')
    plt.title(f'Lensing Bias from Agora Sim / Input Kappa Spectrum, Cinv Method for GMV')
    plt.legend(loc='upper left', fontsize='small')
    plt.xscale('log')
    plt.xlim(50,2001)
    #plt.xlim(50,3001)
    #plt.xlim(10,lmax)
    plt.ylim(-0.2,0.2)
    plt.savefig(dir_out+f'/figs/bias_total_cinv.png',bbox_inches='tight')

    plt.clf()
    plt.axhline(y=0, color='gray', alpha=0.5, linestyle='--')
    plt.plot(bin_centers[:-2], binned_bias_gmv_3000_cinv[:-2,2]/binned_input_clkk[:-2], color='darkorange', marker='o', linestyle='-', ms=3, alpha=0.8, label="Cross-ILC GMV, lmaxT = 3000")
    #plt.plot(bin_centers[:-2], binned_bias_gmv_3000_cinv[:-2,3]/binned_input_clkk[:-2], color='darkorange', marker='o', linestyle='-', ms=3, alpha=0.8, label="Cross-ILC GMV (ONE SED), lmaxT = 3000")
    plt.plot(bin_centers[:-2], binned_bias_gmv_3000_cinv[:-2,1]/binned_input_clkk[:-2], color='forestgreen', marker='o', linestyle='-', ms=3, alpha=0.8, label="MH GMV, lmaxT = 3000")
    plt.plot(bin_centers[:-2], binned_bias_gmv_3000_cinv[:-2,0]/binned_input_clkk[:-2], color='darkblue', marker='o', linestyle='-', ms=3, alpha=0.8, label="Standard GMV, lmaxT = 3000")
    plt.plot(bin_centers[:-2], binned_bias_sqe_3000_cinv[:-2,0]/binned_input_clkk[:-2], color='firebrick', marker='o', linestyle='-', ms=3, alpha=0.8, label=f'Standard SQE, lmaxT = 3000')

    plt.plot(bin_centers[:-2], binned_bias_gmv_3500_cinv[:-2,2]/binned_input_clkk[:-2], color='orange', marker='o', linestyle='--', ms=3, alpha=0.8, label="Cross-ILC GMV, lmaxT = 3500")
    plt.plot(bin_centers[:-2], binned_bias_gmv_4000_cinv[:-2,2]/binned_input_clkk[:-2], color='bisque', marker='o', linestyle=':', ms=3, alpha=0.8, label="Cross-ILC GMV, lmaxT = 4000")
    #plt.plot(bin_centers[:-2], binned_bias_gmv_3500_cinv[:-2,3]/binned_input_clkk[:-2], color='orange', marker='o', linestyle='--', ms=3, alpha=0.8, label="Cross-ILC GMV (ONE SED), lmaxT = 3500")
    #plt.plot(bin_centers[:-2], binned_bias_gmv_4000_cinv[:-2,3]/binned_input_clkk[:-2], color='bisque', marker='o', linestyle=':', ms=3, alpha=0.8, label="Cross-ILC GMV (ONE SED), lmaxT = 4000")

    plt.plot(bin_centers[:-2], binned_bias_gmv_3500_cinv[:-2,1]/binned_input_clkk[:-2], color='mediumseagreen', marker='o', linestyle='--', ms=3, alpha=0.8, label="MH GMV, lmaxT = 3500")
    plt.plot(bin_centers[:-2], binned_bias_gmv_4000_cinv[:-2,1]/binned_input_clkk[:-2], color='lightgreen', marker='o', linestyle=':', ms=3, alpha=0.8, label="MH GMV, lmaxT = 4000")

    plt.plot(bin_centers[:-2], binned_bias_gmv_3500_cinv[:-2,0]/binned_input_clkk[:-2], color='cornflowerblue', marker='o', linestyle='--', ms=3, alpha=0.8, label="Standard GMV, lmaxT = 3500")
    plt.plot(bin_centers[:-2], binned_bias_gmv_4000_cinv[:-2,0]/binned_input_clkk[:-2], color='lightsteelblue', marker='o', linestyle=':', ms=3, alpha=0.8, label="Standard GMV, lmaxT = 4000")
    plt.xlabel('$\ell$')
    plt.title(f'Lensing Bias from Agora Sim / Input Kappa Spectrum, Cinv Method for GMV')
    plt.legend(loc='upper left', fontsize='small')
    plt.xscale('log')
    plt.xlim(50,2001)
    plt.ylim(-0.2,0.2)
    plt.savefig(dir_out+f'/figs/bias_total_different_lmax.png',bbox_inches='tight')

    plt.clf()
    plt.axhline(y=0, color='gray', alpha=0.5, linestyle='--')

    plt.plot(bin_centers[:-2], binned_bias_sqe_3000_rdn0[:-2,0]/binned_input_clkk[:-2], color='firebrick', marker='o', linestyle='-', ms=3, alpha=0.8, label=f'Standard SQE')
    plt.plot(bin_centers[:-2], binned_bias_gmv_3000_rdn0[:-2,0]/binned_input_clkk[:-2], color='darkblue', marker='o', linestyle='-', ms=3, alpha=0.8, label="Standard GMV")

    plt.xlabel('$\ell$')
    #TODO
    #plt.title(f'Lensing Bias from Agora Sim / Input Kappa Spectrum, RDN0')
    plt.title(f'Lensing Bias from Agora Sim / Input Kappa Spectrum, RDN0 with SIM 1 AS RD')
    plt.legend(loc='upper left', fontsize='small')
    plt.xscale('log')
    plt.xlim(50,2001)
    #plt.xlim(50,3001)
    #plt.xlim(10,lmax)
    plt.ylim(-0.2,0.2)
    plt.savefig(dir_out+f'/figs/bias_total_rdn0.png',bbox_inches='tight')

'''
    plt.clf()
    #plt.plot(l, bias_gmv_TTEETE[:,2]/input_clkk, color='goldenrod', linestyle='-', label="Cross-ILC GMV (one component CIB)")
    #plt.plot(l, bias_gmv_TTEETE[:,3]/input_clkk, color='darkorange', linestyle='-', label="Cross-ILC GMV (two component CIB)")
    #plt.plot(l, bias_gmv_TTEETE[:,1]/input_clkk, color='forestgreen', linestyle='-', label="MH GMV")
    #plt.plot(l, bias_gmv_TTEETE[:,4]/input_clkk, color='plum', linestyle='-', label="Profile Hardened GMV")
    #plt.plot(l, bias_sqe_TT[:,0]/input_clkk, color='firebrick', linestyle='-', label=f'Standard SQE')
    #plt.plot(l, bias_gmv_TTEETE[:,0]/input_clkk, color='darkblue', linestyle='-', label="Standard GMV")
    plt.axhline(y=0, color='gray', alpha=0.5, linestyle='--')

    plt.plot(bin_centers, binned_bias_gmv_TTEETE[:,2]/binned_input_clkk, color='goldenrod', marker='o', linestyle='-', ms=3, alpha=0.8, label="Cross-ILC GMV (one component CIB)")
    plt.plot(bin_centers, binned_bias_gmv_TTEETE[:,3]/binned_input_clkk, color='darkorange', marker='o', linestyle='-', ms=3, alpha=0.8, label="Cross-ILC GMV (two component CIB)")
    plt.plot(bin_centers, binned_bias_gmv_TTEETE[:,1]/binned_input_clkk, color='forestgreen', marker='o', linestyle='-', ms=3, alpha=0.8, label="MH GMV")
    plt.plot(bin_centers, binned_bias_gmv_TTEETE[:,4]/binned_input_clkk, color='plum', marker='o', linestyle='-', ms=3, alpha=0.8, label="Profile Hardened GMV")
    plt.plot(bin_centers, binned_bias_sqe_TT[:,0]/binned_input_clkk, color='firebrick', marker='o', linestyle='-', ms=3, alpha=0.8, label=f'Standard SQE')
    plt.plot(bin_centers, binned_bias_gmv_TTEETE[:,0]/binned_input_clkk, color='darkblue', marker='o', linestyle='-', ms=3, alpha=0.8, label="Standard GMV")

    #plt.errorbar(bin_centers, binned_bias_gmv_TTEETE[:,2]/binned_input_clkk, yerr=binned_uncertainty_gmv_TTEETE[:,2]/binned_input_clkk, color='goldenrod', marker='o', linestyle='-', ms=3, alpha=0.8, label="Cross-ILC GMV (one component CIB)")
    #plt.errorbar(bin_centers, binned_bias_gmv_TTEETE[:,3]/binned_input_clkk, yerr=binned_uncertainty_gmv_TTEETE[:,3]/binned_input_clkk, color='darkorange', marker='o', linestyle='-', ms=3, alpha=0.8, label="Cross-ILC GMV (two component CIB)")
    #plt.errorbar(bin_centers, binned_bias_gmv_TTEETE[:,1]/binned_input_clkk, yerr=binned_uncertainty_gmv_TTEETE[:,1]/binned_input_clkk, color='forestgreen', marker='o', linestyle='-', ms=3, alpha=0.8, label="MH GMV")
    #plt.errorbar(bin_centers, binned_bias_gmv_TTEETE[:,4]/binned_input_clkk, yerr=binned_uncertainty_gmv_TTEETE[:,4]/binned_input_clkk, color='plum', marker='o', linestyle='-', ms=3, alpha=0.8, label="Profile Hardened GMV")
    #plt.errorbar(bin_centers, binned_bias_sqe_TT[:,0]/binned_input_clkk, yerr=binned_uncertainty_sqe_TT[:,0]/binned_input_clkk, color='firebrick', marker='o', linestyle='-', ms=3, alpha=0.8, label=f'Standard SQE')
    #plt.errorbar(bin_centers, binned_bias_gmv_TTEETE[:,0]/binned_input_clkk, yerr=binned_uncertainty_gmv_TTEETE[:,0]/binned_input_clkk, color='darkblue', marker='o', linestyle='-', ms=3, alpha=0.8, label="Standard GMV")

    #plt.ylabel("$C_\ell^{\kappa\kappa}$")
    plt.xlabel('$\ell$')
    plt.title(f'Lensing Bias from Agora Sim / Input Kappa Spectrum (TTEETE only)')
    plt.legend(loc='upper left', fontsize='x-small', bbox_to_anchor=(1,0.5))
    plt.xscale('log')
    plt.xlim(10,lmax)
    plt.ylim(-0.5,0.5)
    #plt.ylim(-2,2)
    plt.savefig(dir_out+f'/figs/bias_TTEETE.png',bbox_inches='tight')

    plt.clf()
    # Uncertainty/input kappa test plots
    plt.plot(bin_centers, binned_uncertainty_sqe_TT[:,0]/binned_input_clkk, color='firebrick', marker='o', linestyle='-', ms=3, alpha=0.8, label="Standard SQE TT")
    plt.plot(bin_centers, binned_uncertainty_gmv[:,0]/binned_input_clkk, color='darkblue', marker='o', linestyle='-', ms=3, alpha=0.8, label="Standard GMV Total")
    plt.xlabel('$\ell$')
    plt.legend(loc='upper left', fontsize='x-small', bbox_to_anchor=(1,0.5))
    plt.xscale('log')
    #plt.yscale('log')
    plt.xlim(10,lmax)
    plt.savefig(dir_out+f'/figs/uncertainty_over_input_clkk.png',bbox_inches='tight')

    plt.figure(1)
    plt.clf()
    plt.axhline(y=0, color='gray', alpha=0.5, linestyle='-')

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

    plt.clf()
    plt.axhline(y=0, color='gray', alpha=0.5, linestyle='-')

    plt.plot(bin_centers, binned_bias_gmv_TTEETE[:,2]/binned_uncertainty_gmv_TTEETE[:,2], color='goldenrod', marker='o', linestyle='--', ms=3, alpha=0.8, label="Cross-ILC GMV (one component CIB)")
    plt.plot(bin_centers, binned_bias_gmv_TTEETE[:,3]/binned_uncertainty_gmv_TTEETE[:,3], color='darkorange', marker='o', linestyle='--', ms=3, alpha=0.8, label="Cross-ILC GMV (two component CIB)")
    plt.plot(bin_centers, binned_bias_gmv_TTEETE[:,1]/binned_uncertainty_gmv_TTEETE[:,1], color='forestgreen', marker='o', linestyle='--', ms=3, alpha=0.8, label="MH GMV")
    plt.plot(bin_centers, binned_bias_gmv_TTEETE[:,4]/binned_uncertainty_gmv_TTEETE[:,4], color='plum', marker='o', linestyle='--', ms=3, alpha=0.8, label="Profile Hardened GMV")
    plt.plot(bin_centers, binned_bias_sqe_TT[:,0]/binned_uncertainty_sqe_TT[:,0], color='firebrick', marker='o', linestyle='--', ms=3, alpha=0.8, label=f'Standard SQE')
    plt.plot(bin_centers, binned_bias_gmv_TTEETE[:,0]/binned_uncertainty_gmv_TTEETE[:,0], color='darkblue', marker='o', linestyle='--', ms=3, alpha=0.8, label="Standard GMV")

    #plt.ylabel("$C_\ell^{\kappa\kappa}$")
    plt.xlabel('$\ell$')
    plt.title(f'Bias / Uncertainty (TTEETE only)')
    plt.legend(loc='upper left', fontsize='x-small', bbox_to_anchor=(1,0.5))
    plt.xscale('log')
    plt.xlim(10,lmax)
    plt.ylim(-0.5,0.5)
    plt.savefig(dir_out+f'/figs/bias_over_uncertainty_TTEETE.png',bbox_inches='tight')

    plt.figure(2)
    plt.clf()
    # Cross with input
    plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')
    plt.plot(l, input_clkk, 'gray',linestyle='--', label='Input $C_\ell^{\kappa\kappa}$')

    plt.plot(bin_centers, cross_gmv[:,2], color='goldenrod', marker='o', linestyle='None', ms=3, alpha=0.8, label="Cross-ILC GMV (one component CIB)")
    plt.plot(bin_centers, cross_gmv[:,3], color='darkorange', marker='o', linestyle='None', ms=3, alpha=0.8, label="Cross-ILC GMV (two component CIB)")
    plt.plot(bin_centers, cross_gmv[:,1], color='forestgreen', marker='o', linestyle='None', ms=3, alpha=0.8, label="MH GMV")
    plt.plot(bin_centers, cross_gmv[:,4], color='plum', marker='o', linestyle='None', ms=3, alpha=0.8, label="Profile Hardened GMV")
    plt.plot(bin_centers, cross_sqe[:,0], color='firebrick', marker='o', linestyle='None', ms=3, alpha=0.8, label=f'Standard SQE')
    plt.plot(bin_centers, cross_gmv[:,0], color='darkblue', marker='o', linestyle='None', ms=3, alpha=0.8, label="Standard GMV")

    plt.ylabel("$C_\ell^{\kappa\kappa}$")
    plt.xlabel('$\ell$')
    plt.title(f'Agora Reconstruction x Input Kappa')
    plt.legend(loc='upper left', fontsize='x-small', bbox_to_anchor=(1,0.5))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,lmax)
    plt.ylim(1e-9,1e-6)
    plt.savefig(dir_out+f'/figs/agora_cross_total.png',bbox_inches='tight')

    plt.clf()
    plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')
    plt.plot(l, input_clkk, 'gray',linestyle='--', label='Input $C_\ell^{\kappa\kappa}$')

    plt.plot(bin_centers, cross_gmv_TTEETE[:,2], color='goldenrod', marker='o', linestyle='None', ms=3, alpha=0.8, label="Cross-ILC GMV (one component CIB)")
    plt.plot(bin_centers, cross_gmv_TTEETE[:,3], color='darkorange', marker='o', linestyle='None', ms=3, alpha=0.8, label="Cross-ILC GMV (two component CIB)")
    plt.plot(bin_centers, cross_gmv_TTEETE[:,1], color='forestgreen', marker='o', linestyle='None', ms=3, alpha=0.8, label="MH GMV")
    plt.plot(bin_centers, cross_gmv_TTEETE[:,4], color='plum', marker='o', linestyle='None', ms=3, alpha=0.8, label="Profile Hardened GMV")
    plt.plot(bin_centers, cross_sqe_TT[:,0], color='firebrick', marker='o', linestyle='None', ms=3, alpha=0.8, label=f'Standard SQE')
    plt.plot(bin_centers, cross_gmv_TTEETE[:,0], color='darkblue', marker='o', linestyle='None', ms=3, alpha=0.8, label="Standard GMV")

    plt.ylabel("$C_\ell^{\kappa\kappa}$")
    plt.xlabel('$\ell$')
    plt.title(f'Agora Reconstruction x Input Kappa (TTEETE only)')
    plt.legend(loc='upper left', fontsize='x-small', bbox_to_anchor=(1,0.5))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,lmax)
    plt.ylim(1e-9,1e-6)
    plt.savefig(dir_out+f'/figs/agora_cross_TTEETE.png',bbox_inches='tight')

    plt.clf()
    plt.axhline(y=1, color='gray', alpha=0.5, linestyle='-')

    plt.plot(bin_centers, cross_gmv[:,2]/binned_input_clkk, color='goldenrod', marker='o', linestyle='--', ms=3, alpha=0.8, label="Cross-ILC GMV (one component CIB)")
    plt.plot(bin_centers, cross_gmv[:,3]/binned_input_clkk, color='darkorange', marker='o', linestyle='--', ms=3, alpha=0.8, label="Cross-ILC GMV (two component CIB)")
    plt.plot(bin_centers, cross_gmv[:,1]/binned_input_clkk, color='forestgreen', marker='o', linestyle='--', ms=3, alpha=0.8, label="MH GMV")
    plt.plot(bin_centers, cross_gmv[:,4]/binned_input_clkk, color='plum', marker='o', linestyle='--', ms=3, alpha=0.8, label="Profile Hardened GMV")
    plt.plot(bin_centers, cross_sqe[:,0]/binned_input_clkk, color='firebrick', marker='o', linestyle='--', ms=3, alpha=0.8, label=f'Standard SQE')
    plt.plot(bin_centers, cross_gmv[:,0]/binned_input_clkk, color='darkblue', marker='o', linestyle='--', ms=3, alpha=0.8, label="Standard GMV")

    plt.xlabel('$\ell$')
    plt.title(f'(Agora Reconstruction x Input Kappa) / Input Kappa')
    plt.legend(loc='upper left', fontsize='x-small', bbox_to_anchor=(1,0.5))
    plt.xscale('log')
    plt.xlim(10,lmax)
    plt.savefig(dir_out+f'/figs/agora_cross_total_ratio.png',bbox_inches='tight')

    plt.clf()
    plt.axhline(y=1, color='gray', alpha=0.5, linestyle='-')

    plt.plot(bin_centers, cross_gmv_TTEETE[:,2]/binned_input_clkk, color='goldenrod', marker='o', linestyle='--', ms=3, alpha=0.8, label="Cross-ILC GMV (one component CIB)")
    plt.plot(bin_centers, cross_gmv_TTEETE[:,3]/binned_input_clkk, color='darkorange', marker='o', linestyle='--', ms=3, alpha=0.8, label="Cross-ILC GMV (two component CIB)")
    plt.plot(bin_centers, cross_gmv_TTEETE[:,1]/binned_input_clkk, color='forestgreen', marker='o', linestyle='--', ms=3, alpha=0.8, label="MH GMV")
    plt.plot(bin_centers, cross_gmv_TTEETE[:,4]/binned_input_clkk, color='plum', marker='o', linestyle='--', ms=3, alpha=0.8, label="Profile Hardened GMV")
    plt.plot(bin_centers, cross_sqe_TT[:,0]/binned_input_clkk, color='firebrick', marker='o', linestyle='--', ms=3, alpha=0.8, label=f'Standard SQE')
    plt.plot(bin_centers, cross_gmv_TTEETE[:,0]/binned_input_clkk, color='darkblue', marker='o', linestyle='--', ms=3, alpha=0.8, label="Standard GMV")

    plt.xlabel('$\ell$')
    plt.title(f'(Agora Reconstruction x Input Kappa) / Input Kappa (TTEETE only)')
    plt.legend(loc='upper left', fontsize='x-small', bbox_to_anchor=(1,0.5))
    plt.xscale('log')
    plt.xlim(10,lmax)
    plt.savefig(dir_out+f'/figs/agora_cross_TTEETE_ratio.png',bbox_inches='tight')
'''

def get_lensing_bias(config, append_list, cinv=False, rdn0=False):
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
    klm = hp.read_alm('/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_spt3g_input_klm_lmax4096.fits')
    input_clkk = hp.alm2cl(klm,klm,lmax=lmax)
    binned_input_clkk = [input_clkk[digitized == i].mean() for i in range(1, len(lbins))]

    # Bias
    bias_gmv = np.zeros((len(l),len(append_list)), dtype=np.complex_)
    bias_sqe = np.zeros((len(l),len(append_list)), dtype=np.complex_)
    bias_gmv_TTEETE = np.zeros((len(l),len(append_list)), dtype=np.complex_)
    bias_sqe_TT = np.zeros((len(l),len(append_list)), dtype=np.complex_)
    binned_bias_gmv = np.zeros((len(bin_centers),len(append_list)), dtype=np.complex_)
    binned_bias_sqe = np.zeros((len(bin_centers),len(append_list)), dtype=np.complex_)
    binned_bias_gmv_TTEETE = np.zeros((len(bin_centers),len(append_list)), dtype=np.complex_)
    binned_bias_sqe_TT = np.zeros((len(bin_centers),len(append_list)), dtype=np.complex_)
    # Uncertainty saved from before
    binned_uncertainty_gmv = np.zeros((len(bin_centers),len(append_list)), dtype=np.complex_)
    binned_uncertainty_sqe = np.zeros((len(bin_centers),len(append_list)), dtype=np.complex_)
    binned_uncertainty_gmv_TTEETE = np.zeros((len(bin_centers),len(append_list)), dtype=np.complex_)
    binned_uncertainty_sqe_TT = np.zeros((len(bin_centers),len(append_list)), dtype=np.complex_)
    # Cross with input
    cross_gmv = np.zeros((len(bin_centers),len(append_list)), dtype=np.complex_)
    cross_sqe = np.zeros((len(bin_centers),len(append_list)), dtype=np.complex_)
    cross_gmv_TTEETE = np.zeros((len(bin_centers),len(append_list)), dtype=np.complex_)
    cross_sqe_TT = np.zeros((len(bin_centers),len(append_list)), dtype=np.complex_)

    for j, append in enumerate(append_list):
        append_alt = append[6:]
        print(f'Doing {append_alt}!')
        if append == 'agora_profhrd':
            u = pickle.load(open(profile_file,'rb'))
        else:
            u = None

        if append == 'agora_standard' or append == 'agora_profhrd':
            # Get SQE response
            ests = ['TT', 'EE', 'TE', 'ET', 'TB', 'BT', 'EB', 'BE']
            resps_original = np.zeros((len(l),len(ests)), dtype=np.complex_)
            inv_resps_original = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
            for i, est in enumerate(ests):
                resps_original[:,i] = get_sim_response(est,config,gmv=False,append=append_alt,sims=np.arange(99)+1,cinv=False)
                inv_resps_original[1:,i] = 1/(resps_original)[1:,i]
            resp_original = np.sum(resps_original, axis=1)
            inv_resp_original = np.zeros_like(l,dtype=np.complex_); inv_resp_original[1:] = 1/(resp_original)[1:]

        elif append == 'agora_mh' or append == 'agora_crossilc_onesed' or append == 'agora_crossilc_twoseds':
            # Get SQE response
            ests = ['T1T2', 'T2T1', 'EE', 'TE', 'ET', 'TB', 'BT', 'EB', 'BE']
            resps_original = np.zeros((len(l),len(ests)), dtype=np.complex_)
            inv_resps_original = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
            for i, est in enumerate(ests):
                resps_original[:,i] = get_sim_response(est,config,gmv=False,append=append_alt,sims=np.arange(99)+1,cinv=False)
                inv_resps_original[1:,i] = 1/(resps_original)[1:,i]
            resp_original = 0.5*resps_original[:,0]+0.5*resps_original[:,1]+np.sum(resps_original[:,2:], axis=1)
            inv_resp_original = np.zeros_like(l,dtype=np.complex_); inv_resp_original[1:] = 1/(resp_original)[1:]

        # GMV response
        if cinv:
            resps_gmv = np.zeros((len(l),len(ests)), dtype=np.complex_)
            inv_resps_gmv = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
            for i, est in enumerate(ests):
                resps_gmv[:,i] = get_sim_response(est,config,gmv=True,cinv=True,append=append_alt,sims=np.arange(99)+1)
                inv_resps_gmv[1:,i] = 1/(resps_gmv)[1:,i]
            if append == 'agora_standard' or append == 'agora_profhrd':
                resp_gmv = np.sum(resps_gmv, axis=1)
                resp_gmv_TTEETE = np.sum(resps_gmv[:,:4], axis=1)
                resp_gmv_TBEB = np.sum(resps_gmv[:,4:], axis=1)
            elif append == 'agora_mh' or append == 'agora_crossilc_onesed' or append == 'agora_crossilc_twoseds':
                resp_gmv = 0.5*np.sum(resps_gmv[:,:2], axis=1)+np.sum(resps_gmv[:,2:], axis=1)
                resp_gmv_TTEETE = 0.5*np.sum(resps_gmv[:,:2], axis=1)+np.sum(resps_gmv[:,2:5], axis=1)
                resp_gmv_TBEB = np.sum(resps_gmv[:,5:], axis=1)
        else:
            resp_gmv = get_sim_response('all',config,gmv=True,append=append_alt,sims=np.arange(99)+1,cinv=False)
            resp_gmv_TTEETE = get_sim_response('TTEETE',config,gmv=True,append=append_alt,sims=np.arange(99)+1,cinv=False)
            resp_gmv_TBEB = get_sim_response('TBEB',config,gmv=True,append=append_alt,sims=np.arange(99)+1,cinv=False)
        inv_resp_gmv = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv[1:] = 1./(resp_gmv)[1:]
        inv_resp_gmv_TTEETE = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TTEETE[1:] = 1./(resp_gmv_TTEETE)[1:]
        inv_resp_gmv_TBEB = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TBEB[1:] = 1./(resp_gmv_TBEB)[1:]

        if append == 'agora_profhrd':
            # Get the profile response and weight
            # SQE
            resp_original_TT_ss = get_analytic_response('TTprf',config,gmv=False,append=append_alt,u=u)
            resp_original_TT_sk = get_analytic_response('TTTTprf',config,gmv=False,append=append_alt,u=u)
            weight_original = -1 * resp_original_TT_sk / resp_original_TT_ss
            resp_original_TT_hrd = resps_original[:,0] + weight_original*resp_original_TT_sk
            #resp_original_hrd = resp_original + weight_original*resp_original_TT_sk # Equivalent to resp_original_TT (hardened) + np.sum(resps_original[:,1:], axis=1)
            resp_original_hrd = resp_original_TT_hrd + np.sum(resps_original[:,1:], axis=1)
            inv_resp_original_hrd = np.zeros_like(l,dtype=np.complex_); inv_resp_original_hrd[1:] = 1/(resp_original_hrd)[1:]
            inv_resp_original_TT_hrd = np.zeros_like(l,dtype=np.complex_); inv_resp_original_TT_hrd[1:] = 1/(resp_original_TT_hrd)[1:]

            # GMV
            resp_gmv_TTEETE_ss = get_analytic_response('TTEETEprf',config,gmv=True,append=append_alt,u=u[lmin:])
            resp_gmv_TTEETE_sk = get_analytic_response('TTEETETTEETEprf',config,gmv=True,append=append_alt,u=u[lmin:])
            weight_gmv = -1 * resp_gmv_TTEETE_sk / resp_gmv_TTEETE_ss
            resp_gmv_TTEETE_hrd = resp_gmv_TTEETE + weight_gmv*resp_gmv_TTEETE_sk
            #resp_gmv_hrd = resp_gmv + weight_gmv*resp_gmv_TTEETE_sk # Equivalent to resp_gmv_TTEETE (hardened) + resp_gmv_TBEB (unhardened)
            resp_gmv_hrd = resp_gmv_TTEETE_hrd + resp_gmv_TBEB
            inv_resp_gmv_hrd = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_hrd[1:] = 1./(resp_gmv_hrd)[1:]
            inv_resp_gmv_TTEETE_hrd = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TTEETE_hrd[1:] = 1./(resp_gmv_TTEETE_hrd)[1:]

        # Get N0 correction (remember these still need (l*(l+1))**2/4 factor to convert to kappa)
        if rdn0:
            if cinv:
                #TODO
                #n0_gmv = get_rdn0(sims=np.arange(98)+1,qetype='gmv_cinv',config=config,append=append_alt)
                n0_gmv = get_rdn0(sims=np.arange(97)+2,qetype='gmv_cinv',config=config,append=append_alt)
            else:
                n0_gmv = get_rdn0(sims=np.arange(98)+1,qetype='gmv',config=config,append=append_alt)
            n0_gmv_total = n0_gmv * (l*(l+1))**2/4
            if append == 'agora_profhrd':
                print('NO RDN0 IMPLEMENTATION FOR PROFHRD')
            #TODO
            #n0_original = get_rdn0(sims=np.arange(98)+1,qetype='sqe',config=config,
            #                       append=append_alt)
            n0_original = get_rdn0(sims=np.arange(97)+2,qetype='sqe',config=config,
                                   append=append_alt)
            n0_original_total = n0_original * (l*(l+1))**2/4
        else:
            if cinv:
                n0_gmv = get_n0(sims=np.arange(98)+1,qetype='gmv_cinv',config=config,append=append_alt)
            else:
                n0_gmv = get_n0(sims=np.arange(98)+1,qetype='gmv',config=config,append=append_alt)
            n0_gmv_total = n0_gmv['total'] * (l*(l+1))**2/4
            n0_gmv_TTEETE = n0_gmv['TTEETE'] * (l*(l+1))**2/4
            n0_gmv_TBEB = n0_gmv['TBEB'] * (l*(l+1))**2/4
            if append == 'agora_profhrd':
                n0_gmv_total_hrd = n0_gmv['total_hrd'] * (l*(l+1))**2/4
                n0_gmv_TTEETE_hrd = n0_gmv['TTEETE_hrd'] * (l*(l+1))**2/4
            n0_original = get_n0(sims=np.arange(98)+1,qetype='sqe',config=config,
                                 append=append_alt)
            n0_original_total = n0_original['total'] * (l*(l+1))**2/4
            n0_original_TT = n0_original['TT'] * (l*(l+1))**2/4
            if append == 'agora_profhrd':
                n0_original_total_hrd = n0_original['total_hrd'] * (l*(l+1))**2/4
                n0_original_TT_hrd = n0_original['TT_hrd'] * (l*(l+1))**2/4

        if cinv:
            n1_gmv = get_n1(sims=np.arange(98)+1,qetype='gmv_cinv',config=config,append=append_alt)
        else:
            n1_gmv = get_n1(sims=np.arange(98)+1,qetype='gmv',config=config,append=append_alt)
        n1_gmv_total = n1_gmv['total'] * (l*(l+1))**2/4
        #n1_gmv_TTEETE = n1_gmv['TTEETE'] * (l*(l+1))**2/4
        n1_original = get_n1(sims=np.arange(98)+1,qetype='sqe',config=config,
                             append=append_alt)
        n1_original_total = n1_original['total'] * (l*(l+1))**2/4
        if append == 'agora_mh' or append == 'agora_crossilc_onesed' or append == 'agora_crossilc_twoseds':
            n1_original_T1T2 = n1_original['T1T2'] * (l*(l+1))**2/4
            n1_original_T2T1 = n1_original['T2T1'] * (l*(l+1))**2/4
        else:
            n1_original_TT = n1_original['TT'] * (l*(l+1))**2/4

        # Load GMV plms
        if cinv:
            plms_gmv = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_gmv_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv.npy')),len(ests)), dtype=np.complex_)
            for i, est in enumerate(ests):
                plms_gmv[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv.npy')
            if append == 'agora_mh' or append == 'agora_crossilc_onesed' or append == 'agora_crossilc_twoseds':
                plm_gmv = 0.5*np.sum(plms_gmv[:,:2], axis=1)+np.sum(plms_gmv[:,2:], axis=1)
                plm_gmv_TTEETE = 0.5*np.sum(plms_gmv[:,:2], axis=1)+np.sum(plms_gmv[:,2:5], axis=1)
                plm_gmv_TBEB = np.sum(plms_gmv[:,5:], axis=1)
            else:
                plm_gmv = np.sum(plms_gmv, axis=1)
                plm_gmv_TTEETE = np.sum(plms_gmv[:,:4], axis=1)
                plm_gmv_TBEB = np.sum(plms_gmv[:,4:], axis=1)
        else:
            plm_gmv = np.load(dir_out+f'/plm_all_healqest_gmv_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            plm_gmv_TTEETE = np.load(dir_out+f'/plm_TTEETE_healqest_gmv_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            plm_gmv_TBEB = np.load(dir_out+f'/plm_TBEB_healqest_gmv_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')

        # Load SQE plms
        plms_original = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')),len(ests)), dtype=np.complex_)
        for i, est in enumerate(ests):
            plms_original[:,i] = np.load(dir_out+f'/plm_{est}_healqest_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
        if append == 'agora_mh' or append == 'agora_crossilc_onesed' or append == 'agora_crossilc_twoseds':
            plm_original = 0.5*plms_original[:,0]+0.5*plms_original[:,1]+np.sum(plms_original[:,2:], axis=1)
        else:
            plm_original = np.sum(plms_original, axis=1)

        if append == 'agora_profhrd':
            # Harden!
            glm_prf_TTEETE = np.load(dir_out+f'/plm_TTEETEprf_healqest_gmv_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            plm_gmv_TTEETE_hrd = plm_gmv_TTEETE + hp.almxfl(glm_prf_TTEETE, weight_gmv)
            #plm_gmv_hrd = plm_gmv + hp.almxfl(glm_prf_TTEETE, weight_gmv) # Equivalent to plm_gmv_TTEETE_hrd + plm_gmv_TBEB
            plm_gmv_hrd = plm_gmv_TTEETE_hrd + plm_gmv_TBEB

            # SQE
            glm_prf_TT = np.load(dir_out+f'/plm_TTprf_healqest_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            plm_original_TT_hrd = plms_original[:,0] + hp.almxfl(glm_prf_TT, weight_original)
            plm_original_hrd = plm_original_TT_hrd + np.sum(plms_original[:,1:], axis=1)

        # Response correct
        plm_gmv_resp_corr = hp.almxfl(plm_gmv,inv_resp_gmv)
        plm_gmv_resp_corr_TTEETE = hp.almxfl(plm_gmv_TTEETE,inv_resp_gmv_TTEETE)
        plm_gmv_resp_corr_TBEB = hp.almxfl(plm_gmv_TBEB,inv_resp_gmv_TBEB)
        if append == 'agora_profhrd':
            plm_gmv_resp_corr_hrd = hp.almxfl(plm_gmv_hrd,inv_resp_gmv_hrd)
            plm_gmv_TTEETE_resp_corr_hrd = hp.almxfl(plm_gmv_TTEETE_hrd,inv_resp_gmv_TTEETE_hrd)
        plm_original_resp_corr = hp.almxfl(plm_original,inv_resp_original)
        if append == 'agora_mh' or append == 'agora_crossilc_onesed' or append == 'agora_crossilc_twoseds':
            plm_original_resp_corr_T1T2 = hp.almxfl(plms_original[:,0],inv_resps_original[:,0])
            plm_original_resp_corr_T2T1 = hp.almxfl(plms_original[:,1],inv_resps_original[:,1])
        else:
            plm_original_resp_corr_TT = hp.almxfl(plms_original[:,0],inv_resps_original[:,0])
            if append == 'agora_profhrd':
                plm_original_resp_corr_hrd = hp.almxfl(plm_original_hrd,inv_resp_original_hrd)
                plm_original_TT_resp_corr_hrd = hp.almxfl(plm_original_TT_hrd,inv_resp_original_TT_hrd)

        # Get spectra
        auto_gmv = hp.alm2cl(plm_gmv_resp_corr, plm_gmv_resp_corr, lmax=lmax) * (l*(l+1))**2/4
        auto_gmv_TTEETE = hp.alm2cl(plm_gmv_resp_corr_TTEETE, plm_gmv_resp_corr_TTEETE, lmax=lmax) * (l*(l+1))**2/4
        auto_gmv_TBEB = hp.alm2cl(plm_gmv_resp_corr_TBEB, plm_gmv_resp_corr_TBEB, lmax=lmax) * (l*(l+1))**2/4
        if append == 'agora_profhrd':
            auto_gmv_hrd = hp.alm2cl(plm_gmv_resp_corr_hrd, plm_gmv_resp_corr_hrd, lmax=lmax) * (l*(l+1))**2/4
            auto_gmv_TTEETE_hrd = hp.alm2cl(plm_gmv_TTEETE_resp_corr_hrd, plm_gmv_TTEETE_resp_corr_hrd, lmax=lmax) * (l*(l+1))**2/4
        auto_original = hp.alm2cl(plm_original_resp_corr, plm_original_resp_corr, lmax=lmax) * (l*(l+1))**2/4
        if append == 'agora_mh' or append == 'agora_crossilc_onesed' or append == 'agora_crossilc_twoseds':
            auto_original_T1T2 = hp.alm2cl(plm_original_resp_corr_T1T2, plm_original_resp_corr_T1T2, lmax=lmax) * (l*(l+1))**2/4
            auto_original_T2T1 = hp.alm2cl(plm_original_resp_corr_T2T1, plm_original_resp_corr_T2T1, lmax=lmax) * (l*(l+1))**2/4
        else:
            auto_original_TT = hp.alm2cl(plm_original_resp_corr_TT, plm_original_resp_corr_TT, lmax=lmax) * (l*(l+1))**2/4
        if append == 'agora_profhrd':
            auto_original_hrd = hp.alm2cl(plm_original_resp_corr_hrd, plm_original_resp_corr_hrd, lmax=lmax) * (l*(l+1))**2/4
            auto_original_TT_hrd = hp.alm2cl(plm_original_TT_resp_corr_hrd, plm_original_TT_resp_corr_hrd, lmax=lmax) * (l*(l+1))**2/4

        # Cross with input
        if append == 'agora_profhrd':
            cross_gmv_unbinned = hp.alm2cl(klm, plm_gmv_resp_corr_hrd) * (l*(l+1))/2
            cross_sqe_unbinned = hp.alm2cl(klm, plm_original_resp_corr_hrd) * (l*(l+1))/2
            cross_gmv_TTEETE_unbinned = hp.alm2cl(klm, plm_gmv_TTEETE_resp_corr_hrd) * (l*(l+1))/2
        else:
            cross_gmv_unbinned = hp.alm2cl(klm, plm_gmv_resp_corr) * (l*(l+1))/2
            cross_sqe_unbinned = hp.alm2cl(klm, plm_original_resp_corr) * (l*(l+1))/2
            cross_gmv_TTEETE_unbinned = hp.alm2cl(klm, plm_gmv_resp_corr_TTEETE) * (l*(l+1))/2
        if append == 'agora_mh' or append == 'agora_crossilc_onesed' or append == 'agora_crossilc_twoseds':
            cross_sqe_TT_unbinned = hp.alm2cl(klm, 0.5*(plm_original_resp_corr_T1T2+plm_original_resp_corr_T2T1)) * (l*(l+1))/2
        elif append == 'agora_profhrd':
            cross_sqe_TT_unbinned = hp.alm2cl(klm, plm_original_TT_resp_corr_hrd) * (l*(l+1))/2
        else:
            cross_sqe_TT_unbinned = hp.alm2cl(klm, plm_original_resp_corr_TT) * (l*(l+1))/2
        cross_gmv[:,j] = [cross_gmv_unbinned[digitized == i].mean() for i in range(1, len(lbins))]
        cross_sqe[:,j] = [cross_sqe_unbinned[digitized == i].mean() for i in range(1, len(lbins))]
        cross_gmv_TTEETE[:,j] = [cross_gmv_TTEETE_unbinned[digitized == i].mean() for i in range(1, len(lbins))]
        cross_sqe_TT[:,j] = [cross_sqe_TT_unbinned[digitized == i].mean() for i in range(1, len(lbins))]

        # N0 and N1 subtract
        auto_gmv_debiased = auto_gmv - n0_gmv_total - n1_gmv_total
        #auto_gmv_debiased_TTEETE = auto_gmv_TTEETE - n0_gmv_TTEETE - n1_gmv_TTEETE
        auto_original_debiased = auto_original - n0_original_total - n1_original_total
        #if append == 'agora_mh' or append == 'agora_crossilc_onesed' or append == 'agora_crossilc_twoseds':
        #    auto_original_debiased_TT = 0.5*(auto_original_T1T2+auto_original_T2T1-n0_original_TT-n0_original_TT-n1_original_T1T2-n1_original_T2T1)
        #else:
        #    auto_original_debiased_TT = auto_original_TT - n0_original_TT - n1_original_TT
        if append == 'agora_profhrd':
            auto_gmv_debiased_hrd = auto_gmv_hrd - n0_gmv_total_hrd - n1_gmv_total
            auto_original_debiased_hrd = auto_original_hrd - n0_original_total_hrd - n1_original_total
            #auto_gmv_debiased_hrd_TTEETE = auto_gmv_TTEETE_hrd - n0_gmv_TTEETE_hrd - n1_gmv_TTEETE
            #auto_original_debiased_hrd_TT = auto_original_TT_hrd - n0_original_TT_hrd - n1_original_TT

        # Bin!
        binned_auto_gmv_debiased = [auto_gmv_debiased[digitized == i].mean() for i in range(1, len(lbins))]
        binned_auto_original_debiased = [auto_original_debiased[digitized == i].mean() for i in range(1, len(lbins))]
        #binned_auto_gmv_debiased_TTEETE = [auto_gmv_debiased_TTEETE[digitized == i].mean() for i in range(1, len(lbins))]
        #binned_auto_original_debiased_TT = [auto_original_debiased_TT[digitized == i].mean() for i in range(1, len(lbins))]
        if append == 'agora_profhrd':
            binned_auto_gmv_debiased_hrd = [auto_gmv_debiased_hrd[digitized == i].mean() for i in range(1, len(lbins))]
            binned_auto_original_debiased_hrd = [auto_original_debiased_hrd[digitized == i].mean() for i in range(1, len(lbins))]
            #binned_auto_gmv_debiased_hrd_TTEETE = [auto_gmv_debiased_hrd_TTEETE[digitized == i].mean() for i in range(1, len(lbins))]
            #binned_auto_original_debiased_hrd_TT = [auto_original_debiased_hrd_TT[digitized == i].mean() for i in range(1, len(lbins))]

        # Get bias
        if append == 'agora_profhrd':
            bias_gmv[:,j] = auto_gmv_debiased_hrd - input_clkk
            bias_sqe[:,j] = auto_original_debiased_hrd - input_clkk
            #bias_gmv_TTEETE[:,j] = auto_gmv_debiased_hrd_TTEETE - input_clkk
            #bias_sqe_TT[:,j] = auto_original_debiased_hrd_TT - input_clkk
        else:
            bias_gmv[:,j] = auto_gmv_debiased - input_clkk
            bias_sqe[:,j] = auto_original_debiased - input_clkk
            #bias_gmv_TTEETE[:,j] = auto_gmv_debiased_TTEETE - input_clkk
            #bias_sqe_TT[:,j] = auto_original_debiased_TT - input_clkk
        #binned_bias_gmv[:,j] = [bias_gmv[:,j][digitized == i].mean() for i in range(1, len(lbins))]
        #binned_bias_sqe[:,j] = [bias_sqe[:,j][digitized == i].mean() for i in range(1, len(lbins))]
        binned_bias_gmv[:,j] = np.array(binned_auto_gmv_debiased) - np.array(binned_input_clkk)
        binned_bias_sqe[:,j] = np.array(binned_auto_original_debiased) - np.array(binned_input_clkk)
        #binned_bias_gmv_TTEETE[:,j] = [bias_gmv_TTEETE[:,j][digitized == i].mean() for i in range(1, len(lbins))]
        #binned_bias_sqe_TT[:,j] = [bias_sqe_TT[:,j][digitized == i].mean() for i in range(1, len(lbins))]

        # Get uncertainty
        #if cinv:
        #    uncertainty_cinv = np.load(dir_out+f'/agora_reconstruction/measurement_uncertainty_lmaxT{lmaxT}_{append_alt}_cinv.npy')[:,0]
        #    uncertainty_sqe = np.load(dir_out+f'/agora_reconstruction/measurement_uncertainty_lmaxT{lmaxT}_{append_alt}.npy')[:,1]
        #    uncertainty_cinv_TTEETE = np.load(dir_out+f'/agora_reconstruction/measurement_uncertainty_lmaxT{lmaxT}_{append_alt}_cinv.npy')[:,1]
        #    uncertainty_sqe_TT = np.load(dir_out+f'/agora_reconstruction/measurement_uncertainty_lmaxT{lmaxT}_{append_alt}.npy')[:,3]
        #    binned_uncertainty_cinv[:,j] = np.load(dir_out+f'/agora_reconstruction/binned_measurement_uncertainty_lmaxT{lmaxT}_{append_alt}_cinv.npy')[:,0]
        #    binned_uncertainty_sqe[:,j] = np.load(dir_out+f'/agora_reconstruction/binned_measurement_uncertainty_lmaxT{lmaxT}_{append_alt}.npy')[:,1]
        #    binned_uncertainty_cinv_TTEETE[:,j] = np.load(dir_out+f'/agora_reconstruction/binned_measurement_uncertainty_lmaxT{lmaxT}_{append_alt}_cinv.npy')[:,1]
        #    binned_uncertainty_sqe_TT[:,j] = np.load(dir_out+f'/agora_reconstruction/binned_measurement_uncertainty_lmaxT{lmaxT}_{append_alt}.npy')[:,3]
        #else:
        #    uncertainty_gmv = np.load(dir_out+f'/agora_reconstruction/measurement_uncertainty_lmaxT{lmaxT}_{append_alt}.npy')[:,0]
        #    uncertainty_sqe = np.load(dir_out+f'/agora_reconstruction/measurement_uncertainty_lmaxT{lmaxT}_{append_alt}.npy')[:,1]
        #    uncertainty_gmv_TTEETE = np.load(dir_out+f'/agora_reconstruction/measurement_uncertainty_lmaxT{lmaxT}_{append_alt}.npy')[:,2]
        #    uncertainty_sqe_TT = np.load(dir_out+f'/agora_reconstruction/measurement_uncertainty_lmaxT{lmaxT}_{append_alt}.npy')[:,3]
        #    binned_uncertainty_gmv[:,j] = np.load(dir_out+f'/agora_reconstruction/binned_measurement_uncertainty_lmaxT{lmaxT}_{append_alt}.npy')[:,0]
        #    binned_uncertainty_sqe[:,j] = np.load(dir_out+f'/agora_reconstruction/binned_measurement_uncertainty_lmaxT{lmaxT}_{append_alt}.npy')[:,1]
        #    binned_uncertainty_gmv_TTEETE[:,j] = np.load(dir_out+f'/agora_reconstruction/binned_measurement_uncertainty_lmaxT{lmaxT}_{append_alt}.npy')[:,2]
        #    binned_uncertainty_sqe_TT[:,j] = np.load(dir_out+f'/agora_reconstruction/binned_measurement_uncertainty_lmaxT{lmaxT}_{append_alt}.npy')[:,3]

        # Smooth the unbinned lines because it's noisy af...
        #B, A = signal.butter(3, 0.1, output='ba')
        #bias_over_uncertainty_gmv[:,j] = signal.filtfilt(B,A,bias_over_uncertainty_gmv[:,j])
        #bias_over_uncertainty_sqe[:,j] = signal.filtfilt(B,A,bias_over_uncertainty_sqe[:,j])
        #bias_over_uncertainty_gmv_TTEETE[:,j] = signal.filtfilt(B,A,bias_over_uncertainty_gmv_TTEETE[:,j])
        #bias_over_uncertainty_sqe_TT[:,j] = signal.filtfilt(B,A,bias_over_uncertainty_sqe_TT[:,j])
        #bias_over_uncertainty_gmv[:,j] = savitzky_golay(bias_over_uncertainty_gmv[:,j],101,3)
        #bias_over_uncertainty_sqe[:,j] = savitzky_golay(bias_over_uncertainty_sqe[:,j],101,3)
        #bias_over_uncertainty_gmv_TTEETE[:,j] = savitzky_golay(bias_over_uncertainty_gmv_TTEETE[:,j],101,3)
        #bias_over_uncertainty_sqe_TT[:,j] = savitzky_golay(bias_over_uncertainty_sqe_TT[:,j],101,3)
        #bias_over_uncertainty_gmv[:,j] = lfilter([1.0 / 150] * 150, 1, bias_over_uncertainty_gmv[:,j])
        #bias_over_uncertainty_sqe[:,j] = lfilter([1.0 / 150] * 150, 1, bias_over_uncertainty_sqe[:,j])
        #bias_over_uncertainty_gmv_TTEETE[:,j] = lfilter([1.0 / 150] * 150, 1, bias_over_uncertainty_gmv_TTEETE[:,j])
        #bias_over_uncertainty_sqe_TT[:,j] = lfilter([1.0 / 150] * 150, 1, bias_over_uncertainty_sqe_TT[:,j])

    return binned_bias_gmv, binned_bias_sqe

def get_rdn0(sims,qetype,config,append):
    '''
    Only returns total N0, not for each estimator.
    Doing withT3, noT3 not implemented.
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
        filename = dir_out+f'/n0/rdn0_{num}simpairs_healqest_{qetype}_withT3_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_resp_from_sims_9ests.pkl'
    else:
        #TODO
        #filename = dir_out+f'/n0/rdn0_{num}simpairs_healqest_{qetype}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_resp_from_sims.pkl'
        filename = dir_out+f'/n0/rdn0_{num}simpairs_healqest_{qetype}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_resp_from_sims_SIM1_AS_RD.pkl'

    if os.path.isfile(filename):
        print(f'Getting RDN0: {filename}')
        rdn0 = pickle.load(open(filename,'rb'))

    elif qetype == 'gmv' or qetype == 'gmv_cinv' or qetype == 'sqe':
        if append == 'standard' or append == 'profhrd':
            ests = ['TT', 'EE', 'TE', 'ET', 'TB', 'BT', 'EB', 'BE']
        elif append == 'mh' or append == 'crossilc_onesed' or append == 'crossilc_twoseds':
            ests = ['T1T2', 'T2T1', 'EE', 'TE', 'ET', 'TB', 'BT', 'EB', 'BE']

        # Get response
        #TODO: hard coded the sims...
        if qetype == 'gmv':
            resp = get_sim_response('all',config,gmv=True,append=append,sims=np.arange(99)+1,cinv=False)
        elif qetype == 'gmv_cinv' or qetype == 'sqe':
            resps = np.zeros((len(l),len(ests)), dtype=np.complex_)
            inv_resps = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
            for i, est in enumerate(ests):
                if qetype == 'gmv_cinv':
                    resps[:,i] = get_sim_response(est,config,gmv=True,cinv=True,append=append,sims=np.arange(99)+1)
                elif qetype == 'sqe':
                    resps[:,i] = get_sim_response(est,config,gmv=False,cinv=False,append=append,sims=np.arange(99)+1)
                inv_resps[1:,i] = 1/(resps)[1:,i]
            if append == 'standard':
                resp = np.sum(resps, axis=1)
            elif append == 'mh' or append == 'crossilc_onesed' or append == 'crossilc_twoseds':
                resp = 0.5*np.sum(resps[:,:2], axis=1)+np.sum(resps[:,2:], axis=1)
        inv_resp = np.zeros(len(l),dtype=np.complex_); inv_resp[1:] = 1./(resp)[1:]

        # Get sim-based N0
        #TODO: hard coded the sims...
        if qetype == 'gmv':
            n0 = get_n0(sims=np.arange(98)+1,qetype='gmv',config=config,append=append)
        elif qetype == 'gmv_cinv':
            n0 = get_n0(sims=np.arange(98)+1,qetype='gmv_cinv',config=config,append=append)
        elif qetype == 'sqe':
            n0 = get_n0(sims=np.arange(98)+1,qetype='sqe',config=config,append=append)
        n0_total = n0['total']

        rdn0 = 0
        for i, sim in enumerate(sims):
            if qetype == 'gmv':
                plm_ir = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
                plm_ri = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            elif qetype == 'gmv_cinv':
                # Get ir sims
                plms_ir = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_gmv_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv.npy')),len(ests)), dtype=np.complex_)
                for i, est in enumerate(ests):
                    #TODO
                    #plms_ir[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv.npy')
                    plms_ir[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim}_seed2_1_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv.npy')
                if append == 'standard':
                    plm_ir = np.sum(plms_ir, axis=1)
                elif append == 'mh' or append == 'crossilc_onesed' or append == 'crossilc_twoseds':
                    plm_ir = 0.5*np.sum(plms_ir[:,:2], axis=1)+np.sum(plms_ir[:,2:], axis=1)
                # Get ri sims
                plms_ri = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_gmv_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv.npy')),len(ests)), dtype=np.complex_)
                for i, est in enumerate(ests):
                    #TODO
                    #plms_ri[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv.npy')
                    plms_ri[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_1_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv.npy')
                if append == 'standard':
                    plm_ri = np.sum(plms_ri, axis=1)
                elif append == 'mh' or append == 'crossilc_onesed' or append == 'crossilc_twoseds':
                    plm_ri = 0.5*np.sum(plms_ri[:,:2], axis=1)+np.sum(plms_ri[:,2:], axis=1)
            elif qetype == 'sqe':
                # Get ir sims
                plms_ir = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')),len(ests)), dtype=np.complex_)
                for i, est in enumerate(ests):
                    #TODO
                    #plms_ir[:,i] = np.load(dir_out+f'/plm_{est}_healqest_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
                    plms_ir[:,i] = np.load(dir_out+f'/plm_{est}_healqest_seed1_{sim}_seed2_1_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
                if append == 'standard':
                    plm_ir = np.sum(plms_ir, axis=1)
                elif append == 'mh' or append == 'crossilc_onesed' or append == 'crossilc_twoseds':
                    plm_ir = 0.5*np.sum(plms_ir[:,:2], axis=1)+np.sum(plms_ir[:,2:], axis=1)
                # Get ri sims
                plms_ri = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')),len(ests)), dtype=np.complex_)
                for i, est in enumerate(ests):
                    #TODO
                    #plms_ri[:,i] = np.load(dir_out+f'/plm_{est}_healqest_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
                    plms_ri[:,i] = np.load(dir_out+f'/plm_{est}_healqest_seed1_1_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
                if append == 'standard':
                    plm_ri = np.sum(plms_ri, axis=1)
                elif append == 'mh' or append == 'crossilc_onesed' or append == 'crossilc_twoseds':
                    plm_ri = 0.5*np.sum(plms_ri[:,:2], axis=1)+np.sum(plms_ri[:,2:], axis=1)

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
    if qetype == 'gmv' and (append=='crossilc_twoseds' or append=='crossilc_onesed' or append=='mh'):
        filename = dir_out+f'/n0/n0_{num}simpairs_healqest_{qetype}_withT3_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_resp_from_sims_9ests_fixedweights.pkl'
    elif qetype == 'gmv':
        filename = dir_out+f'/n0/n0_{num}simpairs_healqest_{qetype}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_resp_from_sims.pkl'
    elif qetype == 'gmv_cinv' and (append=='crossilc_twoseds' or append=='crossilc_onesed' or append=='mh'):
        filename = dir_out+f'/n0/n0_{num}simpairs_healqest_{qetype}_withT3_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_resp_from_sims_9ests.pkl'
    elif qetype == 'gmv_cinv':
        filename = dir_out+f'/n0/n0_{num}simpairs_healqest_{qetype}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_resp_from_sims.pkl'
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
    if qetype == 'gmv' and (append=='crossilc_twoseds' or append=='crossilc_onesed' or append=='mh'):
        filename = dir_out+f'/n1/n1_{num}simpairs_healqest_{qetype}_withT3_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_resp_from_sims_9ests_fixedweights.pkl'
    elif qetype == 'gmv':
        filename = dir_out+f'/n1/n1_{num}simpairs_healqest_{qetype}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_resp_from_sims.pkl'
    elif qetype == 'gmv_cinv' and (append=='crossilc_twoseds' or append=='crossilc_onesed' or append=='mh'):
        filename = dir_out+f'/n1/n1_{num}simpairs_healqest_{qetype}_withT3_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_resp_from_sims_n0_with_9ests.pkl'
    elif qetype == 'gmv_cinv':
        filename = dir_out+f'/n1/n1_{num}simpairs_healqest_{qetype}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_resp_from_sims.pkl'
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
        if gmv and cinv and (append=='crossilc_twoseds' or append=='crossilc_onesed' or append=='mh'):
            fn += '_withT3'
        elif gmv and (append=='crossilc_twoseds' or append=='crossilc_onesed' or append=='mh'):
            fn += '_withT3_fixedweights'
        filename = dir_out+f'/resp/sim_resp{fn}.npy'

    if os.path.isfile(filename):
        print('Loading from existing file!')
        sim_resp = np.load(filename)
    else:
        print(f"File {filename} doesn't exist!")
    return sim_resp

def get_analytic_response(est,config,gmv,append,u,filename=None):
    '''
    Should only be used for the profile...
    '''
    print(f'Computing analytic response for est {est}')
    lmax = config['lensrec']['Lmax']
    lmaxT = config['lensrec']['lmaxT']
    lmaxP = config['lensrec']['lmaxP']
    lmin = config['lensrec']['lminT']
    nside = config['lensrec']['nside']
    cltype = config['lensrec']['cltype']
    cls = config['cls']
    sl = {ee:config['cls'][cltype][ee] for ee in config['cls'][cltype].keys()}
    ell = np.arange(lmax+1,dtype=np.float_)
    dir_out = config['dir_out']

    if filename is None:
        fn = ''
        if gmv and (est=='all' or est=='TTEETE' or est=='TBEB'):
            fn += '_gmv_estall'
        elif gmv:
            fn += f'_gmv_est{est}'
        else:
            fn += f'_sqe_est{est}'
        fn += f'_lmaxT{lmaxT}_lmaxP{lmaxP}_lmin{lmin}_cltype{cltype}_{append}'
        filename = dir_out+f'/resp/an_resp{fn}.npy'

    if os.path.isfile(filename):
        print('Loading from existing file!')
        R = np.load(filename)
    else:
        print("File doesn't exist!")

    if gmv:
        # If GMV, save file has columns L, TTEETE, TBEB, all
        if est == 'TTEETE' or est == 'TTEETEprf' or est == 'TTEETETTEETEprf':
            R = R[:,1]
        elif est == 'TBEB':
            R = R[:,2]
        elif est == 'all':
            R = R[:,3]

    return R

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """
    Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError(msg):
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

####################

analyze()
