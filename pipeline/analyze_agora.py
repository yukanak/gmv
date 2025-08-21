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
    Baselining 16 estimators and NO T3 for MH, cross-ILC.
    '''
    fg_model = 'agora'
    #fg_model = 'websky'
    lmax = 4096
    l = np.arange(0,lmax+1)
    lbins = np.logspace(np.log10(50),np.log10(3000),20)
    bin_centers = (lbins[:-1] + lbins[1:]) / 2
    digitized = np.digitize(l, lbins)
    # Theory spectrum
    clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
    ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
    clkk = slpp * (l*(l+1))**2/4
    binned_clkk = np.array([clkk[digitized == i].mean() for i in range(1, len(lbins))])
    # Input kappa
    if fg_model == 'agora':
        klm = hp.almxfl(hp.read_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_spt3g_input_plm_lmax{lmax}.fits'),(l*(l+1))/2)
        klm = utils.reduce_lmax(klm,lmax=lmax)
    else:
        kap = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/websky/kap.fits')
        klm = hp.map2alm(kap)
        klm = utils.reduce_lmax(klm,lmax=lmax)
    input_clkk = hp.alm2cl(klm,klm,lmax=lmax)
    binned_input_clkk = np.array([input_clkk[digitized == i].mean() for i in range(1, len(lbins))])

    # lmaxT = 3000, cinv
    config_file='test_yuka.yaml'
    config = utils.parse_yaml(config_file)
    dir_out = config['dir_out']
    append_list = ['standard']
    ret = get_lensing_bias(config,append_list,cinv=True,sqe=False,sims=np.arange(250)+1,n0_n1_sims=np.arange(249)+1,fg_model=fg_model)
    binned_bias_gmv_3000_cinv = ret['binned_bias'];  binned_uncertainty_gmv_3000_cinv = ret['binned_uncertainty']; binned_debiased_clkk_gmv_3000_cinv = ret['binned_debiased_clkk']
    # Additive MC correction -- subtract out bias of the Gaussian sims with respect to the input
    mc_corrs = []
    for append in append_list:
        data = pickle.load(open(dir_out+f'/gaussian_reconstruction/gaussian_reconstruction_results_lmaxT3000_{fg_model}_{append}_cinv.npy', 'rb'))
        mc_corrs.append(data['ratio_cinv'] - 1)
    mc_corr_gmv_3000_cinv = np.column_stack(mc_corrs)
    if fg_model == 'agora':
        ret = get_lensing_bias(config,append_list,cinv=False,sqe=True,sims=np.arange(250)+1,n0_n1_sims=np.arange(249)+1,fg_model=fg_model)
        binned_bias_sqe_3000 = ret['binned_bias']; binned_uncertainty_sqe_3000 = ret['binned_uncertainty']; binned_debiased_clkk_sqe_3000_cinv = ret['binned_debiased_clkk']
        # Additive MC correction -- subtract out bias of the Gaussian sims with respect to the input
        mc_corrs = []
        for append in append_list:
            data = pickle.load(open(dir_out+f'/gaussian_reconstruction/gaussian_reconstruction_results_lmaxT3000_{fg_model}_{append}_sqe.npy', 'rb'))
            mc_corrs.append(data['ratio_sqe'] - 1)
        mc_corr_sqe_3000 = np.column_stack(mc_corrs)

    # lmaxT = 3500, cinv
    config_file='test_yuka_lmaxT3500.yaml'
    config = utils.parse_yaml(config_file)
    append_list = ['standard', 'mh', 'crossilc_onesed']
    #append_list = ['standard']
    # GMV
    ret = get_lensing_bias(config,append_list,cinv=True,sqe=False,sims=np.arange(250)+1,n0_n1_sims=np.arange(249)+1,fg_model=fg_model)
    binned_bias_gmv_3500_cinv = ret['binned_bias']; binned_uncertainty_gmv_3500_cinv = ret['binned_uncertainty']; binned_debiased_clkk_gmv_3500_cinv = ret['binned_debiased_clkk']
    cross_gmv_3500_cinv = ret['cross']
    # Additive MC correction -- subtract out bias of the Gaussian sims with respect to the input
    mc_corrs = []
    for append in append_list:
        data = pickle.load(open(dir_out+f'/gaussian_reconstruction/gaussian_reconstruction_results_lmaxT3500_{fg_model}_{append}_cinv.npy', 'rb'))
        mc_corrs.append(data['ratio_cinv'] - 1)
    mc_corr_gmv_3500_cinv = np.column_stack(mc_corrs)
    if fg_model == 'agora':
        # SQE
        ret = get_lensing_bias(config,append_list,cinv=False,sqe=True,sims=np.arange(250)+1,n0_n1_sims=np.arange(249)+1,fg_model=fg_model)
        binned_bias_sqe_3500 = ret['binned_bias']; binned_uncertainty_sqe_3500 = ret['binned_uncertainty']; binned_debiased_clkk_sqe_3500_cinv = ret['binned_debiased_clkk']
        # Additive MC correction -- subtract out bias of the Gaussian sims with respect to the input
        mc_corrs = []
        for append in append_list:
            data = pickle.load(open(dir_out+f'/gaussian_reconstruction/gaussian_reconstruction_results_lmaxT3500_{fg_model}_{append}_sqe.npy', 'rb'))
            mc_corrs.append(data['ratio_sqe'] - 1)
        mc_corr_sqe_3500 = np.column_stack(mc_corrs)

    # lmaxT = 4000, cinv
    config_file='test_yuka_lmaxT4000.yaml'
    config = utils.parse_yaml(config_file)
    append_list = ['standard', 'mh', 'crossilc_onesed']
    #append_list = ['standard']
    # GMV
    ret = get_lensing_bias(config,append_list,cinv=True,sqe=False,sims=np.arange(250)+1,n0_n1_sims=np.arange(249)+1,fg_model=fg_model)
    binned_bias_gmv_4000_cinv = ret['binned_bias']; binned_uncertainty_gmv_4000_cinv = ret['binned_uncertainty']; binned_debiased_clkk_gmv_4000_cinv = ret['binned_debiased_clkk']
    cross_gmv_4000_cinv = ret['cross']
    # Additive MC correction -- subtract out bias of the Gaussian sims with respect to the input
    mc_corrs = []
    for append in append_list:
        data = pickle.load(open(dir_out+f'/gaussian_reconstruction/gaussian_reconstruction_results_lmaxT4000_{fg_model}_{append}_cinv.npy', 'rb'))
        mc_corrs.append(data['ratio_cinv'] - 1)
    mc_corr_gmv_4000_cinv = np.column_stack(mc_corrs)
    if fg_model == 'agora':
        # SQE
        ret = get_lensing_bias(config,append_list,cinv=False,sqe=True,sims=np.arange(250)+1,n0_n1_sims=np.arange(249)+1,fg_model=fg_model)
        binned_bias_sqe_4000 = ret['binned_bias']; binned_uncertainty_sqe_4000 = ret['binned_uncertainty']; binned_debiased_clkk_sqe_4000_cinv = ret['binned_debiased_clkk']
        # Additive MC correction -- subtract out bias of the Gaussian sims with respect to the input
        mc_corrs = []
        for append in append_list:
            data = pickle.load(open(dir_out+f'/gaussian_reconstruction/gaussian_reconstruction_results_lmaxT4000_{fg_model}_{append}_sqe.npy', 'rb'))
            mc_corrs.append(data['ratio_sqe'] - 1)
        mc_corr_sqe_4000 = np.column_stack(mc_corrs)
    # Agora reconstruction with Gaussian foregrounds
    #agora_nofg_gmv_4000_cinv_standard = pickle.load(open(dir_out+f'/agora_reconstruction/agora_reconstruction_results_lmaxT4000_{fg_model}_cinv_nofg.npy','rb'))
    #agora_nofg_clkk_gmv_4000_cinv_standard = agora_nofg_gmv_4000_cinv_standard['binned_debiased_clkk']

    dir_out = config['dir_out']

    # Plot
    plt.figure(0)
    plt.clf()
    plt.axhline(y=0, color='gray', alpha=0.5, linestyle='--')
    if fg_model == 'agora':
        plt.errorbar(bin_centers, binned_bias_sqe_3500[:,0]/binned_input_clkk - mc_corr_sqe_3500[:,0], yerr=binned_uncertainty_sqe_3500[:,0]/binned_input_clkk, color='salmon', marker='o', markerfacecolor='none', markeredgecolor='salmon', linestyle='None', ms=7, alpha=0.8, label=f'Standard SQE')
    plt.errorbar(bin_centers, binned_bias_gmv_3500_cinv[:,0]/binned_input_clkk - mc_corr_gmv_3500_cinv[:,0], yerr=binned_uncertainty_gmv_3500_cinv[:,0]/binned_input_clkk, color='cornflowerblue', marker='o', markerfacecolor='none', markeredgecolor='cornflowerblue', linestyle='None', ms=7, alpha=0.8, label="Standard GMV")
    plt.errorbar(bin_centers, binned_bias_gmv_3500_cinv[:,1]/binned_input_clkk - mc_corr_gmv_3500_cinv[:,1], yerr=binned_uncertainty_gmv_3500_cinv[:,1]/binned_input_clkk, color='darkseagreen', marker='o', markerfacecolor='none', markeredgecolor='forestgreen', linestyle='None', ms=7, alpha=0.8, label="MH GMV")
    plt.errorbar(bin_centers, binned_bias_gmv_3500_cinv[:,2]/binned_input_clkk - mc_corr_gmv_3500_cinv[:,2], yerr=binned_uncertainty_gmv_3500_cinv[:,2]/binned_input_clkk, color='burlywood', marker='o', markerfacecolor='none', markeredgecolor='darkorange', linestyle='None', ms=7, alpha=0.8, label="Cross-ILC GMV")
    plt.xlabel('$L$')
    if fg_model == 'agora':
        plt.title(f'Lensing Bias from Agora Sims / Input Kappa Spectrum, lmaxT = 3500',pad=10)
    else:
        plt.title(f'Lensing Bias from WebSky Sims / Input Kappa Spectrum, lmaxT = 3500',pad=10)
    plt.legend(loc='upper left', fontsize='small')
    plt.xscale('log')
    plt.xlim(50,3001)
    if fg_model == 'agora':
        plt.ylim(-0.2,0.2)
    elif fg_model == 'websky':
        plt.ylim(-0.5,0.5)
    #plt.savefig(dir_out+f'/figs/bias_total.png',bbox_inches='tight')
    plt.savefig(dir_out+f'/figs/bias_total_cinv_{fg_model}.png',bbox_inches='tight')

    plt.clf()
    plt.axhline(y=0, color='gray', alpha=0.5, linestyle='--')
    if fg_model == 'agora':
        #plt.plot(bin_centers, binned_bias_sqe_3000[:,0]/binned_input_clkk - mc_corr_sqe_3000[:,0], color='firebrick', marker='o', linestyle=':', ms=3, alpha=0.8, label="Standard SQE, lmaxT = 3000")
        #plt.plot(bin_centers, binned_bias_sqe_3500[:,0]/binned_input_clkk - mc_corr_sqe_3500[:,0], color='salmon', marker='o', linestyle='--', ms=3, alpha=0.8, label="Standard SQE, lmaxT = 3500")
        #plt.plot(bin_centers, binned_bias_sqe_4000[:,0]/binned_input_clkk - mc_corr_sqe_4000[:,0], color='pink', marker='o', linestyle='-', ms=3, alpha=0.8, label="Standard SQE, lmaxT = 4000")
        plt.errorbar(bin_centers, binned_bias_sqe_3000[:,0]/binned_input_clkk - mc_corr_sqe_3000[:,0], yerr=binned_uncertainty_sqe_3000[:,0]/binned_input_clkk, color='firebrick', marker='x', markerfacecolor='none', markeredgecolor='firebrick', linestyle=':', ms=7, alpha=0.8, label=f'Standard SQE, lmaxT = 3000')
        plt.errorbar(bin_centers, binned_bias_sqe_3500[:,0]/binned_input_clkk - mc_corr_sqe_3500[:,0], yerr=binned_uncertainty_sqe_3500[:,0]/binned_input_clkk, color='salmon', marker='o', markerfacecolor='none', markeredgecolor='salmon', linestyle='--', ms=7, alpha=0.8, label=f'Standard SQE, lmaxT = 3500')
        plt.errorbar(bin_centers, binned_bias_sqe_4000[:,0]/binned_input_clkk - mc_corr_sqe_4000[:,0], yerr=binned_uncertainty_sqe_4000[:,0]/binned_input_clkk, color='pink', marker='s', markerfacecolor='none', markeredgecolor='pink', linestyle='-', ms=7, alpha=0.8, label=f'Standard SQE, lmaxT = 4000')
    plt.errorbar(bin_centers, binned_bias_gmv_3000_cinv[:,0]/binned_input_clkk - mc_corr_gmv_3000_cinv[:,0], yerr=binned_uncertainty_gmv_3000_cinv[:,0]/binned_input_clkk, color='darkblue', marker='x', markerfacecolor='none', markeredgecolor='darkblue', linestyle=':', ms=7, alpha=0.8, label="Standard GMV, lmaxT = 3000")
    plt.errorbar(bin_centers, binned_bias_gmv_3500_cinv[:,0]/binned_input_clkk - mc_corr_gmv_3500_cinv[:,0], yerr=binned_uncertainty_gmv_3500_cinv[:,0]/binned_input_clkk, color='cornflowerblue', marker='o', markerfacecolor='none', markeredgecolor='cornflowerblue', linestyle='--', ms=7, alpha=0.8, label="Standard GMV, lmaxT = 3500")
    plt.errorbar(bin_centers, binned_bias_gmv_4000_cinv[:,0]/binned_input_clkk - mc_corr_gmv_4000_cinv[:,0], yerr=binned_uncertainty_gmv_4000_cinv[:,0]/binned_input_clkk, color='lightsteelblue', marker='s', markerfacecolor='none', markeredgecolor='lightsteelblue', linestyle='-', ms=7, alpha=0.8, label="Standard GMV, lmaxT = 4000")
    plt.errorbar(bin_centers, binned_bias_gmv_3000_cinv[:,1]/binned_input_clkk - mc_corr_gmv_3000_cinv[:,1], yerr=binned_uncertainty_gmv_3000_cinv[:,1]/binned_input_clkk, color='darkgreen', marker='x', markerfacecolor='none', markeredgecolor='darkgreen', linestyle=':', ms=7, alpha=0.8, label="Gradient Cleaning GMV, lmaxT = 3000")
    plt.errorbar(bin_centers, binned_bias_gmv_3500_cinv[:,1]/binned_input_clkk - mc_corr_gmv_3500_cinv[:,1], yerr=binned_uncertainty_gmv_3500_cinv[:,1]/binned_input_clkk, color='darkseagreen', marker='o', markerfacecolor='none', markeredgecolor='darkseagreen', linestyle='--', ms=7, alpha=0.8, label="Gradient Cleaning GMV, lmaxT = 3500")
    plt.errorbar(bin_centers, binned_bias_gmv_4000_cinv[:,1]/binned_input_clkk - mc_corr_gmv_4000_cinv[:,1], yerr=binned_uncertainty_gmv_4000_cinv[:,1]/binned_input_clkk, color='palegreen', marker='s', markerfacecolor='none', markeredgecolor='palegreen', linestyle='-', ms=7, alpha=0.8, label="Gradient Cleaning GMV, lmaxT = 4000")
    plt.errorbar(bin_centers, binned_bias_gmv_3000_cinv[:,2]/binned_input_clkk - mc_corr_gmv_3000_cinv[:,2] yerr=binned_uncertainty_gmv_3000_cinv[:,2]/binned_input_clkk, color='darkorange', marker='x', markerfacecolor='none', markeredgecolor='darkorange', linestyle=':', ms=7, alpha=0.8, label="Cross-ILC GMV, lmaxT = 3000")
    plt.errorbar(bin_centers, binned_bias_gmv_3500_cinv[:,2]/binned_input_clkk - mc_corr_gmv_3500_cinv[:,2, yerr=binned_uncertainty_gmv_3500_cinv[:,2]/binned_input_clkk, color='burlywood', marker='o', markerfacecolor='none', markeredgecolor='burlywood', linestyle='--', ms=7, alpha=0.8, label="Cross-ILC GMV, lmaxT = 3500")
    plt.errorbar(bin_centers, binned_bias_gmv_4000_cinv[:,2]/binned_input_clkk - mc_corr_gmv_4000_cinv[:,2, yerr=binned_uncertainty_gmv_4000_cinv[:,2]/binned_input_clkk, color='bisque', marker='s', markerfacecolor='none', markeredgecolor='bisque', linestyle='-', ms=7, alpha=0.8, label="Cross-ILC GMV, lmaxT = 4000")
    #plt.plot(bin_centers, binned_bias_gmv_3000_cinv[:,0]/binned_input_clkk - mc_corr_gmv_3000_cinv[:,0], color='darkblue', marker='o', linestyle=':', ms=3, alpha=0.8, label="Standard GMV, lmaxT = 3000")
    #plt.plot(bin_centers, binned_bias_gmv_3500_cinv[:,0]/binned_input_clkk - mc_corr_gmv_3500_cinv[:,0], color='cornflowerblue', marker='o', linestyle='--', ms=3, alpha=0.8, label="Standard GMV, lmaxT = 3500")
    #plt.plot(bin_centers, binned_bias_gmv_4000_cinv[:,0]/binned_input_clkk - mc_corr_gmv_4000_cinv[:,0], color='lightsteelblue', marker='o', linestyle='-', ms=3, alpha=0.8, label="Standard GMV, lmaxT = 4000")
    #plt.plot(bin_centers, binned_bias_gmv_3000_cinv[:,1]/binned_input_clkk - mc_corr_gmv_3000_cinv[:,1], color='darkgreen', marker='o', linestyle=':', ms=3, alpha=0.8, label="Gradient Cleaning GMV, lmaxT = 3000")
    #plt.plot(bin_centers, binned_bias_gmv_3500_cinv[:,1]/binned_input_clkk - mc_corr_gmv_3500_cinv[:,1], color='darkseagreen', marker='o', linestyle='--', ms=3, alpha=0.8, label="Gradient Cleaning GMV, lmaxT = 3500")
    #plt.plot(bin_centers, binned_bias_gmv_4000_cinv[:,1]/binned_input_clkk - mc_corr_gmv_4000_cinv[:,1], color='palegreen', marker='o', linestyle='-', ms=3, alpha=0.8, label="Gradient Cleaning GMV, lmaxT = 4000")
    #plt.plot(bin_centers, binned_bias_gmv_3000_cinv[:,2]/binned_input_clkk - mc_corr_gmv_3000_cinv[:,2], color='darkorange', marker='o', linestyle=':', ms=3, alpha=0.8, label="Cross-ILC GMV, lmaxT = 3000")
    #plt.plot(bin_centers, binned_bias_gmv_3500_cinv[:,2]/binned_input_clkk - mc_corr_gmv_3500_cinv[:,2], color='burlywood', marker='o', linestyle='--', ms=3, alpha=0.8, label="Cross-ILC GMV, lmaxT = 3500")
    #plt.plot(bin_centers, binned_bias_gmv_4000_cinv[:,2]/binned_input_clkk - mc_corr_gmv_4000_cinv[:,2], color='bisque', marker='o', linestyle='-', ms=3, alpha=0.8, label="Cross-ILC GMV, lmaxT = 4000")
    #plt.errorbar(bin_centers, np.array(binned_debiased_clkk_gmv_4000_cinv[:,0])/binned_input_clkk - mc_corr_gmv_4000_cinv[:,0], yerr=binned_uncertainty_gmv_4000_cinv[:,0]/binned_input_clkk, color='lightsteelblue', marker='s', markerfacecolor='none', markeredgecolor='lightsteelblue', linestyle='None', ms=7, alpha=0.5, label="Standard GMV, lmaxT = 4000, wrt Agora Input Kappa")
    #plt.errorbar(bin_centers, np.array(binned_debiased_clkk_gmv_4000_cinv[:,0])/binned_clkk - mc_corr_gmv_4000_cinv[:,0], yerr=binned_uncertainty_gmv_4000_cinv[:,0]/binned_clkk, color='cornflowerblue', marker='o', markerfacecolor='none', markeredgecolor='cornflowerblue', linestyle='None', ms=7, alpha=0.5, label="Standard GMV, lmaxT = 4000, wrt CAMB clkk")
    #plt.errorbar(bin_centers, np.array(binned_debiased_clkk_gmv_4000_cinv[:,0])/np.array(gaussian_results_gmv_4000_cinv_standard['binned_auto_cinv_debiased_avg']) - mc_corr_gmv_4000_cinv[:,0], yerr=binned_uncertainty_gmv_4000_cinv[:,0]/np.array(gaussian_results_gmv_4000_cinv_standard['binned_auto_cinv_debiased_avg']), color='darkblue', marker='x', markerfacecolor='none', markeredgecolor='darkblue', linestyle='None', ms=7, alpha=0.5, label="Standard GMV, lmaxT = 4000, wrt Averaged Gaussian Reconstruction")
    #plt.errorbar(bin_centers, np.array(binned_debiased_clkk_gmv_4000_cinv[:,0])/np.array(agora_nofg_clkk_gmv_4000_cinv_standard[:,0]) - mc_corr_gmv_4000_cinv[:,0], yerr=binned_uncertainty_gmv_4000_cinv[:,0]/np.array(agora_nofg_clkk_gmv_4000_cinv_standard[:,0]), color='pink', marker='x', markerfacecolor='none', markeredgecolor='pink', linestyle='None', ms=7, alpha=0.5, label="Standard GMV, lmaxT = 4000, wrt No NG Foregrounds Agora Reconstruction")
    plt.xlabel('$L$')
    if fg_model == 'agora':
        plt.title(f'Lensing Bias from Agora Sims / Input Kappa Spectrum',pad=10)
    else:
        plt.title(f'Lensing Bias from WebSky Sims / Input Kappa Spectrum',pad=10)
    plt.legend(loc='upper left', fontsize='small')
    plt.xscale('log')
    plt.xlim(50,3001)
    if fg_model == 'agora':
        plt.ylim(-0.2,0.2)
    elif fg_model == 'websky':
        plt.ylim(-0.5,0.5)
    plt.savefig(dir_out+f'/figs/bias_total_different_lmaxT_{fg_model}.png',bbox_inches='tight')

    plt.clf()
    plt.axhline(y=0, color='gray', alpha=0.5, linestyle='--')
    # Bispectrum and trispectrum
    bispec_3500_standard = cross_gmv_3500_cinv[:,0] - np.array(binned_input_clkk)
    #bispec_3500_mh = cross_gmv_3500_cinv[:,1] - np.array(binned_input_clkk)
    #bispec_3500_crossilc_onesed = cross_gmv_3500_cinv[:,2] - np.array(binned_input_clkk)
    bispec_4000_standard = cross_gmv_4000_cinv[:,0] - np.array(binned_input_clkk)
    plt.errorbar(bin_centers, binned_bias_gmv_4000_cinv[:,0]/binned_input_clkk, yerr=binned_uncertainty_gmv_4000_cinv[:,0]/binned_input_clkk, color='cornflowerblue', marker='x', markerfacecolor='none', markeredgecolor='cornflowerblue', linestyle='--', ms=7, alpha=0.8, label="Standard GMV Total Bias")
    plt.plot(bin_centers, bispec_4000_standard/binned_input_clkk, color='goldenrod', marker='o', markerfacecolor='none', markeredgecolor='goldenrod', linestyle='--', ms=7, alpha=0.8, label="Standard GMV Bispectrum")
    plt.plot(bin_centers, (binned_bias_gmv_4000_cinv[:,0]-bispec_4000_standard)/binned_input_clkk, color='lightgreen', marker='s', markerfacecolor='none', markeredgecolor='lightgreen', linestyle='--', ms=7, alpha=0.8, label="Standard GMV Total Bias - Bispectrum (Trispectrum)")
    #plt.errorbar(bin_centers, binned_bias_gmv_3500_cinv[:,0]/binned_input_clkk, yerr=binned_uncertainty_gmv_3500_cinv[:,0]/binned_input_clkk, color='cornflowerblue', marker='x', markerfacecolor='none', markeredgecolor='cornflowerblue', linestyle='--', ms=7, alpha=0.8, label="Standard GMV Total Bias")
    #plt.plot(bin_centers, bispec_3500_standard/binned_input_clkk, color='goldenrod', marker='o', markerfacecolor='none', markeredgecolor='goldenrod', linestyle='--', ms=7, alpha=0.8, label="Standard GMV Bispectrum")
    #plt.plot(bin_centers, (binned_bias_gmv_3500_cinv[:,0]-bispec_3500_standard)/binned_input_clkk, color='lightgreen', marker='s', markerfacecolor='none', markeredgecolor='lightgreen', linestyle='--', ms=7, alpha=0.8, label="Standard GMV Total Bias - Bispectrum (Trispectrum)")
    plt.xlabel('$L$')
    plt.title(f'Lensing Bias from Agora Sims / Input Kappa Spectrum, Split Up, lmaxT = 4000',pad=10)
    plt.legend(loc='lower left', fontsize='small')
    plt.xscale('log')
    plt.xlim(50,3001)
    plt.savefig(dir_out+f'/figs/bias_split_up_agora.png',bbox_inches='tight')

    plt.clf()
    # Bias vs uncertainty plot
    plt.figure(figsize=(10,6))
    cm_blue = mcolors.LinearSegmentedColormap.from_list("cm_blue", ["#addfed", "#000080"])
    cm_green = mcolors.LinearSegmentedColormap.from_list("cm_green", ["#abdbbe", "#00441b"])
    cm_orange = mcolors.LinearSegmentedColormap.from_list("cm_orange", ["#ffe0c2", "#f5800f"])
    sc1 = plt.scatter(binned_uncertainty_gmv_3500_cinv[:,0]/binned_input_clkk, np.abs(binned_bias_gmv_3500_cinv[:,0]/binned_input_clkk - mc_corr_gmv_3500_cinv[:,0]), c=bin_centers, cmap=cm_blue, marker='x', s=50, vmin=150, vmax=2700, label="Standard GMV")
    sc2 = plt.scatter(binned_uncertainty_gmv_3500_cinv[:,1]/binned_input_clkk, np.abs(binned_bias_gmv_3500_cinv[:,1]/binned_input_clkk - mc_corr_gmv_3500_cinv[:,1]), c=bin_centers, cmap=cm_green, marker='x', s=50, vmin=150, vmax=2700, label="Gradient Cleaning GMV")
    sc3 = plt.scatter(binned_uncertainty_gmv_3500_cinv[:,2]/binned_input_clkk, np.abs(binned_bias_gmv_3500_cinv[:,2]/binned_input_clkk - mc_corr_gmv_3500_cinv[:,2]), c=bin_centers, cmap=cm_orange, marker='x', s=50, vmin=150, vmax=2700, label="Cross-ILC GMV")
    plt.ylabel("|Lensing Bias from Agora Sims / Input Kappa Spectrum|")
    plt.xlabel('Uncertainty / Input Kappa Spectrum')
    plt.title(f'|Lensing Bias| vs Uncertainty, lmaxT = 3500',pad=10)
    plt.legend(loc='lower left', fontsize='small')
    plt.colorbar(sc1, label='L bins')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(dir_out+f'/figs/bias_vs_uncertainty_{fg_model}.png',bbox_inches='tight')

    plt.clf()
    # Define three larger bins
    bin_ranges = [(50, 1000), (1000, 2000), (2000, 3000)]
    # Initialize lists to store per-bin values
    mean_uncertainties = []
    mean_biases = []
    # Compute means per bin
    for (a, b) in bin_ranges:
        mask = (bin_centers >= a) & (bin_centers < b)
        # Compute mean
        mean_uncertainty = np.mean(binned_uncertainty_gmv_3500_cinv[mask] / binned_input_clkk[mask, np.newaxis], axis=0)
        mean_bias = np.mean(np.abs(binned_bias_gmv_3500_cinv[mask] / binned_input_clkk[mask, np.newaxis] - mc_corr_gmv_3500_cinv[mask]), axis=0)
        mean_uncertainties.append(mean_uncertainty)
        mean_biases.append(mean_bias)
    # Convert lists to numpy arrays for easy indexing
    mean_uncertainties = np.array(mean_uncertainties)  # Shape (3, 3)
    mean_biases = np.array(mean_biases)  # Shape (3, 3)
    # Create 3 subplots stacked vertically
    fig, axes = plt.subplots(3, 1, figsize=(8,9))
    plt.subplots_adjust(hspace=0.1, top=0.95, bottom=0.09, left=0.13)
    # Loop over each bin and plot data on the corresponding subplot
    for i, ax in enumerate(axes):  # Loop through the subplots
        # Extract uncertainty and bias for the current bin
        x_vals = mean_uncertainties[i]
        y_vals = mean_biases[i]
        # Plot points for each estimator within this bin
        ax.scatter(x_vals[0], y_vals[0], color="cornflowerblue", marker='x', s=80, label=f"Standard GMV")
        ax.scatter(x_vals[1], y_vals[1], color="darkgreen", marker='x', s=80, label=f"Gradient Cleaning GMV")
        ax.scatter(x_vals[2], y_vals[2], color="darkorange", marker='x', s=80, label=f"Cross-ILC GMV")
        # Ensure the line is horizontal by sorting x_vals correctly
        sorted_indices = np.argsort(y_vals)
        ax.plot(x_vals[sorted_indices], y_vals[sorted_indices], linestyle='dashed', color='gray', alpha=0.7, zorder=-1)
        # Set labels and title for each subplot
        #ax.set_xscale('log')
        #ax.set_yscale('log')
        ax.text(0.95, 0.95, f'{bin_ranges[i][0]} < L < {bin_ranges[i][1]}', transform=ax.transAxes, ha='right', va='top', fontsize=14)
        if i==2:
            ax.legend(loc='lower left', fontsize='small')
    fig.suptitle(f'|Lensing Bias| vs Uncertainty, lmaxT = 3500',fontsize=14)
    fig.text(0.5, 0.04, 'Uncertainty / Input Kappa Spectrum', ha='center', fontsize=12)
    if fg_model == 'agora':
        fig.text(0.04, 0.5, '|Lensing Bias from Agora Sims / Input Kappa Spectrum|', va='center', rotation='vertical', fontsize=12)
    else:
        fig.text(0.04, 0.5, '|Lensing Bias from WebSky Sims / Input Kappa Spectrum|', va='center', rotation='vertical', fontsize=12)
    plt.savefig(dir_out + f'/figs/bias_vs_uncertainty_three_bins_{fg_model}.png', bbox_inches='tight')

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
    plt.savefig(dir_out+f'/figs/bias_over_uncertainty_total_{fg_model}.png',bbox_inches='tight')
'''

def test_n0():
    '''
    Compare N0 for different cases.
    '''
    fg_model = 'agora'
    lmax = 4096
    l = np.arange(0,lmax+1)
    lbins = np.logspace(np.log10(50),np.log10(3000),20)
    bin_centers = (lbins[:-1] + lbins[1:]) / 2
    digitized = np.digitize(l, lbins)
    # Input kappa
    if fg_model == 'agora':
        klm = hp.almxfl(hp.read_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_spt3g_input_plm_lmax{lmax}.fits'),(l*(l+1))/2)
        klm = utils.reduce_lmax(klm,lmax=lmax)
    else:
        kap = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/websky/kap.fits')
        klm = hp.map2alm(kap)
        klm = utils.reduce_lmax(klm,lmax=lmax)
    input_clkk = hp.alm2cl(klm,klm,lmax=lmax)
    binned_input_clkk = np.array([input_clkk[digitized == i].mean() for i in range(1, len(lbins))])
    # Get output directory
    config_file = 'test_yuka_lmaxT3500.yaml'
    config = utils.parse_yaml(config_file)
    dir_out = config['dir_out']
    n0_n1_sims=np.arange(249)+1
    # lmaxT = 3500, no T3, cinv
    config_file='test_yuka_lmaxT3500.yaml'
    config = utils.parse_yaml(config_file)

    #=========================================================================#

    # SIM-BASED N0
    # GMV
    n0_lmaxT3500_cinv_standard = get_n0(sims=n0_n1_sims,qetype='gmv_cinv',config=config,append='standard',fg_model=fg_model)
    n0_lmaxT3500_cinv_standard['total'] *= (l*(l+1))**2/4
    binned_n0_lmaxT3500_cinv_standard = [n0_lmaxT3500_cinv_standard['total'][digitized == i].mean() for i in range(1, len(lbins))]

    n0_lmaxT3500_cinv_mh = get_n0(sims=n0_n1_sims,qetype='gmv_cinv',config=config,append='mh',fg_model=fg_model)
    n0_lmaxT3500_cinv_mh['total'] *= (l*(l+1))**2/4
    binned_n0_lmaxT3500_cinv_mh = [n0_lmaxT3500_cinv_mh['total'][digitized == i].mean() for i in range(1, len(lbins))]

    n0_lmaxT3500_cinv_crossilc = get_n0(sims=n0_n1_sims,qetype='gmv_cinv',config=config,append='crossilc_onesed',fg_model=fg_model)
    n0_lmaxT3500_cinv_crossilc['total'] *= (l*(l+1))**2/4
    binned_n0_lmaxT3500_cinv_crossilc = [n0_lmaxT3500_cinv_crossilc['total'][digitized == i].mean() for i in range(1, len(lbins))]

    # SQE
    # lmaxT = 3000
    filename = dir_out+f'/n0/n0_249simpairs_healqest_sqe_lmaxT3000_lmaxP4096_nside2048_agora_standard_resp_from_sims.pkl'
    n0_sqe_3000 = pickle.load(open(filename,'rb'))
    n0_sqe_3000_total = n0_sqe_3000['total'] * (l*(l+1))**2/4
    # lmaxT = 3500
    filename = dir_out+f'/n0/n0_249simpairs_healqest_sqe_lmaxT3500_lmaxP4096_nside2048_agora_standard_resp_from_sims.pkl'
    n0_sqe_3500 = pickle.load(open(filename,'rb'))
    n0_sqe_3500_total = n0_sqe_3500['total'] * (l*(l+1))**2/4

    #=========================================================================#

    # RDN0
    rdn0_lmaxT3500_cinv_standard = get_rdn0(sims=n0_n1_sims,qetype='gmv_cinv',config=config,append='standard',fg_model=fg_model)
    rdn0_lmaxT3500_cinv_standard *= (l*(l+1))**2/4
    binned_rdn0_lmaxT3500_cinv_standard = [rdn0_lmaxT3500_cinv_standard[digitized == i].mean() for i in range(1, len(lbins))]

    #=========================================================================#

    # Response for standard, lmaxT = 3500
    ests = ['TT', 'EE', 'TE', 'ET', 'TB', 'BT', 'EB', 'BE']
    resps = np.zeros((len(l),len(ests)), dtype=np.complex_)
    resps_sqe = np.zeros((len(l),len(ests)), dtype=np.complex_)
    inv_resps = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
    inv_resps_sqe = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
    for i, est in enumerate(ests):
        resps[:,i] = np.load(dir_out+f'/resp/sim_resp_250sims_gmv_cinv_est{est}_lmaxT3500_lmaxP4096_lmin300_cltypelcmb_agora_standard.npy')
        resps_sqe[:,i] = np.load(dir_out+f'/resp/sim_resp_250sims_sqe_est{est}_lmaxT3500_lmaxP4096_lmin300_cltypelcmb_agora_standard.npy')
        inv_resps[1:,i] = 1/(resps)[1:,i]
        inv_resps_sqe[1:,i] = 1/(resps_sqe)[1:,i]
    resp = np.sum(resps, axis=1)
    resp_sqe = np.sum(resps_sqe, axis=1)
    inv_resp = np.zeros_like(l,dtype=np.complex_); inv_resp[1:] = 1/(resp)[1:]
    inv_resp_sqe = np.zeros_like(l,dtype=np.complex_); inv_resp_sqe[1:] = 1/(resp_sqe)[1:]

    #=========================================================================#

    # Ratios
    invR_vs_n0_ratio = inv_resp*(l*(l+1))**2/4 / n0_lmaxT3500_cinv_standard['total']
    invR_vs_rdn0_ratio = inv_resp*(l*(l+1))**2/4 / rdn0_lmaxT3500_cinv_standard
    n0_vs_rdn0_ratio = n0_lmaxT3500_cinv_standard['total'] / rdn0_lmaxT3500_cinv_standard

    # Plot
    plt.clf()
    plt.axhline(y=1, color='gray', alpha=0.5, linestyle='--')
    plt.plot(l, invR_vs_n0_ratio, color='lightcoral', alpha=0.8, linestyle='--',label='Standard GMV 1/R / N0')
    plt.plot(l, invR_vs_rdn0_ratio, color='lightgreen', alpha=0.8, linestyle='--',label='Standard GMV 1/R / RDN0')
    plt.plot(l, n0_vs_rdn0_ratio, color='cornflowerblue', alpha=0.8, linestyle='--',label='Standard GMV N0 / RDN0')
    plt.title('1/R vs N0 comparison, lmaxT = 3500, cinv')
    plt.xlabel('$\ell$')
    plt.legend(loc='upper left', fontsize='small')
    plt.xscale('log')
    plt.xlim(50,3001)
    plt.ylim(0.8,1.2)
    plt.savefig(dir_out+f'/figs/invR_vs_n0_comparison_standard_lmaxT3500_{fg_model}.png',bbox_inches='tight')

    '''
    # Plot
    plt.clf()
    plt.axhline(y=1, color='k', linestyle='--')
    plt.plot(l, n0_sqe_3500_total/n0_lmaxT3500_cinv_standard['total'], color='firebrick', alpha=0.8, linestyle='-',label='lmaxT = 3500 N0 Ratio')
    #plt.plot(l, inv_resp_sqe/inv_resp, color='darkblue', alpha=0.8, linestyle='-',label='lmaxT = 3500 1/R Ratio')
    plt.xlabel('$L$')
    plt.ylabel('N0 Ratio SQE/GMV')
    plt.title(f'GMV N0 Improvement Over SQE')
    plt.legend(loc='upper left', fontsize='small')
    plt.xscale('log')
    plt.ylim(0.75,1.25)
    plt.xlim(10,lmax)
    plt.savefig(dir_out+f'/figs/gmv_n0_improvement_lmaxT3500_standard.png',bbox_inches='tight')
    '''

    '''
    plt.figure(figsize=(10,6))
    plt.clf()
    #cm_blue = plt.cm.get_cmap('Blues')
    #cm_green = plt.cm.get_cmap('Greens')
    #cm_orange = plt.cm.get_cmap('Oranges')
    cm_blue = mcolors.LinearSegmentedColormap.from_list("cm_blue", ["#addfed", "#000080"])
    cm_green = mcolors.LinearSegmentedColormap.from_list("cm_green", ["#abdbbe", "#00441b"])
    cm_orange = mcolors.LinearSegmentedColormap.from_list("cm_orange", ["#ffe0c2", "#f5800f"])
    sc1 = plt.scatter(binned_bias_gmv_3500_cinv_n0[:,0]/binned_input_clkk, binned_n0_lmaxT3500_cinv_standard, c=bin_centers, cmap=cm_blue, marker='.', vmin=150, vmax=2700, label='Standard GMV (lmaxT = 3500)')
    sc2 = plt.scatter(binned_bias_gmv_3500_cinv_n0[:,1]/binned_input_clkk, binned_n0_lmaxT3500_cinv_mh, c=bin_centers, cmap=cm_green, marker='.', vmin=150, vmax=2700, label='MH GMV (lmaxT = 3500)')
    sc3 = plt.scatter(binned_bias_gmv_3500_cinv_n0[:,2]/binned_input_clkk, binned_n0_lmaxT3500_cinv_crossilc, c=bin_centers, cmap=cm_orange, marker='.', vmin=150, vmax=2700, label='Cross-ILC GMV (lmaxT = 3500)')
    plt.scatter(binned_bias_gmv_4000_cinv_n0[:,0]/binned_input_clkk, binned_n0_lmaxT4000_cinv_standard, cmap=cm_blue, c=bin_centers, marker='x', vmin=150, vmax=2700, label='Standard GMV (lmaxT = 4000)')
    plt.scatter(binned_bias_gmv_4000_cinv_n0[:,1]/binned_input_clkk, binned_n0_lmaxT4000_cinv_mh, cmap=cm_green, c=bin_centers, marker='x', vmin=150, vmax=2700, label='MH GMV (lmaxT = 4000)')
    plt.scatter(binned_bias_gmv_4000_cinv_n0[:,2]/binned_input_clkk, binned_n0_lmaxT4000_cinv_crossilc, c=bin_centers, cmap=cm_orange, marker='x', vmin=150, vmax=2700, label='Cross-ILC GMV (lmaxT = 4000)')
    plt.ylabel("$[\ell(\ell+1)]^2$$RDN_0$ / 4 $[\mu K^2]$")
    plt.xlabel('Lensing Bias / Input Kappa Spectrum')
    plt.title(f'Reconstruction Noise vs Lensing Bias, Cinv-Style GMV')
    plt.legend(loc='lower right', fontsize='small')
    plt.colorbar(sc1, label='L bins')
    plt.yscale('log')
    plt.xlim(-0.2,0.1)
    plt.ylim(1e-8,3e-7)
    plt.savefig(dir_out+f'/figs/n0_vs_bias_different_lmaxT_{fg_model}.png',bbox_inches='tight')
    '''

def get_lensing_bias(config, append_list, cinv=False, sqe=False, sims=np.arange(250)+1, n0_n1_sims=np.arange(249)+1, withT3=False, fg_model='agora'):
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
    # Theory spectrum
    clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
    ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
    clkk = slpp * (l*(l+1))**2/4
    binned_clkk = np.array([clkk[digitized == i].mean() for i in range(1, len(lbins))])
    # Input kappa
    if fg_model == 'agora':
        klm = hp.almxfl(hp.read_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_spt3g_input_plm_lmax{lmax}.fits'),(l*(l+1))/2)
        klm = utils.reduce_lmax(klm,lmax=lmax)
    else:
        kap = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/websky/kap.fits')
        klm = hp.map2alm(kap)
        klm = utils.reduce_lmax(klm,lmax=lmax)
    input_clkk = hp.alm2cl(klm,klm,lmax=lmax)
    binned_input_clkk = np.array([input_clkk[digitized == i].mean() for i in range(1, len(lbins))])

    # Bias
    bias = np.zeros((len(l),len(append_list)), dtype=np.complex_)
    binned_bias = np.zeros((len(bin_centers),len(append_list)), dtype=np.complex_)
    binned_debiased_clkk = np.zeros((len(bin_centers),len(append_list)), dtype=np.complex_)
    plm_resp_corrected = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_gmv_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_standard_cinv.npy')),len(append_list)), dtype=np.complex_)
    # Uncertainty saved from before
    binned_uncertainty = np.zeros((len(bin_centers),len(append_list)), dtype=np.complex_)
    # Cross with input
    cross = np.zeros((len(bin_centers),len(append_list)), dtype=np.complex_)

    for j, append in enumerate(append_list):
        print(f'Doing {append}!')
        u = None

        if append == 'standard':
            ests = ['TT', 'EE', 'TE', 'ET', 'TB', 'BT', 'EB', 'BE']
            if sqe or cinv:
                resps = np.zeros((len(l),len(ests)), dtype=np.complex_)
                inv_resps = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
                for i, est in enumerate(ests):
                    if sqe:
                        # Get SQE response
                        resps[:,i] = get_sim_response(est,config,gmv=False,append=append,sims=sims,cinv=False,fg_model=fg_model)
                    elif cinv:
                        # GMV response
                        resps[:,i] = get_sim_response(est,config,gmv=True,cinv=True,append=append,sims=sims,fg_model=fg_model)
                    inv_resps[1:,i] = 1/(resps)[1:,i]
                resp = np.sum(resps, axis=1)
            else:
                resp = get_sim_response('all',config,gmv=True,append=append,sims=sims,cinv=False,fg_model=fg_model)
            inv_resp = np.zeros_like(l,dtype=np.complex_); inv_resp[1:] = 1/(resp)[1:]

        elif append == 'mh' or append == 'crossilc_onesed':
            ests = ['T1T2', 'T2T1', 'EE', 'E2E1', 'TE', 'T2E1', 'ET', 'E2T1', 'TB', 'T2B1', 'BT', 'B2T1', 'EB', 'E2B1', 'BE', 'B2E1']
            if sqe or cinv:
                resps = np.zeros((len(l),len(ests)), dtype=np.complex_)
                inv_resps = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
                for i, est in enumerate(ests):
                    if sqe:
                        # Get SQE response
                        resps[:,i] = get_sim_response(est,config,gmv=False,append=append,sims=sims,cinv=False,withT3=withT3,fg_model=fg_model)
                    elif cinv:
                        # GMV response
                        resps[:,i] = get_sim_response(est,config,gmv=True,cinv=True,append=append,sims=sims,withT3=withT3,fg_model=fg_model)
                    inv_resps[1:,i] = 1/(resps)[1:,i]
                resp = 0.5*np.sum(resps, axis=1)
            else:
                resp = get_sim_response('all',config,gmv=True,append=append,sims=sims,cinv=False,withT3=withT3,fg_model=fg_model)
            inv_resp = np.zeros_like(l,dtype=np.complex_); inv_resp[1:] = 1/(resp)[1:]

        # Get N0 correction (remember these still need (l*(l+1))**2/4 factor to convert to kappa)
        if cinv:
            n0 = get_n0(sims=n0_n1_sims,qetype='gmv_cinv',config=config,append=append,withT3=withT3,fg_model=fg_model)
        elif sqe:
            n0 = get_n0(sims=n0_n1_sims,qetype='sqe',config=config,append=append,withT3=withT3,fg_model=fg_model)
        else:
            n0 = get_n0(sims=n0_n1_sims,qetype='gmv',config=config,append=append,withT3=withT3,fg_model=fg_model)
        n0_total = n0['total'] * (l*(l+1))**2/4
        #n0_total = n0 * (l*(l+1))**2/4

        # N1
        if sqe:
            n1 = get_n1(sims=n0_n1_sims,qetype='sqe',config=config,append=append,withT3=withT3,fg_model=fg_model)
        elif cinv:
            n1 = get_n1(sims=n0_n1_sims,qetype='gmv_cinv',config=config,append=append,withT3=withT3,fg_model=fg_model)
        else:
            n1 = get_n1(sims=n0_n1_sims,qetype='gmv',config=config,append=append,withT3=withT3,fg_model=fg_model)
        n1_total = n1['total'] * (l*(l+1))**2/4

        if cinv:
            if append == 'mh' or append == 'crossilc_onesed':
                # Load GMV plms, cinv-style
                if withT3:
                    plms = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_gmv_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv_withT3.npy')),len(ests)), dtype=np.complex_)
                else:
                    plms = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_gmv_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv_noT3.npy')),len(ests)), dtype=np.complex_)
                for i, est in enumerate(ests):
                    if withT3:
                        plms[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv_withT3.npy')
                    else:
                        plms[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv_noT3.npy')
                plm = 0.5*np.sum(plms, axis=1)
            else:
                # Load GMV plms, cinv-style
                plms = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_gmv_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv.npy')),len(ests)), dtype=np.complex_)
                for i, est in enumerate(ests):
                    # Commented out below: I'm testing the case without NG fg, should give zero for bias
                    #plms[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_1_seed2_1_lmaxT3000_lmaxP4096_nside2048_standard_cinv.npy')
                    #TODO
                    #plms[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv_nofg.npy')
                    #plms[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv_nonoise.npy')
                    #plms[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv_lcmbonly.npy')
                    #plms[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv_tszcibonly.npy')
                    plms[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv.npy')
                plm = np.sum(plms, axis=1)
        elif sqe:
            if append == 'mh' or append == 'crossilc_onesed':
                # Load SQE plms
                if withT3:
                    plms = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_sqe_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_withT3.npy')),len(ests)), dtype=np.complex_)
                else:
                    plms = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_sqe_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_noT3.npy')),len(ests)), dtype=np.complex_)
                for i, est in enumerate(ests):
                    if withT3:
                        plms[:,i] = np.load(dir_out+f'/plm_{est}_healqest_sqe_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_withT3.npy')
                    else:
                        plms[:,i] = np.load(dir_out+f'/plm_{est}_healqest_sqe_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_noT3.npy')
                plm = 0.5*np.sum(plms, axis=1)
            else:
                # Load SQE plms
                plms = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_sqe_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy')),len(ests)), dtype=np.complex_)
                for i, est in enumerate(ests):
                    #TODO
                    #plms[:,i] = np.load(dir_out+f'/plm_{est}_healqest_sqe_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_nofg.npy')
                    #plms[:,i] = np.load(dir_out+f'/plm_{est}_healqest_sqe_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_nonoise.npy')
                    #plms[:,i] = np.load(dir_out+f'/plm_{est}_healqest_sqe_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_tszcibonly.npy')
                    plms[:,i] = np.load(dir_out+f'/plm_{est}_healqest_sqe_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy')
                plm = np.sum(plms, axis=1)
        else:
            # Load GMV plms, not cinv-style
            if withT3:
                plm_gmv = np.load(dir_out+f'/plm_all_healqest_gmv_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_withT3.npy')
            else:
                plm_gmv = np.load(dir_out+f'/plm_all_healqest_gmv_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_noT3.npy')

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
        binned_debiased_clkk[:,j] = binned_auto_debiased
        plm_resp_corrected[:,j] = plm_resp_corr

        # Get uncertainty
        if cinv:
            if append == 'mh' or append == 'crossilc_onesed':
                if withT3:
                    binned_uncertainty[:,j] = np.load(dir_out+f'/measurement_uncertainty/binned_measurement_uncertainty_lmaxT{lmaxT}_{fg_model}_{append}_cinv_withT3.npy')
                else:
                    binned_uncertainty[:,j] = np.load(dir_out+f'/measurement_uncertainty/binned_measurement_uncertainty_lmaxT{lmaxT}_{fg_model}_{append}_cinv_noT3.npy')
            else:
                binned_uncertainty[:,j] = np.load(dir_out+f'/measurement_uncertainty/binned_measurement_uncertainty_lmaxT{lmaxT}_{fg_model}_{append}_cinv.npy')
        elif sqe:
            if append == 'mh' or append == 'crossilc_onesed':
                if withT3:
                    binned_uncertainty[:,j] = np.load(dir_out+f'/measurement_uncertainty/binned_measurement_uncertainty_lmaxT{lmaxT}_{fg_model}_{append}_sqe_withT3.npy')
                else:
                    binned_uncertainty[:,j] = np.load(dir_out+f'/measurement_uncertainty/binned_measurement_uncertainty_lmaxT{lmaxT}_{fg_model}_{append}_sqe_noT3.npy')
            else:
                binned_uncertainty[:,j] = np.load(dir_out+f'/measurement_uncertainty/binned_measurement_uncertainty_lmaxT{lmaxT}_{fg_model}_{append}_sqe.npy')
        else:
            if append == 'mh' or append == 'crossilc_onesed':
                if withT3:
                    binned_uncertainty[:,j] = np.load(dir_out+f'/measurement_uncertainty/binned_measurement_uncertainty_lmaxT{lmaxT}_{fg_model}_{append}_gmv_withT3.npy')
                else:
                    binned_uncertainty[:,j] = np.load(dir_out+f'/measurement_uncertainty/binned_measurement_uncertainty_lmaxT{lmaxT}_{fg_model}_{append}_gmv_noT3.npy')
            else:
                binned_uncertainty[:,j] = np.load(dir_out+f'/measurement_uncertainty/binned_measurement_uncertainty_lmaxT{lmaxT}_{fg_model}_{append}_gmv.npy')

    ret = {}
    ret['append_list'] = append_list
    ret['binned_bias'] = binned_bias
    ret['binned_uncertainty'] = binned_uncertainty
    ret['plm_resp_corrected'] = plm_resp_corrected
    ret['binned_debiased_clkk'] = binned_debiased_clkk
    ret['cross'] = cross
    if cinv:
        s = 'cinv'
    elif sqe:
        s = 'sqe'
    else:
        s = 'gmv'
    '''
    if not os.path.isfile(dir_out+f'/agora_reconstruction/agora_reconstruction_results_lmaxT{lmaxT}_{fg_model}_{s}_nofg.npy'):
        with open(dir_out+f'/agora_reconstruction/agora_reconstruction_results_lmaxT{lmaxT}_{fg_model}_{s}_nofg.npy', 'wb') as f:
            pickle.dump(ret, f)
    '''

    return ret

def get_n0(sims,qetype,config,append,cmbonly=False,withT3=False,fg_model='agora'):
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
    if append=='crossilc_onesed' or append=='mh':
        if withT3:
            filename = dir_out+f'/n0/n0_{num}simpairs_healqest_{qetype}_withT3_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_resp_from_sims.pkl'
        else:
            filename = dir_out+f'/n0/n0_{num}simpairs_healqest_{qetype}_noT3_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_resp_from_sims.pkl'
    else:
        filename = dir_out+f'/n0/n0_{num}simpairs_healqest_{qetype}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_resp_from_sims.pkl'
    if os.path.isfile(filename):
        n0 = pickle.load(open(filename,'rb'))
    else:
        print(f"File {filename} doesn't exist!")

    return n0

def get_n1(sims,qetype,config,append,withT3=False,fg_model='agora'):
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
    if append=='crossilc_onesed' or append=='mh':
        if withT3:
            filename = dir_out+f'/n1/n1_{num}simpairs_healqest_{qetype}_withT3_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_resp_from_sims.pkl'
        else:
            filename = dir_out+f'/n1/n1_{num}simpairs_healqest_{qetype}_noT3_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_resp_from_sims.pkl'
    else:
        filename = dir_out+f'/n1/n1_{num}simpairs_healqest_{qetype}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_resp_from_sims.pkl'
    if os.path.isfile(filename):
        n1 = pickle.load(open(filename,'rb'))
    else:
        print(f"File {filename} doesn't exist!")

    return n1

def get_rdn0(sims,qetype,config,append,withT3=False,fg_model='agora'):
    '''
    Only returns total N0, not for each estimator.
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

    if append=='crossilc_onesed' or append=='mh':
        if withT3:
            filename = dir_out+f'/n0/rdn0_{num}simpairs_healqest_{qetype}_withT3_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_resp_from_sims.pkl'
        else:
            filename = dir_out+f'/n0/rdn0_{num}simpairs_healqest_{qetype}_noT3_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_resp_from_sims.pkl'
    else:
        filename = dir_out+f'/n0/rdn0_{num}simpairs_healqest_{qetype}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_resp_from_sims.pkl'

    if os.path.isfile(filename):
        print(f'Getting RDN0: {filename}')
        rdn0 = pickle.load(open(filename,'rb'))

    elif qetype == 'gmv' or qetype == 'gmv_cinv' or qetype == 'sqe':
        if append == 'standard':
            ests = ['TT', 'EE', 'TE', 'ET', 'TB', 'BT', 'EB', 'BE']
        elif append == 'mh' or append == 'crossilc_onesed':
            ests = ['T1T2', 'T2T1', 'EE', 'E2E1', 'TE', 'T2E1', 'ET', 'E2T1', 'TB', 'T2B1', 'BT', 'B2T1', 'EB', 'E2B1', 'BE', 'B2E1']

        # Get response
        if qetype == 'gmv':
            resp = get_sim_response('all',config,gmv=True,append=append,sims=np.append(sims,num+1),cinv=False,withT3=withT3,fg_model=fg_model)
        elif qetype == 'gmv_cinv' or qetype == 'sqe':
            resps = np.zeros((len(l),len(ests)), dtype=np.complex_)
            inv_resps = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
            for i, est in enumerate(ests):
                if qetype == 'gmv_cinv':
                    resps[:,i] = get_sim_response(est,config,gmv=True,cinv=True,append=append,sims=np.append(sims,num+1),withT3=withT3,fg_model=fg_model)
                elif qetype == 'sqe':
                    resps[:,i] = get_sim_response(est,config,gmv=False,cinv=False,append=append,sims=np.append(sims,num+1),withT3=withT3,fg_model=fg_model)
                inv_resps[1:,i] = 1/(resps)[1:,i]
            if append == 'standard':
                resp = np.sum(resps, axis=1)
            elif append == 'mh' or append == 'crossilc_onesed':
                resp = 0.5*np.sum(resps, axis=1)
        inv_resp = np.zeros(len(l),dtype=np.complex_); inv_resp[1:] = 1./(resp)[1:]

        # Get sim-based N0
        if qetype == 'gmv':
            n0 = get_n0(sims=sims,qetype='gmv',config=config,append=append,withT3=withT3,fg_model=fg_model)
        elif qetype == 'gmv_cinv':
            n0 = get_n0(sims=sims,qetype='gmv_cinv',config=config,append=append,withT3=withT3,fg_model=fg_model)
        elif qetype == 'sqe':
            n0 = get_n0(sims=sims,qetype='sqe',config=config,append=append,withT3=withT3,fg_model=fg_model)
        n0_total = n0['total']

        rdn0 = 0
        for i, sim in enumerate(sims):
            if qetype == 'gmv':
                if withT3:
                    plm_ir = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_withT3.npy')
                    plm_ri = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_withT3.npy')
                else:
                    plm_ir = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_noT3.npy')
                    plm_ri = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_noT3.npy')
            elif qetype == 'gmv_cinv':
                if append == 'standard':
                    # Get ir sims
                    if os.path.isfile(dir_out+f'/plm_summed_healqest_gmv_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv.npy'):
                        plm_ir = np.load(dir_out+f'/plm_summed_healqest_gmv_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv.npy')
                    else:
                        plms_ir = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_gmv_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv.npy')),len(ests)), dtype=np.complex_)
                        for i, est in enumerate(ests):
                            plms_ir[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv.npy')
                        plm_ir = np.sum(plms_ir, axis=1)
                        np.save(dir_out+f'/plm_summed_healqest_gmv_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv.npy', plm_ir)
                        for i, est in enumerate(ests):
                            os.remove(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv.npy')
                elif append == 'mh' or append == 'crossilc_onesed':
                    # Get ir sims
                    if os.path.isfile(dir_out+f'/plm_summed_healqest_gmv_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv_withT3.npy') and withT3:
                        plm_ir = np.load(dir_out+f'/plm_summed_healqest_gmv_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv_withT3.npy')
                    elif os.path.isfile(dir_out+f'/plm_summed_healqest_gmv_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv_noT3.npy') and not withT3:
                        plm_ir = np.load(dir_out+f'/plm_summed_healqest_gmv_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv_noT3.npy')
                    else:
                        if withT3:
                            plms_ir = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_gmv_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv_withT3.npy')),len(ests)), dtype=np.complex_)
                        else:
                            plms_ir = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_gmv_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv_noT3.npy')),len(ests)), dtype=np.complex_)
                        for i, est in enumerate(ests):
                            if withT3:
                                plms_ir[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv_withT3.npy')
                            else:
                                plms_ir[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv_noT3.npy')
                        plm_ir = 0.5*np.sum(plms_ir, axis=1)
                        if withT3:
                            np.save(dir_out+f'/plm_summed_healqest_gmv_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv_withT3.npy', plm_ir)
                        else:
                            np.save(dir_out+f'/plm_summed_healqest_gmv_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv_noT3.npy', plm_ir)
                        for i, est in enumerate(ests):
                            if withT3:
                                os.remove(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv_withT3.npy')
                            else:
                                os.remove(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv_noT3.npy')
                if append == 'standard':
                    # Get ri sims
                    if os.path.isfile(dir_out+f'/plm_summed_healqest_gmv_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv.npy'):
                        plm_ri = np.load(dir_out+f'/plm_summed_healqest_gmv_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv.npy')
                    else:
                        plms_ri = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_gmv_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv.npy')),len(ests)), dtype=np.complex_)
                        for i, est in enumerate(ests):
                            plms_ri[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv.npy')
                        plm_ri = np.sum(plms_ri, axis=1)
                        np.save(dir_out+f'/plm_summed_healqest_gmv_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv.npy', plm_ri)
                        for i, est in enumerate(ests):
                            os.remove(dir_out+f'/plm_{est}_healqest_gmv_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv.npy')
                elif append == 'mh' or append == 'crossilc_onesed':
                    # Get ri sims
                    if os.path.isfile(dir_out+f'/plm_summed_healqest_gmv_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv_withT3.npy') and withT3:
                        plm_ri = np.load(dir_out+f'/plm_summed_healqest_gmv_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv_withT3.npy')
                    elif os.path.isfile(dir_out+f'/plm_summed_healqest_gmv_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv_noT3.npy') and not withT3:
                        plm_ri = np.load(dir_out+f'/plm_summed_healqest_gmv_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv_noT3.npy')
                    else:
                        if withT3:
                            plms_ri = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_gmv_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv_withT3.npy')),len(ests)), dtype=np.complex_)
                        else:
                            plms_ri = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_gmv_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv_noT3.npy')),len(ests)), dtype=np.complex_)
                        for i, est in enumerate(ests):
                            if withT3:
                                plms_ri[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv_withT3.npy')
                            else:
                                plms_ri[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv_noT3.npy')
                        plm_ri = 0.5*np.sum(plms_ri, axis=1)
                        if withT3:
                            np.save(dir_out+f'/plm_summed_healqest_gmv_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv_withT3.npy', plm_ri)
                        else:
                            np.save(dir_out+f'/plm_summed_healqest_gmv_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv_noT3.npy', plm_ri)
                        for i, est in enumerate(ests):
                            if withT3:
                                os.remove(dir_out+f'/plm_{est}_healqest_gmv_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv_withT3.npy')
                            else:
                                os.remove(dir_out+f'/plm_{est}_healqest_gmv_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv_noT3.npy')
            elif qetype == 'sqe':
                if append == 'standard':
                    # Get ir sims
                    if os.path.isfile(dir_out+f'/plm_summed_healqest_sqe_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy'):
                        plm_ir = np.load(dir_out+f'/plm_summed_healqest_sqe_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy')
                    else:
                        plms_ir = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_sqe_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy')),len(ests)), dtype=np.complex_)
                        for i, est in enumerate(ests):
                            plms_ir[:,i] = np.load(dir_out+f'/plm_{est}_healqest_sqe_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy')
                        plm_ir = np.sum(plms_ir, axis=1)
                        np.save(dir_out+f'/plm_summed_healqest_sqe_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy', plm_ir)
                        for i, est in enumerate(ests):
                            os.remove(dir_out+f'/plm_{est}_healqest_sqe_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy')
                elif append == 'mh' or append == 'crossilc_onesed':
                    # Get ir sims
                    if os.path.isfile(dir_out+f'/plm_summed_healqest_sqe_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_withT3.npy') and withT3:
                        plm_ir = np.load(dir_out+f'/plm_summed_healqest_sqe_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_withT3.npy')
                    elif os.path.isfile(dir_out+f'/plm_summed_healqest_sqe_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_noT3.npy') and not withT3:
                        plm_ir = np.load(dir_out+f'/plm_summed_healqest_sqe_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_noT3.npy')
                    else:
                        if withT3:
                            plms_ir = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_sqe_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_withT3.npy')),len(ests)), dtype=np.complex_)
                        else:
                            plms_ir = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_sqe_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_noT3.npy')),len(ests)), dtype=np.complex_)
                        for i, est in enumerate(ests):
                            if withT3:
                                plms_ir[:,i] = np.load(dir_out+f'/plm_{est}_healqest_sqe_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_withT3.npy')
                            else:
                                plms_ir[:,i] = np.load(dir_out+f'/plm_{est}_healqest_sqe_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_noT3.npy')
                        plm_ir = 0.5*np.sum(plms_ir, axis=1)
                        if withT3:
                            np.save(dir_out+f'/plm_summed_healqest_sqe_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_withT3.npy', plm_ir)
                        else:
                            np.save(dir_out+f'/plm_summed_healqest_sqe_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_noT3.npy', plm_ir)
                        for i, est in enumerate(ests):
                            if withT3:
                                os.remove(dir_out+f'/plm_{est}_healqest_sqe_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_withT3.npy')
                            else:
                                os.remove(dir_out+f'/plm_{est}_healqest_sqe_seed1_{sim}_seed2_r_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_noT3.npy')
                if append == 'standard':
                    # Get ri sims
                    if os.path.isfile(dir_out+f'/plm_summed_healqest_sqe_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy'):
                        plm_ri = np.load(dir_out+f'/plm_summed_healqest_sqe_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy')
                    else:
                        plms_ri = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_sqe_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy')),len(ests)), dtype=np.complex_)
                        for i, est in enumerate(ests):
                            plms_ri[:,i] = np.load(dir_out+f'/plm_{est}_healqest_sqe_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy')
                        plm_ri = np.sum(plms_ri, axis=1)
                        np.save(dir_out+f'/plm_summed_healqest_sqe_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy', plm_ri)
                        for i, est in enumerate(ests):
                            os.remove(dir_out+f'/plm_{est}_healqest_sqe_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy')
                elif append == 'mh' or append == 'crossilc_onesed':
                    # Get ri sims
                    if os.path.isfile(dir_out+f'/plm_summed_healqest_sqe_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_withT3.npy') and withT3:
                        plm_ri = np.load(dir_out+f'/plm_summed_healqest_sqe_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_withT3.npy')
                    elif os.path.isfile(dir_out+f'/plm_summed_healqest_sqe_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_noT3.npy') and not withT3:
                        plm_ri = np.load(dir_out+f'/plm_summed_healqest_sqe_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_noT3.npy')
                    else:
                        if withT3:
                            plms_ri = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_sqe_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_withT3.npy')),len(ests)), dtype=np.complex_)
                        else:
                            plms_ri = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_sqe_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_noT3.npy')),len(ests)), dtype=np.complex_)
                        for i, est in enumerate(ests):
                            if withT3:
                                plms_ri[:,i] = np.load(dir_out+f'/plm_{est}_healqest_sqe_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_withT3.npy')
                            else:
                                plms_ri[:,i] = np.load(dir_out+f'/plm_{est}_healqest_sqe_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_noT3.npy')
                        plm_ri = 0.5*np.sum(plms_ri, axis=1)
                        if withT3:
                            np.save(dir_out+f'/plm_summed_healqest_sqe_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_withT3.npy', plm_ri)
                        else:
                            np.save(dir_out+f'/plm_summed_healqest_sqe_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_noT3.npy', plm_ri)
                        for i, est in enumerate(ests):
                            if withT3:
                                os.remove(dir_out+f'/plm_{est}_healqest_sqe_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_withT3.npy')
                            else:
                                os.remove(dir_out+f'/plm_{est}_healqest_sqe_seed1_r_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_noT3.npy')

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

def get_sim_response(est,config,gmv,append,sims,filename=None,cinv=False,withT3=False,fg_model='agora'):
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
        fn += f'_lmaxT{lmaxT}_lmaxP{lmaxP}_lmin{lmin}_cltype{cltype}_{fg_model}_{append}'
        if append=='crossilc_onesed' or append=='mh':
            if withT3:
                fn += '_withT3'
            else:
                fn += '_noT3'
        filename = dir_out+f'/resp/sim_resp_{num}sims{fn}.npy'

    if os.path.isfile(filename):
        print('Loading from existing file!')
        sim_resp = np.load(filename)
    else:
        print(f"File {filename} doesn't exist!")
    return sim_resp

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

####################

analyze()
#test_n0()

#dir_out = '/oak/stanford/orgs/kipac/users/yukanaka/outputs/'
#fg_model = 'websky'
#lmax = 4096
#l = np.arange(0,lmax+1)
#lbins = np.logspace(np.log10(50),np.log10(3000),20)
#bin_centers = (lbins[:-1] + lbins[1:]) / 2
#digitized = np.digitize(l, lbins)
#config_file='test_yuka_lmaxT3500.yaml'
#config = utils.parse_yaml(config_file)
#kap = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/websky/kap.fits')
#klm = hp.map2alm(kap)
#klm = utils.reduce_lmax(klm,lmax=lmax)
#input_clkk = hp.alm2cl(klm,klm,lmax=lmax)
#binned_input_clkk = np.array([input_clkk[digitized == i].mean() for i in range(1, len(lbins))])
#append_list = ['standard']
#ret = get_lensing_bias(config,append_list,cinv=True,sqe=False,sims=np.arange(250)+1,n0_n1_sims=np.arange(249)+1,fg_model=fg_model)
#binned_bias_gmv_3500_cinv = ret['binned_bias']; binned_uncertainty_gmv_3500_cinv = ret['binned_uncertainty']
## Plot
#plt.figure(0)
#plt.clf()
#plt.axhline(y=0, color='gray', alpha=0.5, linestyle='--')
#plt.plot(bin_centers, binned_bias_gmv_3500_cinv[:,0]/binned_input_clkk, color='cornflowerblue', marker='o', linestyle='-', ms=3, alpha=0.8, label="Standard GMV")
#plt.xlabel('$L$')
#plt.title(f'Lensing Bias from WebSky Sims / Input Kappa Spectrum, lmaxT = 3500',pad=10)
#plt.legend(loc='upper left', fontsize='small')
#plt.xscale('log')
#plt.xlim(50,3001)
#plt.savefig(dir_out+f'/figs/bias_total_cinv_{fg_model}.png',bbox_inches='tight')

#plm_resp_corr = ret['plm_resp_corrected'][:,0]
#cross = hp.alm2cl(klm, plm_resp_corr, lmax=lmax) * (l*(l+1))/2
## Plot
#plt.figure(0)
#plt.clf()
#plt.plot(l, cross, color='lightcoral', linestyle='-', alpha=0.5, label='Cross $C_L^{\kappa\kappa}$ with Input')
#plt.plot(l, input_clkk, 'k', label='Input WebSky $C_L^{\kappa\kappa}$')
#plt.grid(True, linestyle="--", alpha=0.5)
#plt.ylabel("$C_L^{\kappa\kappa}$")
#plt.xlabel('$L$')
#plt.title(f'WebSky Reconstruction x Input, lmaxT = 3500',pad=10)
#plt.legend(loc='upper right', fontsize='small')
#plt.xscale('log')
#plt.yscale('log')
#plt.xlim(10,lmax)
#plt.ylim(1e-9,1e-6)
#plt.tight_layout()
#plt.savefig(dir_out+f'/figs/test.png',bbox_inches='tight')

