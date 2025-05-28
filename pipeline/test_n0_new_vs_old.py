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

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def compare_n0(config_file='test_yuka_lmaxT3500.yaml', cinv=True):
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
    filename = dir_out+f'/n0/n0_249simpairs_healqest_sqe_lmaxT3000_lmaxP4096_nside2048_agora_standard_resp_from_sims.pkl'
    n0_sqe_3000 = pickle.load(open(filename,'rb'))
    n0_sqe_3000_total = n0_sqe_3000['total'] * (l*(l+1))**2/4
    # lmaxT = 3500
    filename = dir_out+f'/n0/n0_249simpairs_healqest_sqe_lmaxT3500_lmaxP4096_nside2048_agora_standard_resp_from_sims.pkl'
    n0_sqe_3500 = pickle.load(open(filename,'rb'))
    n0_sqe_3500_total = n0_sqe_3500['total'] * (l*(l+1))**2/4

    # Standard GMV
    # lmaxT = 3000
    filename = dir_out+f'/n0/n0_249simpairs_healqest_gmv_cinv_lmaxT3000_lmaxP4096_nside2048_agora_standard_resp_from_sims.pkl'
    n0_standard_3000 = pickle.load(open(filename,'rb'))
    n0_standard_3000_total = n0_standard_3000['total'] * (l*(l+1))**2/4
    # lmaxT = 3500
    filename = dir_out+f'/n0/n0_249simpairs_healqest_gmv_cinv_lmaxT3500_lmaxP4096_nside2048_agora_standard_resp_from_sims.pkl'
    n0_standard_3500 = pickle.load(open(filename,'rb'))
    n0_standard_3500_total = n0_standard_3500['total'] * (l*(l+1))**2/4
    # lmaxT = 4000
    filename = dir_out+f'/n0/n0_249simpairs_healqest_gmv_cinv_lmaxT4000_lmaxP4096_nside2048_agora_standard_resp_from_sims.pkl'
    n0_standard_4000 = pickle.load(open(filename,'rb'))
    n0_standard_4000_total = n0_standard_4000['total'] * (l*(l+1))**2/4
    # lmaxT = 3500, RDN0
    filename = dir_out+f'/n0/rdn0_249simpairs_healqest_gmv_cinv_lmaxT3500_lmaxP4096_nside2048_agora_standard_resp_from_sims.pkl'
    rdn0_standard_3500 = pickle.load(open(filename,'rb'))
    rdn0_standard_3500_total = rdn0_standard_3500 * (l*(l+1))**2/4

    # Standard GMV, WEBSKY
    # lmaxT = 3000
    filename = dir_out+f'/n0/n0_249simpairs_healqest_gmv_cinv_lmaxT3000_lmaxP4096_nside2048_websky_standard_resp_from_sims.pkl'
    n0_standard_3000_websky = pickle.load(open(filename,'rb'))
    n0_standard_3000_websky_total = n0_standard_3000_websky['total'] * (l*(l+1))**2/4
    # lmaxT = 3500
    filename = dir_out+f'/n0/n0_249simpairs_healqest_gmv_cinv_lmaxT3500_lmaxP4096_nside2048_websky_standard_resp_from_sims.pkl'
    n0_standard_3500_websky = pickle.load(open(filename,'rb'))
    n0_standard_3500_websky_total = n0_standard_3500_websky['total'] * (l*(l+1))**2/4
    # lmaxT = 4000
    filename = dir_out+f'/n0/n0_249simpairs_healqest_gmv_cinv_lmaxT4000_lmaxP4096_nside2048_websky_standard_resp_from_sims.pkl'
    n0_standard_4000_websky = pickle.load(open(filename,'rb'))
    n0_standard_4000_websky_total = n0_standard_4000_websky['total'] * (l*(l+1))**2/4

    # Standard GMV, OLD
    # lmaxT = 3500
    filename = '/oak/stanford/orgs/kipac/users/yukanaka/outputs_with_frequency_separated_inputs'+f'/n0/n0_249simpairs_healqest_gmv_cinv_lmaxT3500_lmaxP4096_nside2048_standard_resp_from_sims.pkl'
    n0_standard_3500_old = pickle.load(open(filename,'rb'))
    n0_standard_3500_old_total = n0_standard_3500_old['total'] * (l*(l+1))**2/4
    # lmaxT = 3500, RDN0
    filename = '/oak/stanford/orgs/kipac/users/yukanaka/outputs_with_frequency_separated_inputs'+f'/n0/rdn0_249simpairs_healqest_gmv_cinv_lmaxT3500_lmaxP4096_nside2048_standard_resp_from_sims.pkl'
    rdn0_standard_3500_old = pickle.load(open(filename,'rb'))
    rdn0_standard_3500_old_total = rdn0_standard_3500_old * (l*(l+1))**2/4

    # Standard GMV, lmaxT = 3500, FAKE (using sim 1 as real data)
    filename = dir_out+f'/n0/fake_rdn0_248simpairs_healqest_gmv_cinv_lmaxT3500_lmaxP4096_nside2048_agora_standard_resp_from_sims.pkl'
    fake_rdn0_standard_3500 = pickle.load(open(filename,'rb'))
    fake_rdn0_standard_3500_total = fake_rdn0_standard_3500 * (l*(l+1))**2/4
    # This is sim-based N0, not necessarily "fake" but it omits sim 1
    filename = dir_out+f'/n0/n0_248simpairs_healqest_gmv_cinv_lmaxT3500_lmaxP4096_nside2048_agora_standard_resp_from_sims.pkl'
    fake_n0_standard_3500 = pickle.load(open(filename,'rb'))
    fake_n0_standard_3500_total = fake_n0_standard_3500['total'] * (l*(l+1))**2/4

    # Theory spectrum
    clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
    ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
    clkk = slpp * (l*(l+1))**2/4

    # Plot
    plt.clf()
    plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')
    plt.plot(l, n0_standard_3500_total, color='darkblue', alpha=0.8, linestyle='-',label='N0 Standard GMV, New Inputs')
    plt.plot(l, n0_standard_3500_old_total, color='firebrick', alpha=0.8, linestyle='-',label='N0 Standard GMV, Old Inputs')
    plt.plot(l, rdn0_standard_3500_old_total, color='forestgreen', alpha=0.8, linestyle='-',label='RDN0 Standard GMV, Old Inputs')
    plt.ylabel("$[\ell(\ell+1)]^2$$N_0$ / 4 $[\mu K^2]$")
    plt.xlabel('$\ell$')
    plt.title(f'GMV Reconstruction Noise Comparison')
    plt.legend(loc='upper left', fontsize='small')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,lmax)
    plt.ylim(1e-8,1e-7)
    plt.savefig(dir_out+f'/figs/test_n0_comparison_gmv_lmaxT3500_old_vs_new_inputs.png',bbox_inches='tight')

    plt.clf()
    ratio_n0 = n0_standard_3500_old_total/n0_standard_3500_total
    ratio_rdn0 = rdn0_standard_3500_old_total/n0_standard_3500_total
    ratio_old = rdn0_standard_3500_old_total/n0_standard_3500_old_total
    ratio_rdn0_new = rdn0_standard_3500_total/n0_standard_3500_total
    ratio_rdn0_new_vs_old = rdn0_standard_3500_total/rdn0_standard_3500_old_total
    ratio_rdn0_fake = fake_rdn0_standard_3500_total/fake_n0_standard_3500_total
    ratio_n0 = moving_average(ratio_n0, window_size=50)
    ratio_rdn0 = moving_average(ratio_rdn0, window_size=50)
    ratio_old = moving_average(ratio_old, window_size=50)
    ratio_rdn0_new = moving_average(ratio_rdn0_new, window_size=50)
    ratio_rdn0_new_vs_old = moving_average(ratio_rdn0_new_vs_old, window_size=50)
    ratio_rdn0_fake = moving_average(ratio_rdn0_fake, window_size=50)
    # Ratios with error bars
    plt.axhline(y=1, color='k', linestyle='--')
    #plt.plot(l, ratio_n0, color='firebrick', alpha=0.8, linestyle='-',label='Ratio Old N0 / New N0')
    #plt.plot(l, ratio_rdn0, color='forestgreen', alpha=0.8, linestyle='-',label='Ratio Old RDN0 / New N0')
    plt.plot(l, ratio_old, color='magenta', alpha=0.8, linestyle='--',label='Ratio Old RDN0 / Old N0')
    plt.plot(l, ratio_rdn0_new, color='orange', alpha=0.8, linestyle='--',label='Ratio New RDN0 / New N0')
    #plt.plot(l, ratio_rdn0_new_vs_old, color='cornflowerblue', alpha=0.8, linestyle='--',label='Ratio New RDN0 / Old RDN0')
    plt.plot(l, ratio_rdn0_fake, color='cornflowerblue', alpha=0.8, linestyle='--',label='Ratio New Fake RDN0 / New N0')
    plt.xlabel('$\ell$')
    plt.title(f'GMV Reconstruction Noise Comparison')
    plt.legend(loc='upper left', fontsize='small')
    plt.xscale('log')
    plt.ylim(0.95,1.025)
    plt.xlim(10,lmax)
    plt.savefig(dir_out+f'/figs/test_n0_comparison_gmv_lmaxT3500_old_vs_new_inputs_ratio.png',bbox_inches='tight')

    '''
    plt.clf()
    # Check input kappa just in case
    # Input kappa
    klm = hp.read_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_spt3g_input_klm_lmax{lmax}.fits')
    input_clkk = hp.alm2cl(klm,klm,lmax=lmax)
    # Input kappa sims
    mean_input_clkk = 0
    for sim in np.arange(250)+1:
        input_plm = hp.read_alm(f'/oak/stanford/orgs/kipac//users/yukanaka/lensing19-20/inputcmb/phi/phi_lmax_{lmax}/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_phi1_seed{sim}_lmax{lmax}.alm')
        mean_input_clkk += hp.alm2cl(input_plm) * (l*(l+1))**2/4
    mean_input_clkk /= 250
    input_clkk = moving_average(input_clkk, window_size=50)
    mean_input_clkk = moving_average(mean_input_clkk, window_size=50)
    plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')
    plt.plot(l, input_clkk, color='darkblue', alpha=0.8, linestyle='-',label='Input kappa spectrum Agora')
    plt.plot(l, mean_input_clkk, color='firebrick', alpha=0.8, linestyle='-', label='Mean of input kappa sims')
    plt.xlabel('$\ell$')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,lmax)
    plt.ylim(1e-8,1e-7)
    plt.savefig(dir_out+f'/figs/test_input_kappa.png',bbox_inches='tight')

    plt.clf()
    ratio_input_kappa_spec_agora = input_clkk/clkk
    ratio_mean_from_sims = mean_input_clkk/clkk
    # Ratios with error bars
    plt.axhline(y=1, color='k', linestyle='--')
    plt.plot(l, ratio_input_kappa_spec_agora, color='darkblue', alpha=0.8, linestyle='-',label='Ratio Kappa Spec from Agora Map / Fiducial')
    plt.plot(l, ratio_mean_from_sims, color='firebrick', alpha=0.8, linestyle='-',label='Ratio Mean of Input Kappa Sims / Fiducial')
    plt.xlabel('$\ell$')
    plt.legend(loc='upper left', fontsize='small')
    plt.xscale('log')
    #plt.ylim(0.9,1.4)
    plt.xlim(10,lmax)
    plt.savefig(dir_out+f'/figs/test_input_kappa_ratio.png',bbox_inches='tight')
    '''

compare_n0()
