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
    '''
    config = utils.parse_yaml(config_file)
    lmax = config['lensrec']['Lmax']
    lmin = config['lensrec']['lminT']
    lmaxT = config['lensrec']['lmaxT']
    lmaxP = config['lensrec']['lmaxP']
    nside = config['lensrec']['nside']
    dir_out = config['dir_out']
    l = np.arange(0,lmax+1)

    if cinv:
        # Standard SQE
        # Full sky, no masking, MV ILC weighting
        # Sims are signal + foreground sims from Yuuki + frequency correlated noise realizations generated from frequency separated noise spectra
        filename = dir_out+f'/n0/n0_98simpairs_healqest_sqe_lmaxT3000_lmaxP4096_nside2048_standard_resp_from_sims.pkl'
        n0_sqe = pickle.load(open(filename,'rb'))
        n0_sqe_total = n0_sqe['total'] * (l*(l+1))**2/4
        n0_sqe_TT = n0_sqe['TT'] * (l*(l+1))**2/4
        # lmaxT = 3500
        filename = dir_out+f'/n0/n0_98simpairs_healqest_sqe_lmaxT3500_lmaxP4096_nside2048_standard_resp_from_sims.pkl'
        n0_sqe_3500 = pickle.load(open(filename,'rb'))
        n0_sqe_3500_total = n0_sqe_3500['total'] * (l*(l+1))**2/4
        n0_sqe_3500_TT = n0_sqe_3500['TT'] * (l*(l+1))**2/4
        # lmaxT = 4000
        filename = dir_out+f'/n0/n0_98simpairs_healqest_sqe_lmaxT4000_lmaxP4096_nside2048_standard_resp_from_sims.pkl'
        n0_sqe_4000 = pickle.load(open(filename,'rb'))
        n0_sqe_4000_total = n0_sqe_4000['total'] * (l*(l+1))**2/4
        n0_sqe_4000_TT = n0_sqe_4000['TT'] * (l*(l+1))**2/4

        # Standard GMV
        # Full sky, no masking, MV ILC weighting
        # Sims are signal + foreground sims from Yuuki + frequency correlated noise realizations generated from frequency separated noise spectra
        filename = dir_out+f'/n0/n0_98simpairs_healqest_gmv_cinv_lmaxT3000_lmaxP4096_nside2048_standard_resp_from_sims.pkl'
        n0_standard = pickle.load(open(filename,'rb'))
        n0_standard_total = n0_standard['total'] * (l*(l+1))**2/4
        n0_standard_TTEETE = n0_standard['TTEETE'] * (l*(l+1))**2/4
        n0_standard_TBEB = n0_standard['TBEB'] * (l*(l+1))**2/4
        # Version using Eq. 45-49
        filename = dir_out+f'/n0/n0_98simpairs_healqest_gmv_lmaxT3000_lmaxP4096_nside2048_standard_resp_from_sims.pkl'
        n0_standard_eq45to49 = pickle.load(open(filename,'rb'))
        n0_standard_eq45to49_total = n0_standard['total'] * (l*(l+1))**2/4
        n0_standard_eq45to49_TTEETE = n0_standard['TTEETE'] * (l*(l+1))**2/4
        n0_standard_eq45to49_TBEB = n0_standard['TBEB'] * (l*(l+1))**2/4
        # lmaxT = 3500
        # lmaxT = 4000

        # MH GMV
        # Full sky, no masking, one leg is MV ILC weighted and the other is tSZ-nulled ILC weighted
        # Sims are signal + foreground sims from Yuuki + frequency correlated noise realizations generated from frequency separated noise spectra
        filename = dir_out+f'/n0/n0_98simpairs_healqest_gmv_cinv_withT3_lmaxT3000_lmaxP4096_nside2048_mh_resp_from_sims_9ests.pkl'
        n0_mh = pickle.load(open(filename,'rb'))
        n0_mh_total = n0_mh['total'] * (l*(l+1))**2/4
        n0_mh_TTEETE = n0_mh['TTEETE'] * (l*(l+1))**2/4
        n0_mh_TBEB = n0_mh['TBEB'] * (l*(l+1))**2/4
        # Version using Eq. 45-49
        filename = dir_out+f'/n0/n0_98simpairs_healqest_gmv_lmaxT3000_lmaxP4096_nside2048_mh_resp_from_sims.pkl'
        n0_mh_eq45to49 = pickle.load(open(filename,'rb'))
        n0_mh_eq45to49_total = n0_mh['total'] * (l*(l+1))**2/4
        n0_mh_eq45to49_TTEETE = n0_mh['TTEETE'] * (l*(l+1))**2/4
        n0_mh_eq45to49_TBEB = n0_mh['TBEB'] * (l*(l+1))**2/4
        # SQE
        filename = dir_out+f'/n0/n0_98simpairs_healqest_sqe_lmaxT3000_lmaxP4096_nside2048_mh_resp_from_sims.pkl'
        n0_mh_sqe = pickle.load(open(filename,'rb'))
        n0_mh_sqe_TT = n0_mh_sqe['TT'] * (l*(l+1))**2/4
        # lmaxT = 3500
        # SQE
        filename = dir_out+f'/n0/n0_98simpairs_healqest_sqe_lmaxT3500_lmaxP4096_nside2048_mh_resp_from_sims.pkl'
        n0_mh_sqe_3500 = pickle.load(open(filename,'rb'))
        n0_mh_sqe_3500_TT = n0_mh_sqe_3500['TT'] * (l*(l+1))**2/4
        # lmaxT = 4000
        # SQE
        filename = dir_out+f'/n0/n0_98simpairs_healqest_sqe_lmaxT4000_lmaxP4096_nside2048_mh_resp_from_sims.pkl'
        n0_mh_sqe_4000 = pickle.load(open(filename,'rb'))
        n0_mh_sqe_4000_TT = n0_mh_sqe_4000['TT'] * (l*(l+1))**2/4

        # Cross-ILC GMV
        # Full sky, no masking, one leg is CIB-nulled ILC weighted and the other is tSZ-nulled ILC weighted
        # Sims are signal + foreground sims from Yuuki + frequency correlated noise realizations generated from frequency separated noise spectra
        # One component
        # SQE
        filename = dir_out+f'/n0/n0_98simpairs_healqest_sqe_lmaxT3000_lmaxP4096_nside2048_crossilc_onesed_resp_from_sims.pkl'
        n0_crossilc_onesed_sqe = pickle.load(open(filename,'rb'))
        n0_crossilc_onesed_sqe_TT = n0_crossilc_onesed_sqe['TT'] * (l*(l+1))**2/4

        # Two component
        filename = dir_out+f'/n0/n0_98simpairs_healqest_gmv_cinv_withT3_lmaxT3000_lmaxP4096_nside2048_crossilc_twoseds_resp_from_sims_9ests.pkl'
        n0_crossilc_twoseds = pickle.load(open(filename,'rb'))
        n0_crossilc_twoseds_total = n0_crossilc_twoseds['total'] * (l*(l+1))**2/4
        n0_crossilc_twoseds_TTEETE = n0_crossilc_twoseds['TTEETE'] * (l*(l+1))**2/4
        n0_crossilc_twoseds_TBEB = n0_crossilc_twoseds['TBEB'] * (l*(l+1))**2/4
        # Version using Eq. 45-49
        filename = dir_out+f'/n0/n0_98simpairs_healqest_gmv_lmaxT3000_lmaxP4096_nside2048_crossilc_twoseds_resp_from_sims.pkl'
        n0_crossilc_twoseds_eq45to49 = pickle.load(open(filename,'rb'))
        n0_crossilc_twoseds_eq45to49_total = n0_crossilc_twoseds['total'] * (l*(l+1))**2/4
        n0_crossilc_twoseds_eq45to49_TTEETE = n0_crossilc_twoseds['TTEETE'] * (l*(l+1))**2/4
        n0_crossilc_twoseds_eq45to49_TBEB = n0_crossilc_twoseds['TBEB'] * (l*(l+1))**2/4
        # SQE
        filename = dir_out+f'/n0/n0_98simpairs_healqest_sqe_lmaxT3000_lmaxP4096_nside2048_crossilc_twoseds_resp_from_sims.pkl'
        n0_crossilc_twoseds_sqe = pickle.load(open(filename,'rb'))
        n0_crossilc_twoseds_sqe_TT = n0_crossilc_twoseds_sqe['TT'] * (l*(l+1))**2/4
        # lmaxT = 3500
        # SQE
        filename = dir_out+f'/n0/n0_98simpairs_healqest_sqe_lmaxT3500_lmaxP4096_nside2048_crossilc_twoseds_resp_from_sims.pkl'
        n0_crossilc_twoseds_sqe_3500 = pickle.load(open(filename,'rb'))
        n0_crossilc_twoseds_sqe_3500_TT = n0_crossilc_twoseds_sqe_3500['TT'] * (l*(l+1))**2/4
        # lmaxT = 4000
        # SQE
        filename = dir_out+f'/n0/n0_98simpairs_healqest_sqe_lmaxT4000_lmaxP4096_nside2048_crossilc_twoseds_resp_from_sims.pkl'
        n0_crossilc_twoseds_sqe_4000 = pickle.load(open(filename,'rb'))
        n0_crossilc_twoseds_sqe_4000_TT = n0_crossilc_twoseds_sqe_4000['TT'] * (l*(l+1))**2/4

        # Profile hardened GMV
        # Full sky, no masking, MV ILC weighting
        # Sims are signal + frequency correlated tSZ foregrounds from tSZ curves + frequency correlated noise realizations generated from frequency separated noise spectra

        # Theory spectrum
        clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
        ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
        clkk = slpp * (l*(l+1))**2/4

        # Plot
        plt.clf()
        plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')

        plt.plot(l, n0_mh_total, color='forestgreen', alpha=0.8, linestyle='-',label='MH GMV, total')
        #plt.plot(l, n0_crossilc_onesed_total, color='goldenrod', alpha=0.8, linestyle='-',label='Cross-ILC GMV (one component CIB), total')
        #plt.plot(l, n0_crossilc_twoseds_total, color='darkorange', alpha=0.8, linestyle='-',label='Cross-ILC GMV (two component CIB), total')
        plt.plot(l, n0_crossilc_twoseds_total, color='darkorange', alpha=0.8, linestyle='-',label='Cross-ILC GMV, total')
        #plt.plot(l, n0_profhrd_total, color='plum', alpha=0.8, linestyle='-',label='Profile Hardened GMV, total')
        plt.plot(l, n0_sqe_total, color='firebrick', alpha=0.8, linestyle='-',label='Standard SQE, MV')
        plt.plot(l, n0_standard_total, color='darkblue', alpha=0.8, linestyle='-',label='Standard GMV, total')

        plt.ylabel("$[\ell(\ell+1)]^2$$N_0$ / 4 $[\mu K^2]$")
        plt.xlabel('$\ell$')
        plt.title(f'GMV Reconstruction Noise Comparison')
        plt.legend(loc='upper left', fontsize='small')
        plt.xscale('log')
        plt.yscale('log')
        #plt.xlim(10,lmax)
        plt.xlim(10,2000)
        #plt.ylim(1e-8,1e-6)
        plt.ylim(1e-8,2e-7)
        plt.savefig(dir_out+f'/figs/n0_comparison_gmv_cinv.png',bbox_inches='tight')

        plt.clf()
        #ratio_n0_crossilc_onesed_total = n0_crossilc_onesed_total/n0_standard_total
        ratio_n0_crossilc_twoseds_total = n0_crossilc_twoseds_total/n0_standard_total
        ratio_n0_mh_total = n0_mh_total/n0_standard_total
        #ratio_n0_profhrd_total = n0_profhrd_total/n0_standard_total
        ratio_n0_sqe_total = n0_sqe_total/n0_standard_total
        # Ratios with error bars
        plt.axhline(y=1, color='k', linestyle='--')
        plt.plot(l, ratio_n0_mh_total, color='forestgreen', alpha=0.8, linestyle='-',label='Ratio MH GMV / Standard GMV')
        #plt.plot(l, ratio_n0_crossilc_onesed_total,color='goldenrod', alpha=0.8, linestyle='-',label='Ratio Cross-ILC GMV (one component CIB) / Standard GMV')
        #plt.plot(l, ratio_n0_crossilc_twoseds_total, color='darkorange', alpha=0.8, linestyle='-',label='Ratio Cross-ILC GMV (two component CIB) / Standard GMV')
        plt.plot(l, ratio_n0_crossilc_twoseds_total, color='darkorange', alpha=0.8, linestyle='-',label='Ratio Cross-ILC GMV / Standard GMV')
        #plt.plot(l, ratio_n0_profhrd_total, color='plum', alpha=0.8, linestyle='-',label='Ratio Profile Hardened GMV / Standard GMV')
        plt.plot(l, ratio_n0_sqe_total, color='firebrick', alpha=0.8, linestyle='-',label='Ratio Standard SQE / Standard GMV')
        plt.xlabel('$\ell$')
        plt.title(f'GMV Reconstruction Noise Comparison')
        plt.legend(loc='upper left', fontsize='small')
        plt.xscale('log')
        plt.ylim(0.9,1.4)
        plt.xlim(10,lmax)
        plt.savefig(dir_out+f'/figs/n0_comparison_gmv_ratio_cinv.png',bbox_inches='tight')

        #plt.clf()
        #plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')

        #plt.plot(l, n0_mh_total, color='darkolivegreen', alpha=0.8, linestyle='-',label='MH GMV, lmaxT = 3000')
        #plt.plot(l, n0_mh_3500_total, color='mediumseagreen', alpha=0.8, linestyle='-',label='MH GMV, lmaxT = 3500')
        #plt.plot(l, n0_mh_4000_total, color='lightgreen', alpha=0.8, linestyle='-',label='MH GMV, lmaxT = 4000')
        #plt.plot(l, n0_crossilc_twoseds_total, color='darkorange', alpha=0.8, linestyle='-',label='Cross-ILC GMV, lmaxT = 3000')
        #plt.plot(l, n0_crossilc_twoseds_3500_total, color='burlywood', alpha=0.8, linestyle='-',label='Cross-ILC GMV, lmaxT = 3500')
        #plt.plot(l, n0_crossilc_twoseds_4000_total, color='bisque', alpha=0.8, linestyle='-',label='Cross-ILC GMV, lmaxT = 4000')
        #plt.plot(l, n0_sqe_total, color='firebrick', alpha=0.8, linestyle='-',label='Standard SQE, lmaxT = 3000')
        #plt.plot(l, n0_standard_total, color='darkblue', alpha=0.8, linestyle='-',label='Standard GMV, lmaxT = 3000')
        #plt.plot(l, n0_standard_3500_total, color='cornflowerblue', alpha=0.8, linestyle='-',label='Standard GMV, lmaxT = 3500')
        #plt.plot(l, n0_standard_4000_total, color='lightsteelblue', alpha=0.8, linestyle='',label='Standard GMV, lmaxT = 4000')

        #plt.ylabel("$[\ell(\ell+1)]^2$$N_0$ / 4 $[\mu K^2]$")
        #plt.xlabel('$\ell$')
        #plt.title(f'GMV Reconstruction Noise Comparison')
        #plt.legend(loc='upper left', fontsize='small')
        #plt.xscale('log')
        #plt.yscale('log')
        #plt.xlim(10,2000)
        #plt.ylim(1e-8,2e-7)
        #plt.savefig(dir_out+f'/figs/n0_comparison_different_lmax_cinv.png',bbox_inches='tight')

        #plt.clf()
        #ratio_n0_crossilc_twoseds_total = n0_crossilc_twoseds_total/n0_standard_total
        #ratio_n0_mh_total = n0_mh_total/n0_standard_total
        #ratio_n0_sqe_total = n0_sqe_total/n0_standard_total
        #ratio_n0_mh_3500_total = n0_mh_3500_total/n0_standard_total
        #ratio_n0_mh_4000_total = n0_mh_4000_total/n0_standard_total
        #ratio_n0_crossilc_twoseds_3500_total = n0_crossilc_twoseds_3500_total/n0_standard_total
        #ratio_n0_crossilc_twoseds_4000_total = n0_crossilc_twoseds_4000_total/n0_standard_total

        #plt.axhline(y=1, color='k', linestyle='--')
        #plt.plot(l, ratio_n0_mh_total, color='darkolivegreen', alpha=0.8, linestyle='-',label='Ratio MH GMV / Standard GMV, lmaxT = 3000')
        #plt.plot(l, ratio_n0_mh_3500_total, color='mediumseagreen', alpha=0.8, linestyle='-',label='Ratio MH GMV (lmaxT = 3500) / Standard GMV (lmaxT = 3000)')
        #plt.plot(l, ratio_n0_mh_4000_total, color='lightgreen', alpha=0.8, linestyle='-',label='Ratio MH GMV (lmaxT = 4000) / Standard GMV (lmaxT = 3000)')
        #plt.plot(l, ratio_n0_crossilc_twoseds_total, color='darkorange', alpha=0.8, linestyle='-',label='Ratio Cross-ILC GMV / Standard GMV, lmaxT = 3000')
        #plt.plot(l, ratio_n0_crossilc_twoseds_3500_total, color='burlywood', alpha=0.8, linestyle='-',label='Ratio Cross-ILC GMV (lmaxT = 3500) / Standard GMV (lmaxT = 3000)')
        #plt.plot(l, ratio_n0_crossilc_twoseds_4000_total, color='bisque', alpha=0.8, linestyle='-',label='Ratio Cross-ILC GMV (lmaxT = 4000) / Standard GMV (lmaxT = 3000)')
        #plt.plot(l, ratio_n0_sqe_total, color='firebrick', alpha=0.8, linestyle='-',label='Ratio Standard SQE / Standard GMV, lmaxT = 3000')
        #plt.xlabel('$\ell$')
        #plt.title(f'GMV Reconstruction Noise Comparison')
        #plt.legend(loc='lower left', fontsize='small')
        #plt.xscale('log')
        #plt.ylim(0.5,1.5)
        #plt.xlim(10,lmax)
        #plt.savefig(dir_out+f'/figs/n0_comparison_different_lmax_ratio_cinv.png',bbox_inches='tight')

    else:
        # Standard SQE
        # Full sky, no masking, no ILC weighting
        # No foregrounds, just signal + Gaussian uncorrelated noise generated from 2019/2020 noise curves
        #filename = dir_out+f'/n0/n0_lensing19-20_no_foregrounds_with_ilc_noise/n0_98simpairs_healqest_sqe_lmaxT3000_lmaxP4096_nside2048_cmbonly_resp_from_sims.pkl'
        #n0_sqe = pickle.load(open(filename,'rb'))
        #n0_sqe_total = n0_sqe['total'] * (l*(l+1))**2/4
        # Full sky, no masking, MV ILC weighting
        # Sims are signal + foreground sims from Yuuki + frequency correlated noise realizations generated from frequency separated noise spectra
        filename = dir_out+f'/n0/n0_98simpairs_healqest_sqe_lmaxT3000_lmaxP4096_nside2048_standard_resp_from_sims.pkl'
        n0_sqe = pickle.load(open(filename,'rb'))
        n0_sqe_total = n0_sqe['total'] * (l*(l+1))**2/4
        n0_sqe_TT = n0_sqe['TT'] * (l*(l+1))**2/4
        # lmaxT = 3500
        filename = dir_out+f'/n0/n0_98simpairs_healqest_sqe_lmaxT3500_lmaxP4096_nside2048_standard_resp_from_sims.pkl'
        n0_sqe_3500 = pickle.load(open(filename,'rb'))
        n0_sqe_3500_total = n0_sqe_3500['total'] * (l*(l+1))**2/4
        n0_sqe_3500_TT = n0_sqe_3500['TT'] * (l*(l+1))**2/4
        # lmaxT = 4000
        filename = dir_out+f'/n0/n0_98simpairs_healqest_sqe_lmaxT4000_lmaxP4096_nside2048_standard_resp_from_sims.pkl'
        n0_sqe_4000 = pickle.load(open(filename,'rb'))
        n0_sqe_4000_total = n0_sqe_4000['total'] * (l*(l+1))**2/4
        n0_sqe_4000_TT = n0_sqe_4000['TT'] * (l*(l+1))**2/4

        # Standard GMV
        # Full sky, no masking, no ILC weighting
        # No foregrounds, just signal + Gaussian uncorrelated noise generated from 2019/2020 noise curves
        #filename = dir_out+f'/n0/n0_lensing19-20_no_foregrounds_with_ilc_noise/n0_98simpairs_healqest_gmv_lmaxT3000_lmaxP4096_nside2048_cmbonly_resp_from_sims.pkl'
        #n0_standard = pickle.load(open(filename,'rb'))
        #n0_standard_total = n0_standard['total'] * (l*(l+1))**2/4
        #n0_standard_TTEETE = n0_standard['TTEETE'] * (l*(l+1))**2/4
        #n0_standard_TBEB = n0_standard['TBEB'] * (l*(l+1))**2/4
        # Full sky, no masking, MV ILC weighting
        # Sims are signal + foreground sims from Yuuki + frequency correlated noise realizations generated from frequency separated noise spectra
        filename = dir_out+f'/n0/n0_98simpairs_healqest_gmv_lmaxT3000_lmaxP4096_nside2048_standard_resp_from_sims.pkl'
        n0_standard = pickle.load(open(filename,'rb'))
        n0_standard_total = n0_standard['total'] * (l*(l+1))**2/4
        n0_standard_TTEETE = n0_standard['TTEETE'] * (l*(l+1))**2/4
        n0_standard_TBEB = n0_standard['TBEB'] * (l*(l+1))**2/4
        # lmaxT = 3500
        filename = dir_out+f'/n0/n0_98simpairs_healqest_gmv_lmaxT3500_lmaxP4096_nside2048_standard_resp_from_sims.pkl'
        n0_standard_3500 = pickle.load(open(filename,'rb'))
        n0_standard_3500_total = n0_standard_3500['total'] * (l*(l+1))**2/4
        n0_standard_3500_TTEETE = n0_standard_3500['TTEETE'] * (l*(l+1))**2/4
        n0_standard_3500_TBEB = n0_standard_3500['TBEB'] * (l*(l+1))**2/4
        # lmaxT = 4000
        filename = dir_out+f'/n0/n0_98simpairs_healqest_gmv_lmaxT4000_lmaxP4096_nside2048_standard_resp_from_sims.pkl'
        n0_standard_4000 = pickle.load(open(filename,'rb'))
        n0_standard_4000_total = n0_standard_4000['total'] * (l*(l+1))**2/4
        n0_standard_4000_TTEETE = n0_standard_4000['TTEETE'] * (l*(l+1))**2/4
        n0_standard_4000_TBEB = n0_standard_4000['TBEB'] * (l*(l+1))**2/4

        # MH GMV
        # Full sky, no masking, one leg is MV ILC weighted and the other is tSZ-nulled ILC weighted
        # Sims are signal + foreground sims from Yuuki + frequency correlated noise realizations generated from frequency separated noise spectra
        filename = dir_out+f'/n0/n0_98simpairs_healqest_gmv_lmaxT3000_lmaxP4096_nside2048_mh_resp_from_sims.pkl'
        n0_mh = pickle.load(open(filename,'rb'))
        n0_mh_total = n0_mh['total'] * (l*(l+1))**2/4
        n0_mh_TTEETE = n0_mh['TTEETE'] * (l*(l+1))**2/4
        n0_mh_TBEB = n0_mh['TBEB'] * (l*(l+1))**2/4
        # SQE
        filename = dir_out+f'/n0/n0_98simpairs_healqest_sqe_lmaxT3000_lmaxP4096_nside2048_mh_resp_from_sims.pkl'
        n0_mh_sqe = pickle.load(open(filename,'rb'))
        n0_mh_sqe_TT = n0_mh_sqe['TT'] * (l*(l+1))**2/4
        # lmaxT = 3500
        filename = dir_out+f'/n0/n0_98simpairs_healqest_gmv_lmaxT3500_lmaxP4096_nside2048_mh_resp_from_sims.pkl'
        n0_mh_3500 = pickle.load(open(filename,'rb'))
        n0_mh_3500_total = n0_mh_3500['total'] * (l*(l+1))**2/4
        n0_mh_3500_TTEETE = n0_mh_3500['TTEETE'] * (l*(l+1))**2/4
        n0_mh_3500_TBEB = n0_mh_3500['TBEB'] * (l*(l+1))**2/4
        # SQE
        filename = dir_out+f'/n0/n0_98simpairs_healqest_sqe_lmaxT3500_lmaxP4096_nside2048_mh_resp_from_sims.pkl'
        n0_mh_sqe_3500 = pickle.load(open(filename,'rb'))
        n0_mh_sqe_3500_TT = n0_mh_sqe_3500['TT'] * (l*(l+1))**2/4
        # lmaxT = 4000
        filename = dir_out+f'/n0/n0_98simpairs_healqest_gmv_lmaxT4000_lmaxP4096_nside2048_mh_resp_from_sims.pkl'
        n0_mh_4000 = pickle.load(open(filename,'rb'))
        n0_mh_4000_total = n0_mh_4000['total'] * (l*(l+1))**2/4
        n0_mh_4000_TTEETE = n0_mh_4000['TTEETE'] * (l*(l+1))**2/4
        n0_mh_4000_TBEB = n0_mh_4000['TBEB'] * (l*(l+1))**2/4
        # SQE
        filename = dir_out+f'/n0/n0_98simpairs_healqest_sqe_lmaxT4000_lmaxP4096_nside2048_mh_resp_from_sims.pkl'
        n0_mh_sqe_4000 = pickle.load(open(filename,'rb'))
        n0_mh_sqe_4000_TT = n0_mh_sqe_4000['TT'] * (l*(l+1))**2/4

        # Cross-ILC GMV
        # Full sky, no masking, one leg is CIB-nulled ILC weighted and the other is tSZ-nulled ILC weighted
        # Sims are signal + foreground sims from Yuuki + frequency correlated noise realizations generated from frequency separated noise spectra
        # One component
        filename = dir_out+f'/n0/n0_98simpairs_healqest_gmv_lmaxT3000_lmaxP4096_nside2048_crossilc_onesed_resp_from_sims.pkl'
        n0_crossilc_onesed = pickle.load(open(filename,'rb'))
        n0_crossilc_onesed_total = n0_crossilc_onesed['total'] * (l*(l+1))**2/4
        n0_crossilc_onesed_TTEETE = n0_crossilc_onesed['TTEETE'] * (l*(l+1))**2/4
        n0_crossilc_onesed_TBEB = n0_crossilc_onesed['TBEB'] * (l*(l+1))**2/4
        # SQE
        filename = dir_out+f'/n0/n0_98simpairs_healqest_sqe_lmaxT3000_lmaxP4096_nside2048_crossilc_onesed_resp_from_sims.pkl'
        n0_crossilc_onesed_sqe = pickle.load(open(filename,'rb'))
        n0_crossilc_onesed_sqe_TT = n0_crossilc_onesed_sqe['TT'] * (l*(l+1))**2/4
        # Two component
        filename = dir_out+f'/n0/n0_98simpairs_healqest_gmv_lmaxT3000_lmaxP4096_nside2048_crossilc_twoseds_resp_from_sims.pkl'
        n0_crossilc_twoseds = pickle.load(open(filename,'rb'))
        n0_crossilc_twoseds_total = n0_crossilc_twoseds['total'] * (l*(l+1))**2/4
        n0_crossilc_twoseds_TTEETE = n0_crossilc_twoseds['TTEETE'] * (l*(l+1))**2/4
        n0_crossilc_twoseds_TBEB = n0_crossilc_twoseds['TBEB'] * (l*(l+1))**2/4
        # SQE
        filename = dir_out+f'/n0/n0_98simpairs_healqest_sqe_lmaxT3000_lmaxP4096_nside2048_crossilc_twoseds_resp_from_sims.pkl'
        n0_crossilc_twoseds_sqe = pickle.load(open(filename,'rb'))
        n0_crossilc_twoseds_sqe_TT = n0_crossilc_twoseds_sqe['TT'] * (l*(l+1))**2/4
        # lmaxT = 3500
        filename = dir_out+f'/n0/n0_98simpairs_healqest_gmv_lmaxT3500_lmaxP4096_nside2048_crossilc_twoseds_resp_from_sims.pkl'
        n0_crossilc_twoseds_3500 = pickle.load(open(filename,'rb'))
        n0_crossilc_twoseds_3500_total = n0_crossilc_twoseds_3500['total'] * (l*(l+1))**2/4
        n0_crossilc_twoseds_3500_TTEETE = n0_crossilc_twoseds_3500['TTEETE'] * (l*(l+1))**2/4
        n0_crossilc_twoseds_3500_TBEB = n0_crossilc_twoseds_3500['TBEB'] * (l*(l+1))**2/4
        # SQE
        filename = dir_out+f'/n0/n0_98simpairs_healqest_sqe_lmaxT3500_lmaxP4096_nside2048_crossilc_twoseds_resp_from_sims.pkl'
        n0_crossilc_twoseds_sqe_3500 = pickle.load(open(filename,'rb'))
        n0_crossilc_twoseds_sqe_3500_TT = n0_crossilc_twoseds_sqe_3500['TT'] * (l*(l+1))**2/4
        # lmaxT = 4000
        filename = dir_out+f'/n0/n0_98simpairs_healqest_gmv_lmaxT4000_lmaxP4096_nside2048_crossilc_twoseds_resp_from_sims.pkl'
        n0_crossilc_twoseds_4000 = pickle.load(open(filename,'rb'))
        n0_crossilc_twoseds_4000_total = n0_crossilc_twoseds_4000['total'] * (l*(l+1))**2/4
        n0_crossilc_twoseds_4000_TTEETE = n0_crossilc_twoseds_4000['TTEETE'] * (l*(l+1))**2/4
        n0_crossilc_twoseds_4000_TBEB = n0_crossilc_twoseds_4000['TBEB'] * (l*(l+1))**2/4
        # SQE
        filename = dir_out+f'/n0/n0_98simpairs_healqest_sqe_lmaxT4000_lmaxP4096_nside2048_crossilc_twoseds_resp_from_sims.pkl'
        n0_crossilc_twoseds_sqe_4000 = pickle.load(open(filename,'rb'))
        n0_crossilc_twoseds_sqe_4000_TT = n0_crossilc_twoseds_sqe_4000['TT'] * (l*(l+1))**2/4

        # Profile hardened GMV
        # Full sky, no masking, MV ILC weighting
        # Sims are signal + frequency correlated tSZ foregrounds from tSZ curves + frequency correlated noise realizations generated from frequency separated noise spectra
        filename = dir_out+f'/n0/n0_98simpairs_healqest_gmv_lmaxT3000_lmaxP4096_nside2048_profhrd_resp_from_sims_resp_method_B.pkl'
        n0_profhrd = pickle.load(open(filename,'rb'))
        n0_profhrd_total = n0_profhrd['total_hrd'] * (l*(l+1))**2/4
        n0_profhrd_TTEETE = n0_profhrd['TTEETE_hrd'] * (l*(l+1))**2/4
        n0_profhrd_TBEB = n0_profhrd['TBEB'] * (l*(l+1))**2/4

        # Theory spectrum
        clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
        ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
        clkk = slpp * (l*(l+1))**2/4

        # Plot
        plt.clf()
        plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')

        plt.plot(l, n0_mh_total, color='forestgreen', alpha=0.8, linestyle='-',label='MH GMV, total')
        #plt.plot(l, n0_crossilc_onesed_total, color='goldenrod', alpha=0.8, linestyle='-',label='Cross-ILC GMV (one component CIB), total')
        #plt.plot(l, n0_crossilc_twoseds_total, color='darkorange', alpha=0.8, linestyle='-',label='Cross-ILC GMV (two component CIB), total')
        plt.plot(l, n0_crossilc_twoseds_total, color='darkorange', alpha=0.8, linestyle='-',label='Cross-ILC GMV, total')
        #plt.plot(l, n0_profhrd_total, color='plum', alpha=0.8, linestyle='-',label='Profile Hardened GMV, total')
        plt.plot(l, n0_sqe_total, color='firebrick', alpha=0.8, linestyle='-',label='Standard SQE, MV')
        plt.plot(l, n0_standard_total, color='darkblue', alpha=0.8, linestyle='-',label='Standard GMV, total')

        plt.ylabel("$[\ell(\ell+1)]^2$$N_0$ / 4 $[\mu K^2]$")
        plt.xlabel('$\ell$')
        plt.title(f'GMV Reconstruction Noise Comparison')
        plt.legend(loc='upper left', fontsize='small')
        plt.xscale('log')
        plt.yscale('log')
        #plt.xlim(10,lmax)
        plt.xlim(10,2000)
        #plt.ylim(1e-8,1e-6)
        plt.ylim(1e-8,2e-7)
        plt.savefig(dir_out+f'/figs/n0_comparison_gmv.png',bbox_inches='tight')

        plt.clf()
        plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')

        plt.plot(l, n0_sqe_TT, color='firebrick', alpha=0.8, linestyle='-',label='Standard SQE TT, MV')
        plt.plot(l, n0_crossilc_onesed_sqe_TT, color='goldenrod', alpha=0.8, linestyle='-',label='Cross-ILC SQE (one component CIB), TT')
        plt.plot(l, n0_crossilc_twoseds_sqe_TT, color='darkorange', alpha=0.8, linestyle='-',label='Cross-ILC SQE (two component CIB), TT')
        plt.plot(l, n0_mh_sqe_TT, color='forestgreen', alpha=0.8, linestyle='-',label='MH SQE, TT')

        plt.ylabel("$[\ell(\ell+1)]^2$$N_0$ / 4 $[\mu K^2]$")
        plt.xlabel('$\ell$')
        plt.title(f'SQE TT Reconstruction Noise Comparison')
        plt.legend(loc='upper left', fontsize='small')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(80,3000)
        plt.ylim(1e-9,1e-4)
        plt.savefig(dir_out+f'/figs/n0_comparison_sqe.png',bbox_inches='tight')

        plt.clf()
        ratio_n0_crossilc_onesed_total = n0_crossilc_onesed_total/n0_standard_total
        ratio_n0_crossilc_twoseds_total = n0_crossilc_twoseds_total/n0_standard_total
        ratio_n0_mh_total = n0_mh_total/n0_standard_total
        ratio_n0_profhrd_total = n0_profhrd_total/n0_standard_total
        ratio_n0_sqe_total = n0_sqe_total/n0_standard_total
        # Ratios with error bars
        plt.axhline(y=1, color='k', linestyle='--')
        plt.plot(l, ratio_n0_mh_total, color='forestgreen', alpha=0.8, linestyle='-',label='Ratio MH GMV / Standard GMV')
        #plt.plot(l, ratio_n0_crossilc_onesed_total,color='goldenrod', alpha=0.8, linestyle='-',label='Ratio Cross-ILC GMV (one component CIB) / Standard GMV')
        #plt.plot(l, ratio_n0_crossilc_twoseds_total, color='darkorange', alpha=0.8, linestyle='-',label='Ratio Cross-ILC GMV (two component CIB) / Standard GMV')
        plt.plot(l, ratio_n0_crossilc_twoseds_total, color='darkorange', alpha=0.8, linestyle='-',label='Ratio Cross-ILC GMV / Standard GMV')
        #plt.plot(l, ratio_n0_profhrd_total, color='plum', alpha=0.8, linestyle='-',label='Ratio Profile Hardened GMV / Standard GMV')
        plt.plot(l, ratio_n0_sqe_total, color='firebrick', alpha=0.8, linestyle='-',label='Ratio Standard SQE / Standard GMV')
        plt.xlabel('$\ell$')
        plt.title(f'GMV Reconstruction Noise Comparison')
        #plt.legend(loc='upper left', fontsize='small')
        plt.xscale('log')
        plt.ylim(0.9,1.4)
        plt.xlim(10,lmax)
        plt.savefig(dir_out+f'/figs/n0_comparison_gmv_ratio.png',bbox_inches='tight')

        plt.clf()
        plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')

        plt.plot(l, n0_mh_total, color='darkolivegreen', alpha=0.8, linestyle='-',label='MH GMV, lmaxT = 3000')
        plt.plot(l, n0_mh_3500_total, color='mediumseagreen', alpha=0.8, linestyle='-',label='MH GMV, lmaxT = 3500')
        plt.plot(l, n0_mh_4000_total, color='lightgreen', alpha=0.8, linestyle='-',label='MH GMV, lmaxT = 4000')
        plt.plot(l, n0_crossilc_twoseds_total, color='darkorange', alpha=0.8, linestyle='-',label='Cross-ILC GMV, lmaxT = 3000')
        plt.plot(l, n0_crossilc_twoseds_3500_total, color='burlywood', alpha=0.8, linestyle='-',label='Cross-ILC GMV, lmaxT = 3500')
        plt.plot(l, n0_crossilc_twoseds_4000_total, color='bisque', alpha=0.8, linestyle='-',label='Cross-ILC GMV, lmaxT = 4000')
        plt.plot(l, n0_sqe_total, color='firebrick', alpha=0.8, linestyle='-',label='Standard SQE, lmaxT = 3000')
        plt.plot(l, n0_standard_total, color='darkblue', alpha=0.8, linestyle='-',label='Standard GMV, lmaxT = 3000')
        plt.plot(l, n0_standard_3500_total, color='cornflowerblue', alpha=0.8, linestyle='-',label='Standard GMV, lmaxT = 3500')
        plt.plot(l, n0_standard_4000_total, color='lightsteelblue', alpha=0.8, linestyle='',label='Standard GMV, lmaxT = 4000')

        plt.ylabel("$[\ell(\ell+1)]^2$$N_0$ / 4 $[\mu K^2]$")
        plt.xlabel('$\ell$')
        plt.title(f'GMV Reconstruction Noise Comparison')
        plt.legend(loc='upper left', fontsize='small')
        plt.xscale('log')
        plt.yscale('log')
        #plt.xlim(10,lmax)
        plt.xlim(10,2000)
        #plt.ylim(1e-8,1e-6)
        plt.ylim(1e-8,2e-7)
        plt.savefig(dir_out+f'/figs/n0_comparison_different_lmax.png',bbox_inches='tight')

        plt.clf()
        ratio_n0_crossilc_twoseds_total = n0_crossilc_twoseds_total/n0_standard_total
        ratio_n0_mh_total = n0_mh_total/n0_standard_total
        ratio_n0_sqe_total = n0_sqe_total/n0_standard_total
        ratio_n0_mh_3500_total = n0_mh_3500_total/n0_standard_total
        ratio_n0_mh_4000_total = n0_mh_4000_total/n0_standard_total
        ratio_n0_crossilc_twoseds_3500_total = n0_crossilc_twoseds_3500_total/n0_standard_total
        ratio_n0_crossilc_twoseds_4000_total = n0_crossilc_twoseds_4000_total/n0_standard_total
        # Ratios with error bars
        plt.axhline(y=1, color='k', linestyle='--')
        plt.plot(l, ratio_n0_mh_total, color='darkolivegreen', alpha=0.8, linestyle='-',label='Ratio MH GMV / Standard GMV, lmaxT = 3000')
        plt.plot(l, ratio_n0_mh_3500_total, color='mediumseagreen', alpha=0.8, linestyle='-',label='Ratio MH GMV (lmaxT = 3500) / Standard GMV (lmaxT = 3000)')
        plt.plot(l, ratio_n0_mh_4000_total, color='lightgreen', alpha=0.8, linestyle='-',label='Ratio MH GMV (lmaxT = 4000) / Standard GMV (lmaxT = 3000)')
        plt.plot(l, ratio_n0_crossilc_twoseds_total, color='darkorange', alpha=0.8, linestyle='-',label='Ratio Cross-ILC GMV / Standard GMV, lmaxT = 3000')
        plt.plot(l, ratio_n0_crossilc_twoseds_3500_total, color='burlywood', alpha=0.8, linestyle='-',label='Ratio Cross-ILC GMV (lmaxT = 3500) / Standard GMV (lmaxT = 3000)')
        plt.plot(l, ratio_n0_crossilc_twoseds_4000_total, color='bisque', alpha=0.8, linestyle='-',label='Ratio Cross-ILC GMV (lmaxT = 4000) / Standard GMV (lmaxT = 3000)')
        plt.plot(l, ratio_n0_sqe_total, color='firebrick', alpha=0.8, linestyle='-',label='Ratio Standard SQE / Standard GMV, lmaxT = 3000')
        plt.xlabel('$\ell$')
        plt.title(f'GMV Reconstruction Noise Comparison')
        plt.legend(loc='lower left', fontsize='small')
        plt.xscale('log')
        plt.ylim(0.5,1.5)
        plt.xlim(10,lmax)
        plt.savefig(dir_out+f'/figs/n0_comparison_different_lmax_ratio.png',bbox_inches='tight')

compare_n0()
