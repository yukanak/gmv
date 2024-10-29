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

def analyze(cinv=False,lbins=np.logspace(np.log10(50),np.log10(3000),20)):
    '''
    Compare with N0/N1 subtraction.
    '''
    config_file='test_yuka_lmaxT4000.yaml'
    apnd = 'standard'
    #apnd = 'crossilc_twoseds'
    config = utils.parse_yaml(config_file)
    lmax = config['lensrec']['Lmax']
    lmin = config['lensrec']['lminT']
    lmaxT = config['lensrec']['lmaxT']
    lmaxP = config['lensrec']['lmaxP']
    nside = config['lensrec']['nside']
    dir_out = config['dir_out']
    l = np.arange(0,lmax+1)
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

    # Get SQE response
    #ests = ['TT', 'EE', 'TE', 'ET', 'TB', 'BT', 'EB', 'BE']
    #resps_original = np.zeros((len(l),len(ests)), dtype=np.complex_)
    #inv_resps_original = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
    #for i, est in enumerate(ests):
    #    resps_original[:,i] = get_sim_response(est,config,gmv=False,append=apnd,sims=np.arange(99)+1,cinv=False)
    #    inv_resps_original[1:,i] = 1/(resps_original)[1:,i]
    #resp_original = np.sum(resps_original, axis=1)
    #inv_resp_original = np.zeros_like(l,dtype=np.complex_); inv_resp_original[1:] = 1/(resp_original)[1:]

    # GMV response
    if cinv:
        resps_gmv = np.zeros((len(l),len(ests)), dtype=np.complex_)
        inv_resps_gmv = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
        for i, est in enumerate(ests):
            resps_gmv[:,i] = get_sim_response(est,config,gmv=True,cinv=True,append=apnd,sims=np.arange(99)+1)
            inv_resps_gmv[1:,i] = 1/(resps_gmv)[1:,i]
        resp_gmv = np.sum(resps_gmv, axis=1)
        resp_gmv_TTEETE = np.sum(resps_gmv[:,:4], axis=1)
        resp_gmv_TBEB = np.sum(resps_gmv[:,4:], axis=1)
    else:
        resp_gmv = get_sim_response('all',config,gmv=True,append=apnd,sims=np.arange(99)+1,cinv=False)
        resp_gmv_TTEETE = get_sim_response('TTEETE',config,gmv=True,append=apnd,sims=np.arange(99)+1,cinv=False)
        resp_gmv_TBEB = get_sim_response('TBEB',config,gmv=True,append=apnd,sims=np.arange(99)+1,cinv=False)
    inv_resp_gmv = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv[1:] = 1./(resp_gmv)[1:]
    inv_resp_gmv_TTEETE = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TTEETE[1:] = 1./(resp_gmv_TTEETE)[1:]
    inv_resp_gmv_TBEB = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TBEB[1:] = 1./(resp_gmv_TBEB)[1:]

    # Get N0 correction (remember these still need (l*(l+1))**2/4 factor to convert to kappa)
    if cinv:
        n0_gmv = get_n0(sims=np.arange(98)+1,qetype='gmv_cinv',config=config,append=apnd)
    else:
        n0_gmv = get_n0(sims=np.arange(98)+1,qetype='gmv',config=config,append=apnd)
    n0_gmv_total = n0_gmv['total'] * (l*(l+1))**2/4
    n0_gmv_TTEETE = n0_gmv['TTEETE'] * (l*(l+1))**2/4
    n0_gmv_TBEB = n0_gmv['TBEB'] * (l*(l+1))**2/4
    #n0_original = get_n0(sims=np.arange(98)+1,qetype='sqe',config=config,
    #                     append=apnd)
    #n0_original_total = n0_original['total'] * (l*(l+1))**2/4
    #n0_original_TT = n0_original['TT'] * (l*(l+1))**2/4
    #n0_original_EE = n0_original['EE'] * (l*(l+1))**2/4
    #n0_original_TE = n0_original['TE'] * (l*(l+1))**2/4
    #n0_original_ET = n0_original['ET'] * (l*(l+1))**2/4
    #n0_original_TB = n0_original['TB'] * (l*(l+1))**2/4
    #n0_original_BT = n0_original['BT'] * (l*(l+1))**2/4
    #n0_original_EB = n0_original['EB'] * (l*(l+1))**2/4
    #n0_original_BE = n0_original['BE'] * (l*(l+1))**2/4

    if cinv:
        n1_gmv = get_n1(sims=np.arange(98)+1,qetype='gmv_cinv',config=config,append=apnd)
    else:
        n1_gmv = get_n1(sims=np.arange(98)+1,qetype='gmv',config=config,append=apnd)
    n1_gmv_total = n1_gmv['total'] * (l*(l+1))**2/4
    #n1_gmv_TTEETE = n1_gmv['TTEETE'] * (l*(l+1))**2/4
    #n1_original = get_n1(sims=np.arange(98)+1,qetype='sqe',config=config,
    #                     append=apnd)
    #n1_original_total = n1_original['total'] * (l*(l+1))**2/4
    #n1_original_TT = n1_original['TT'] * (l*(l+1))**2/4
    #n1_original_EE = n1_original['EE'] * (l*(l+1))**2/4
    #n1_original_TE = n1_original['TE'] * (l*(l+1))**2/4
    #n1_original_ET = n1_original['ET'] * (l*(l+1))**2/4
    #n1_original_TB = n1_original['TB'] * (l*(l+1))**2/4
    #n1_original_BT = n1_original['BT'] * (l*(l+1))**2/4
    #n1_original_EB = n1_original['EB'] * (l*(l+1))**2/4
    #n1_original_BT = n1_original['BE'] * (l*(l+1))**2/4

    #append_list = ['agora_standard', 'agora_standard_rotatedcmb', 'agora_standard_gaussianfg']#, 'agora_standard_rotatedcmb_gaussianfg', 'agora_standard_rotatedgaussiancmb', 'agora_standard_gaussiancmb', 'agora_standard_separated']
    append_list = ['agora_standard', 'agora_standard_rotatedcmb', 'agora_standard_gaussianfg']
    #append_list = ['agora_crossilc_twoseds', 'agora_crossilc_twoseds_rotatedcmb', 'agora_crossilc_twoseds_gaussianfg']
    binned_gmv_clkk = np.zeros((len(bin_centers),len(append_list)), dtype=np.complex_)
    binned_sqe_clkk = np.zeros((len(bin_centers),len(append_list)), dtype=np.complex_)
    cross_gmv = np.zeros((len(bin_centers),len(append_list)), dtype=np.complex_)
    for ii, append in enumerate(append_list):
        if cinv:
            pass
        else:
            # Load GMV plms
            plm_gmv = np.load(dir_out+f'/plm_all_healqest_gmv_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            plm_gmv_TTEETE = np.load(dir_out+f'/plm_TTEETE_healqest_gmv_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            plm_gmv_TBEB = np.load(dir_out+f'/plm_TBEB_healqest_gmv_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')

        # Load SQE plms
        #plms_original = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')),len(ests)), dtype=np.complex_)
        #for i, est in enumerate(ests):
        #    plms_original[:,i] = np.load(dir_out+f'/plm_{est}_healqest_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
        #plm_original = np.sum(plms_original, axis=1)

        # Response correct
        plm_gmv_resp_corr = hp.almxfl(plm_gmv,inv_resp_gmv)
        plm_gmv_resp_corr_TTEETE = hp.almxfl(plm_gmv_TTEETE,inv_resp_gmv_TTEETE)
        plm_gmv_resp_corr_TBEB = hp.almxfl(plm_gmv_TBEB,inv_resp_gmv_TBEB)
        #plm_original_resp_corr = hp.almxfl(plm_original,inv_resp_original)
        #plm_original_resp_corr_TT = hp.almxfl(plms_original[:,0],inv_resps_original[:,0])
        #plm_original_resp_corr_EE = hp.almxfl(plms_original[:,1],inv_resps_original[:,1])
        #plm_original_resp_corr_TE = hp.almxfl(plms_original[:,2],inv_resps_original[:,2])
        #plm_original_resp_corr_ET = hp.almxfl(plms_original[:,3],inv_resps_original[:,3])
        #plm_original_resp_corr_TB = hp.almxfl(plms_original[:,4],inv_resps_original[:,4])
        #plm_original_resp_corr_BT = hp.almxfl(plms_original[:,5],inv_resps_original[:,5])
        #plm_original_resp_corr_EB = hp.almxfl(plms_original[:,6],inv_resps_original[:,6])
        #plm_original_resp_corr_BE = hp.almxfl(plms_original[:,7],inv_resps_original[:,7])

        # Get spectra
        auto_gmv = hp.alm2cl(plm_gmv_resp_corr, plm_gmv_resp_corr, lmax=lmax) * (l*(l+1))**2/4
        auto_gmv_TTEETE = hp.alm2cl(plm_gmv_resp_corr_TTEETE, plm_gmv_resp_corr_TTEETE, lmax=lmax) * (l*(l+1))**2/4
        auto_gmv_TBEB = hp.alm2cl(plm_gmv_resp_corr_TBEB, plm_gmv_resp_corr_TBEB, lmax=lmax) * (l*(l+1))**2/4
        #auto_original = hp.alm2cl(plm_original_resp_corr, plm_original_resp_corr, lmax=lmax) * (l*(l+1))**2/4
        #auto_original_TT = hp.alm2cl(plm_original_resp_corr_TT, plm_original_resp_corr_TT, lmax=lmax) * (l*(l+1))**2/4
        #auto_original_EE = hp.alm2cl(plm_original_resp_corr_EE, plm_original_resp_corr_EE, lmax=lmax) * (l*(l+1))**2/4
        #auto_original_TE = hp.alm2cl(plm_original_resp_corr_TE, plm_original_resp_corr_TE, lmax=lmax) * (l*(l+1))**2/4
        #auto_original_ET = hp.alm2cl(plm_original_resp_corr_ET, plm_original_resp_corr_ET, lmax=lmax) * (l*(l+1))**2/4
        #auto_original_TB = hp.alm2cl(plm_original_resp_corr_TB, plm_original_resp_corr_TB, lmax=lmax) * (l*(l+1))**2/4
        #auto_original_BT = hp.alm2cl(plm_original_resp_corr_BT, plm_original_resp_corr_BT, lmax=lmax) * (l*(l+1))**2/4
        #auto_original_EB = hp.alm2cl(plm_original_resp_corr_EB, plm_original_resp_corr_EB, lmax=lmax) * (l*(l+1))**2/4
        #auto_original_BE = hp.alm2cl(plm_original_resp_corr_BE, plm_original_resp_corr_BE, lmax=lmax) * (l*(l+1))**2/4

        # Response corrected but not N0/N1 subtracted reconstructed phi? N0/N1 shouldn't affect the cross spectrum with input
        # Cross with input
        cross_gmv_unbinned = hp.alm2cl(klm, plm_gmv_resp_corr) * (l*(l+1))/2
        cross_gmv[:,ii] = [cross_gmv_unbinned[digitized == i].mean() for i in range(1, len(lbins))]

        # N0 and N1 subtract
        auto_gmv_debiased = auto_gmv - n0_gmv_total - n1_gmv_total
        #auto_gmv_debiased_TTEETE = auto_gmv_TTEETE - n0_gmv_TTEETE - n1_gmv_TTEETE
        #auto_original_debiased = auto_original - n0_original_total - n1_original_total
        #auto_original_debiased_TT = auto_original_TT - n0_original_TT - n1_original_TT

        # Bin!
        binned_auto_gmv_debiased = [auto_gmv_debiased[digitized == i].mean() for i in range(1, len(lbins))]
        #binned_auto_original_debiased = [auto_original_debiased[digitized == i].mean() for i in range(1, len(lbins))]
        #binned_auto_gmv_debiased_TTEETE = [auto_gmv_debiased_TTEETE[digitized == i].mean() for i in range(1, len(lbins))]
        #binned_auto_original_debiased_TT = [auto_original_debiased_TT[digitized == i].mean() for i in range(1, len(lbins))]

        # Save
        binned_gmv_clkk[:,ii] = binned_auto_gmv_debiased
        #binned_sqe_clkk[:,ii] = binned_auto_original_debiased

    # Get bias
    bias = np.array(binned_gmv_clkk[:,0]) - np.array(binned_input_clkk)
    bias_alt = np.array(binned_gmv_clkk[:,0]) - np.array(binned_gmv_clkk[:,2])
    trispectrum = np.array(binned_gmv_clkk[:,1]) - np.array(binned_input_clkk)
    trispectrum_alt = np.array(binned_gmv_clkk[:,1]) - np.array(binned_gmv_clkk[:,2])
    bispectrum = np.array(binned_gmv_clkk[:,0]) - np.array(binned_gmv_clkk[:,1])
    bispectrum_from_cross = np.array(cross_gmv[:,0]) - np.array(binned_input_clkk)

    bias_sqe = np.array(binned_sqe_clkk[:,0]) - np.array(binned_input_clkk)
    trispectrum_sqe = np.array(binned_sqe_clkk[:,1]) - np.array(binned_input_clkk)
    trispectrum_alt_sqe = np.array(binned_sqe_clkk[:,1]) - np.array(binned_sqe_clkk[:,2])
    bispectrum_sqe = np.array(binned_sqe_clkk[:,0]) - np.array(binned_sqe_clkk[:,1])

    # Plot
    plt.figure(0)
    plt.clf()
    plt.axhline(y=0, color='gray', alpha=0.5, linestyle='--')
    #plt.axhline(y=1, color='gray', alpha=0.5, linestyle='--')

    plt.plot(bin_centers, bias/binned_input_clkk, color='darkblue', marker='o', linestyle='-', ms=3, alpha=0.8, label="Total Bias")
    plt.plot(bin_centers, trispectrum/binned_input_clkk, color='forestgreen', marker='o', linestyle='-', ms=3, alpha=0.8, label="Trispectrum (subtracting input kappa)")
    #plt.plot(bin_centers, trispectrum_alt/binned_input_clkk, color='lightgreen', marker='o', linestyle='-', ms=3, alpha=0.8, label=f'Trispectrum (subtracting reconstruction with Gaussian foregrounds)')
    plt.plot(bin_centers, bispectrum/binned_input_clkk, color='darkorange', marker='o', linestyle='-', ms=3, alpha=0.8, label="Bispectrum")
    plt.plot(bin_centers, bispectrum_from_cross/binned_input_clkk, color='goldenrod', marker='o', linestyle='-', ms=3, alpha=0.8, label=f'Bispectrum (crossing with input)')
    #plt.plot(bin_centers, (trispectrum+bispectrum)/binned_input_clkk, color='cornflowerblue', marker='o', linestyle='-', ms=3, alpha=0.8, label="Trispectrum + Bispectrum")
    #plt.plot(bin_centers, bias_alt/binned_input_clkk, color='lightskyblue', marker='o', linestyle='-', ms=3, alpha=0.8, label="Total Bias (subtracting reconstruction with Gaussian foregrounds)")
    #plt.plot(bin_centers, np.array(cross_gmv[:,0])/binned_input_clkk, color='darkorange', marker='o', linestyle='-', ms=3, alpha=0.8, label="$C_\ell^{\kappa\kappa}(CMB^{NG,1}+FG^{NG})$ / Input Kappa")
    #plt.plot(bin_centers, np.array(cross_gmv[:,1])/binned_input_clkk, color='goldenrod', marker='o', linestyle='-', ms=3, alpha=0.8, label="$C_\ell^{\kappa\kappa}(CMB^{NG,2}+FG^{NG})$ / Input Kappa")
    #plt.plot(bin_centers, np.array(cross_gmv[:,2])/binned_input_clkk, color='pink', marker='o', linestyle='-', ms=3, alpha=0.8, label="$C_\ell^{\kappa\kappa}(CMB^{NG,1}+FG^{G})$ / Input Kappa")
    #plt.plot(bin_centers, np.array(cross_gmv[:,3])/binned_input_clkk, color='palevioletred', marker='o', linestyle='-', ms=3, alpha=0.8, label="$C_\ell^{\kappa\kappa}(CMB^{NG,2}+FG^{G})$ / Input Kappa")
    #plt.plot(bin_centers, np.array(cross_gmv[:,4])/binned_input_clkk, color='mediumpurple', marker='o', linestyle='-', ms=3, alpha=0.8, label="$C_\ell^{\kappa\kappa}(CMB^{G,2}+FG^{NG})$ / Input Kappa")
    #plt.plot(bin_centers, np.array(cross_gmv[:,5])/binned_input_clkk, color='violet', marker='o', linestyle='-', ms=3, alpha=0.8, label="$C_\ell^{\kappa\kappa}(CMB^{G,1}+FG^{NG})$ / Input Kappa")
    #plt.plot(bin_centers, np.array(cross_gmv[:,6])/binned_input_clkk, color='rosybrown', marker='o', linestyle='-', ms=3, alpha=0.8, label="$C_\ell^{\kappa\kappa}(CMB^{NG,1}+FG^{NG})$ (separated and re-added) / Input Kappa")
    #plt.plot(bin_centers, np.array(binned_gmv_clkk[:,0])/binned_input_clkk, color='darkorange', marker='o', linestyle='-', ms=3, alpha=0.8, label="$C_\ell^{\kappa\kappa}(CMB^{NG,1}+FG^{NG})$ / Input Kappa")
    #plt.plot(bin_centers, np.array(binned_gmv_clkk[:,1])/binned_input_clkk, color='goldenrod', marker='o', linestyle='-', ms=3, alpha=0.8, label="$C_\ell^{\kappa\kappa}(CMB^{NG,2}+FG^{NG})$ / Input Kappa")
    #plt.plot(bin_centers, np.array(binned_gmv_clkk[:,2])/binned_input_clkk, color='pink', marker='o', linestyle='-', ms=3, alpha=0.8, label="$C_\ell^{\kappa\kappa}(CMB^{NG,1}+FG^{G})$ / Input Kappa")
    #plt.plot(bin_centers, np.array(binned_gmv_clkk[:,3])/binned_input_clkk, color='palevioletred', marker='o', linestyle='-', ms=3, alpha=0.8, label="$C_\ell^{\kappa\kappa}(CMB^{NG,2}+FG^{G})$ / Input Kappa")
    #plt.plot(bin_centers, np.array(binned_gmv_clkk[:,4])/binned_input_clkk, color='mediumpurple', marker='o', linestyle='-', ms=3, alpha=0.8, label="$C_\ell^{\kappa\kappa}(CMB^{G,2}+FG^{NG})$ / Input Kappa")
    #plt.plot(bin_centers, np.array(binned_gmv_clkk[:,5])/binned_input_clkk, color='violet', marker='o', linestyle='-', ms=3, alpha=0.8, label="$C_\ell^{\kappa\kappa}(CMB^{G,1}+FG^{NG})$ / Input Kappa")
    #plt.plot(bin_centers, np.array(binned_gmv_clkk[:,6])/binned_input_clkk, color='rosybrown', marker='o', linestyle='-', ms=3, alpha=0.8, label="$C_\ell^{\kappa\kappa}(CMB^{NG,1}+FG^{NG})$ (separated and re-added) / Input Kappa")

    #plt.ylabel("$C_\ell^{\kappa\kappa}$")
    plt.xlabel('$\ell$')
    plt.title(f'Lensing Bias from Agora Sim / Input Kappa Spectrum, lmaxT = {lmaxT}')
    plt.legend(loc='lower left', fontsize='x-small')
    plt.xscale('log')
    #plt.xlim(10,lmax)
    plt.ylim(-0.2,0.2)
    #plt.ylim(-0.5,0.5)
    #plt.ylim(0.8,1.2)
    #plt.ylim(-2,2)
    plt.savefig(dir_out+f'/figs/bias_split_up.png',bbox_inches='tight')

    '''
    plt.clf()
    #plt.axhline(y=0, color='gray', alpha=0.5, linestyle='--')
    plt.axhline(y=1, color='gray', alpha=0.5, linestyle='--')

    #plt.plot(bin_centers, bias/np.array(binned_gmv_clkk[:,2]), color='darkblue', marker='o', linestyle='-', ms=3, alpha=0.8, label="Total Bias")
    #plt.plot(bin_centers, trispectrum/np.array(binned_gmv_clkk[:,2]), color='forestgreen', marker='o', linestyle='-', ms=3, alpha=0.8, label="Trispectrum (subtracting input kappa)")
    #plt.plot(bin_centers, trispectrum_alt/np.array(binned_gmv_clkk[:,2]), color='lightgreen', marker='o', linestyle='-', ms=3, alpha=0.8, label=f'Trispectrum (subtracting reconstruction with Gaussian foregrounds)')
    plt.plot(bin_centers, bispectrum/np.array(binned_gmv_clkk[:,2]), color='darkorange', marker='o', linestyle='-', ms=3, alpha=0.8, label="Bispectrum")
    plt.plot(bin_centers, bispectrum_from_cross/np.array(binned_gmv_clkk[:,2]), color='goldenrod', marker='o', linestyle='-', ms=3, alpha=0.8, label=f'Bispectrum (crossing with input)')
    #plt.plot(bin_centers, (trispectrum+bispectrum)/np.array(binned_gmv_clkk[:,2]), color='cornflowerblue', marker='o', linestyle='-', ms=3, alpha=0.8, label="Trispectrum + Bispectrum")
    #plt.plot(bin_centers, bias_alt/np.array(binned_gmv_clkk[:,2]), color='lightskyblue', marker='o', linestyle='-', ms=3, alpha=0.8, label="Total Bias (subtracting reconstruction with Gaussian foregrounds)")
    #plt.plot(bin_centers, np.array(binned_gmv_clkk[:,0])/np.array(binned_gmv_clkk[:,2]), color='darkorange', marker='o', linestyle='-', ms=3, alpha=0.8, label="$C_\ell^{\kappa\kappa}(CMB^{NG,1}+FG^{NG})$ / $C_\ell^{\kappa\kappa}(CMB^{NG,1}+FG^{G})$")
    #plt.plot(bin_centers, np.array(binned_gmv_clkk[:,1])/np.array(binned_gmv_clkk[:,2]), color='goldenrod', marker='o', linestyle='-', ms=3, alpha=0.8, label="$C_\ell^{\kappa\kappa}(CMB^{NG,2}+FG^{NG})$ / $C_\ell^{\kappa\kappa}(CMB^{NG,1}+FG^{G})$")
    #plt.plot(bin_centers, np.array(binned_gmv_clkk[:,2])/np.array(binned_gmv_clkk[:,2]), color='darkkhaki', marker='o', linestyle='-', ms=3, alpha=0.8, label="$C_\ell^{\kappa\kappa}(CMB^{NG,1}+FG^{G})$ / $C_\ell^{\kappa\kappa}(CMB^{NG,1}+FG^{G})$")

    #plt.ylabel("$C_\ell^{\kappa\kappa}$")
    plt.xlabel('$\ell$')
    plt.title('Lensing Bias from Agora Sim / $C_\ell^{\kappa\kappa}(CMB^{NG,1}+FG^{G})$')
    plt.legend(loc='upper left', fontsize='small')
    plt.xscale('log')
    #plt.xlim(10,lmax)
    #plt.ylim(-0.2,0.2)
    plt.ylim(0.8,1.2)
    #plt.ylim(-2,2)
    plt.savefig(dir_out+f'/figs/bias_split_up_alt.png',bbox_inches='tight')
    '''

    '''
    plt.clf()
    plt.axhline(y=0, color='gray', alpha=0.5, linestyle='--')

    plt.plot(bin_centers, bias_sqe/binned_input_clkk, color='darkblue', marker='o', linestyle='-', ms=3, alpha=0.8, label="Total Bias")
    plt.plot(bin_centers, trispectrum_sqe/binned_input_clkk, color='forestgreen', marker='o', linestyle='-', ms=3, alpha=0.8, label="Trispectrum (subtracting input kappa)")
    plt.plot(bin_centers, trispectrum_alt_sqe/binned_input_clkk, color='lightgreen', marker='o', linestyle='-', ms=3, alpha=0.8, label=f'Trispectrum (subtracting reconstruction with Gaussian foregrounds)')
    plt.plot(bin_centers, bispectrum_sqe/binned_input_clkk, color='darkorange', marker='o', linestyle='-', ms=3, alpha=0.8, label="Bispectrum")
    plt.plot(bin_centers, (trispectrum_sqe+bispectrum_sqe)/binned_input_clkk, color='cornflowerblue', marker='o', linestyle='-', ms=3, alpha=0.8, label="Trispectrum + Bispectrum")

    #plt.ylabel("$C_\ell^{\kappa\kappa}$")
    plt.xlabel('$\ell$')
    plt.title(f'SQE Lensing Bias from Agora Sim / Input Kappa Spectrum')
    plt.legend(loc='upper left', fontsize='small')
    plt.xscale('log')
    plt.xlim(10,lmax)
    #plt.ylim(-0.2,0.2)
    plt.ylim(-2,2)
    plt.savefig(dir_out+f'/figs/bias_split_up_sqe.png',bbox_inches='tight')
    '''

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

####################

analyze()
