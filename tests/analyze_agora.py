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

def analyze(config_file='test_yuka.yaml',
            lbins=np.logspace(np.log10(50),np.log10(3000),20)):
    '''
    Compare with N0/N1 subtraction.
    '''
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
    # Input kappa
    klm = hp.read_alm('/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_spt3g_input_klm_lmax4096.fits')
    input_clkk = hp.alm2cl(klm,klm,lmax=lmax)
    binned_input_clkk = [input_clkk[digitized == i].mean() for i in range(1, len(lbins))]
    append_list = ['agora_standard', 'agora_mh', 'agora_crossilc_onesed', 'agora_crossilc_twoseds', 'agora_profhrd']
    #append_list = ['agora_standard', 'agora_mh', 'agora_crossilc_onesed', 'agora_crossilc_twoseds']
    bias_over_uncertainty_gmv = np.zeros((len(l),len(append_list)), dtype=np.complex_)
    binned_bias_over_uncertainty_gmv = np.zeros((len(bin_centers),len(append_list)), dtype=np.complex_)
    bias_over_uncertainty_sqe = np.zeros((len(l),len(append_list)), dtype=np.complex_)
    binned_bias_over_uncertainty_sqe = np.zeros((len(bin_centers),len(append_list)), dtype=np.complex_)
    bias_over_uncertainty_gmv_TTEETE = np.zeros((len(l),len(append_list)), dtype=np.complex_)
    binned_bias_over_uncertainty_gmv_TTEETE = np.zeros((len(bin_centers),len(append_list)), dtype=np.complex_)
    bias_over_uncertainty_sqe_TT = np.zeros((len(l),len(append_list)), dtype=np.complex_)
    binned_bias_over_uncertainty_sqe_TT = np.zeros((len(bin_centers),len(append_list)), dtype=np.complex_)
    bias_gmv = np.zeros((len(l),len(append_list)), dtype=np.complex_)
    bias_sqe = np.zeros((len(l),len(append_list)), dtype=np.complex_)
    bias_gmv_TTEETE = np.zeros((len(l),len(append_list)), dtype=np.complex_)
    bias_sqe_TT = np.zeros((len(l),len(append_list)), dtype=np.complex_)
    binned_bias_gmv = np.zeros((len(bin_centers),len(append_list)), dtype=np.complex_)
    binned_bias_sqe = np.zeros((len(bin_centers),len(append_list)), dtype=np.complex_)
    binned_bias_gmv_TTEETE = np.zeros((len(bin_centers),len(append_list)), dtype=np.complex_)
    binned_bias_sqe_TT = np.zeros((len(bin_centers),len(append_list)), dtype=np.complex_)
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
        #clkk_filename = dir_out+f'/agora_reconstruction/reconstructed_clkk_{append}.npy'
        #binned_clkk_filename = dir_out+f'/agora_reconstruction/binned_reconstructed_clkk_{append}.npy'

        #if os.path.isfile(clkk_filename):
        #    print('Loading from existing file!')
        #    reconstructed_clkk_gmv = np.load(clkk_filename)[:,0]
        #    binned_reconstructed_clkk_gmv = np.load(binned_clkk_filename)[:,0]
        #    reconstructed_clkk_sqe = np.load(clkk_filename)[:,1]
        #    binned_reconstructed_clkk_sqe = np.load(binned_clkk_filename)[:,1]
        #    reconstructed_clkk_gmv_TTEETE = np.load(clkk_filename)[:,2]
        #    binned_reconstructed_clkk_gmv_TTEETE = np.load(binned_clkk_filename)[:,2]
        #    reconstructed_clkk_sqe_TT = np.load(clkk_filename)[:,3]
        #    binned_reconstructed_clkk_sqe_TT = np.load(binned_clkk_filename)[:,3]
        #else:
        if append == 'agora_standard' or append == 'agora_profhrd':
            # Get SQE response
            ests = ['TT', 'EE', 'TE', 'ET', 'TB', 'BT', 'EB', 'BE']
            resps_original = np.zeros((len(l),len(ests)), dtype=np.complex_)
            inv_resps_original = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
            for i, est in enumerate(ests):
                resps_original[:,i] = get_sim_response(est,config,gmv=False,append=append_alt,sims=np.arange(99)+1)
                inv_resps_original[1:,i] = 1/(resps_original)[1:,i]
            resp_original = np.sum(resps_original, axis=1)
            inv_resp_original = np.zeros_like(l,dtype=np.complex_); inv_resp_original[1:] = 1/(resp_original)[1:]

        elif append == 'agora_mh' or append == 'agora_crossilc_onesed' or append == 'agora_crossilc_twoseds':
            # Get SQE response
            ests = ['T1T2', 'T2T1', 'EE', 'TE', 'ET', 'TB', 'BT', 'EB', 'BE']
            resps_original = np.zeros((len(l),len(ests)), dtype=np.complex_)
            inv_resps_original = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
            for i, est in enumerate(ests):
                resps_original[:,i] = get_sim_response(est,config,gmv=False,append=append_alt,sims=np.arange(99)+1)
                inv_resps_original[1:,i] = 1/(resps_original)[1:,i]
            resp_original = 0.5*resps_original[:,0]+0.5*resps_original[:,1]+np.sum(resps_original[:,2:], axis=1)
            inv_resp_original = np.zeros_like(l,dtype=np.complex_); inv_resp_original[1:] = 1/(resp_original)[1:]

        # GMV response
        resp_gmv = get_sim_response('all',config,gmv=True,append=append_alt,sims=np.arange(99)+1)
        resp_gmv_TTEETE = get_sim_response('TTEETE',config,gmv=True,append=append_alt,sims=np.arange(99)+1)
        resp_gmv_TBEB = get_sim_response('TBEB',config,gmv=True,append=append_alt,sims=np.arange(99)+1)
        inv_resp_gmv = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv[1:] = 1./(resp_gmv)[1:]
        inv_resp_gmv_TTEETE = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TTEETE[1:] = 1./(resp_gmv_TTEETE)[1:]
        inv_resp_gmv_TBEB = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TBEB[1:] = 1./(resp_gmv_TBEB)[1:]

        if append == 'agora_profhrd':
            # Get the profile response and weight
            # SQE
            resp_original_TT_ss = get_analytic_response('TTprf',config,gmv=False,append=append_alt,u=u)
            resp_original_TT_sk = get_analytic_response('TTTTprf',config,gmv=False,append=append_alt,u=u)
            weight_original = -1 * resp_original_TT_sk / resp_original_TT_ss
            resp_original_hrd = resp_original + weight_original*resp_original_TT_sk # Equivalent to resp_original_TT (hardened) + np.sum(resps_original[:,1:], axis=1)
            resp_original_TT_hrd = resps_original[:,0] + weight_original*resp_original_TT_sk
            inv_resp_original_hrd = np.zeros_like(l,dtype=np.complex_); inv_resp_original_hrd[1:] = 1/(resp_original_hrd)[1:]
            inv_resp_original_TT_hrd = np.zeros_like(l,dtype=np.complex_); inv_resp_original_TT_hrd[1:] = 1/(resp_original_TT_hrd)[1:]

            # GMV
            resp_gmv_TTEETE_ss = get_analytic_response('TTEETEprf',config,gmv=True,append=append_alt,u=u[lmin:])
            resp_gmv_TTEETE_sk = get_analytic_response('TTEETETTEETEprf',config,gmv=True,append=append_alt,u=u[lmin:])
            weight_gmv = -1 * resp_gmv_TTEETE_sk / resp_gmv_TTEETE_ss
            resp_gmv_hrd = resp_gmv + weight_gmv*resp_gmv_TTEETE_sk # Equivalent to resp_gmv_TTEETE (hardened) + resp_gmv_TBEB (unhardened)
            resp_gmv_TTEETE_hrd = resp_gmv_TTEETE + weight_gmv*resp_gmv_TTEETE_sk
            inv_resp_gmv_hrd = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_hrd[1:] = 1./(resp_gmv_hrd)[1:]
            inv_resp_gmv_TTEETE_hrd = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TTEETE_hrd[1:] = 1./(resp_gmv_TTEETE_hrd)[1:]

        # Get N0 correction (remember these still need (l*(l+1))**2/4 factor to convert to kappa)
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
        if append == 'agora_mh':
            n0_original_TT = n0_original['T1T2'] * (l*(l+1))**2/4
        else:
            n0_original_TT = n0_original['TT'] * (l*(l+1))**2/4
        n0_original_EE = n0_original['EE'] * (l*(l+1))**2/4
        n0_original_TE = n0_original['TE'] * (l*(l+1))**2/4
        n0_original_ET = n0_original['ET'] * (l*(l+1))**2/4
        n0_original_TB = n0_original['TB'] * (l*(l+1))**2/4
        n0_original_BT = n0_original['BT'] * (l*(l+1))**2/4
        n0_original_EB = n0_original['EB'] * (l*(l+1))**2/4
        n0_original_BE = n0_original['BE'] * (l*(l+1))**2/4
        if append == 'agora_profhrd':
            n0_original_total_hrd = n0_original['total_hrd'] * (l*(l+1))**2/4
            n0_original_TT_hrd = n0_original['TT_hrd'] * (l*(l+1))**2/4

        n1_gmv = get_n1(sims=np.arange(98)+1,qetype='gmv',config=config,
                        append=append_alt)
        n1_gmv_total = n1_gmv['total'] * (l*(l+1))**2/4
        n1_gmv_TTEETE = n1_gmv['TTEETE'] * (l*(l+1))**2/4
        n1_gmv_TBEB = n1_gmv['TBEB'] * (l*(l+1))**2/4
        n1_original = get_n1(sims=np.arange(98)+1,qetype='sqe',config=config,
                             append=append_alt)
        n1_original_total = n1_original['total'] * (l*(l+1))**2/4
        if append == 'agora_mh' or append == 'agora_crossilc_onesed' or append == 'agora_crossilc_twoseds':
            n1_original_TT = n1_original['T1T2'] * (l*(l+1))**2/4
        else:
            n1_original_TT = n1_original['TT'] * (l*(l+1))**2/4
        n1_original_EE = n1_original['EE'] * (l*(l+1))**2/4
        n1_original_TE = n1_original['TE'] * (l*(l+1))**2/4
        n1_original_ET = n1_original['ET'] * (l*(l+1))**2/4
        n1_original_TB = n1_original['TB'] * (l*(l+1))**2/4
        n1_original_BT = n1_original['BT'] * (l*(l+1))**2/4
        n1_original_EB = n1_original['EB'] * (l*(l+1))**2/4
        n1_original_BT = n1_original['BE'] * (l*(l+1))**2/4

        # Load GMV plms
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
            plm_gmv_hrd = plm_gmv + hp.almxfl(glm_prf_TTEETE, weight_gmv) # Equivalent to plm_gmv_TTEETE_hrd + plm_gmv_TBEB

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
            plm_original_resp_corr_EE = hp.almxfl(plms_original[:,2],inv_resps_original[:,2])
            plm_original_resp_corr_TE = hp.almxfl(plms_original[:,3],inv_resps_original[:,3])
            plm_original_resp_corr_ET = hp.almxfl(plms_original[:,4],inv_resps_original[:,4])
            plm_original_resp_corr_TB = hp.almxfl(plms_original[:,5],inv_resps_original[:,5])
            plm_original_resp_corr_BT = hp.almxfl(plms_original[:,6],inv_resps_original[:,6])
            plm_original_resp_corr_EB = hp.almxfl(plms_original[:,7],inv_resps_original[:,7])
            plm_original_resp_corr_BE = hp.almxfl(plms_original[:,8],inv_resps_original[:,8])
        else:
            plm_original_resp_corr_TT = hp.almxfl(plms_original[:,0],inv_resps_original[:,0])
            plm_original_resp_corr_EE = hp.almxfl(plms_original[:,1],inv_resps_original[:,1])
            plm_original_resp_corr_TE = hp.almxfl(plms_original[:,2],inv_resps_original[:,2])
            plm_original_resp_corr_ET = hp.almxfl(plms_original[:,3],inv_resps_original[:,3])
            plm_original_resp_corr_TB = hp.almxfl(plms_original[:,4],inv_resps_original[:,4])
            plm_original_resp_corr_BT = hp.almxfl(plms_original[:,5],inv_resps_original[:,5])
            plm_original_resp_corr_EB = hp.almxfl(plms_original[:,6],inv_resps_original[:,6])
            plm_original_resp_corr_BE = hp.almxfl(plms_original[:,7],inv_resps_original[:,7])
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
        auto_original_EE = hp.alm2cl(plm_original_resp_corr_EE, plm_original_resp_corr_EE, lmax=lmax) * (l*(l+1))**2/4
        auto_original_TE = hp.alm2cl(plm_original_resp_corr_TE, plm_original_resp_corr_TE, lmax=lmax) * (l*(l+1))**2/4
        auto_original_ET = hp.alm2cl(plm_original_resp_corr_ET, plm_original_resp_corr_ET, lmax=lmax) * (l*(l+1))**2/4
        auto_original_TB = hp.alm2cl(plm_original_resp_corr_TB, plm_original_resp_corr_TB, lmax=lmax) * (l*(l+1))**2/4
        auto_original_BT = hp.alm2cl(plm_original_resp_corr_BT, plm_original_resp_corr_BT, lmax=lmax) * (l*(l+1))**2/4
        auto_original_EB = hp.alm2cl(plm_original_resp_corr_EB, plm_original_resp_corr_EB, lmax=lmax) * (l*(l+1))**2/4
        auto_original_BE = hp.alm2cl(plm_original_resp_corr_BE, plm_original_resp_corr_BE, lmax=lmax) * (l*(l+1))**2/4
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
        auto_gmv_debiased_TTEETE = auto_gmv_TTEETE - n0_gmv_TTEETE - n1_gmv_TTEETE
        auto_original_debiased = auto_original - n0_original_total - n1_original_total
        if append == 'agora_mh' or append == 'agora_crossilc_onesed' or append == 'agora_crossilc_twoseds':
            auto_original_debiased_TT = 0.5*(auto_original_T1T2+auto_original_T2T1) - n0_original_TT - n1_original_TT
        else:
            auto_original_debiased_TT = auto_original_TT - n0_original_TT - n1_original_TT
        if append == 'agora_profhrd':
            auto_gmv_debiased_hrd = auto_gmv_hrd - n0_gmv_total_hrd - n1_gmv_total
            auto_original_debiased_hrd = auto_original_hrd - n0_original_total_hrd - n1_original_total
            auto_gmv_debiased_hrd_TTEETE = auto_gmv_TTEETE_hrd - n0_gmv_TTEETE_hrd - n1_gmv_TTEETE
            auto_original_debiased_hrd_TT = auto_original_TT_hrd - n0_original_TT_hrd - n1_original_TT

        # Bin!
        ratio_gmv = np.zeros((len(lbins)-1),dtype=np.complex_)
        ratio_original = np.zeros((len(lbins)-1),dtype=np.complex_)
        ratio_gmv_TTEETE = np.zeros((len(lbins)-1),dtype=np.complex_)
        ratio_original_TT = np.zeros((len(lbins)-1),dtype=np.complex_)
        binned_auto_gmv_debiased = [auto_gmv_debiased[digitized == i].mean() for i in range(1, len(lbins))]
        binned_auto_original_debiased = [auto_original_debiased[digitized == i].mean() for i in range(1, len(lbins))]
        binned_auto_gmv_debiased_TTEETE = [auto_gmv_debiased_TTEETE[digitized == i].mean() for i in range(1, len(lbins))]
        binned_auto_original_debiased_TT = [auto_original_debiased_TT[digitized == i].mean() for i in range(1, len(lbins))]
        if append == 'agora_profhrd':
            ratio_gmv_hrd = np.zeros((len(lbins)-1),dtype=np.complex_)
            ratio_original_hrd = np.zeros((len(lbins)-1),dtype=np.complex_)
            ratio_gmv_hrd_TTEETE = np.zeros((len(lbins)-1),dtype=np.complex_)
            ratio_original_hrd_TT = np.zeros((len(lbins)-1),dtype=np.complex_)
            binned_auto_gmv_debiased_hrd = [auto_gmv_debiased_hrd[digitized == i].mean() for i in range(1, len(lbins))]
            binned_auto_original_debiased_hrd = [auto_original_debiased_hrd[digitized == i].mean() for i in range(1, len(lbins))]
            binned_auto_gmv_debiased_hrd_TTEETE = [auto_gmv_debiased_hrd_TTEETE[digitized == i].mean() for i in range(1, len(lbins))]
            binned_auto_original_debiased_hrd_TT = [auto_original_debiased_hrd_TT[digitized == i].mean() for i in range(1, len(lbins))]

        # Save...
        if append == 'agora_profhrd':
            reconstructed_clkk = np.zeros((len(l),8), dtype=np.complex_)
            binned_reconstructed_clkk = np.zeros((len(bin_centers),8), dtype=np.complex_)
            reconstructed_clkk[:,0] = auto_gmv_debiased_hrd
            reconstructed_clkk[:,1] = auto_original_debiased_hrd
            reconstructed_clkk[:,2] = auto_gmv_debiased_hrd_TTEETE
            reconstructed_clkk[:,3] = auto_original_debiased_hrd_TT
            reconstructed_clkk[:,4] = auto_gmv_debiased
            reconstructed_clkk[:,5] = auto_original_debiased
            reconstructed_clkk[:,6] = auto_gmv_debiased_TTEETE
            reconstructed_clkk[:,7] = auto_original_debiased_TT
            binned_reconstructed_clkk[:,0] = binned_auto_gmv_debiased_hrd
            binned_reconstructed_clkk[:,1] = binned_auto_original_debiased_hrd
            binned_reconstructed_clkk[:,2] = binned_auto_gmv_debiased_hrd_TTEETE
            binned_reconstructed_clkk[:,3] = binned_auto_original_debiased_hrd_TT
            binned_reconstructed_clkk[:,4] = binned_auto_gmv_debiased
            binned_reconstructed_clkk[:,5] = binned_auto_original_debiased
            binned_reconstructed_clkk[:,6] = binned_auto_gmv_debiased_TTEETE
            binned_reconstructed_clkk[:,7] = binned_auto_original_debiased_TT
        else:
            reconstructed_clkk = np.zeros((len(l),4), dtype=np.complex_)
            binned_reconstructed_clkk = np.zeros((len(bin_centers),4), dtype=np.complex_)
            reconstructed_clkk[:,0] = auto_gmv_debiased
            reconstructed_clkk[:,1] = auto_original_debiased
            reconstructed_clkk[:,2] = auto_gmv_debiased_TTEETE
            reconstructed_clkk[:,3] = auto_original_debiased_TT
            binned_reconstructed_clkk[:,0] = binned_auto_gmv_debiased
            binned_reconstructed_clkk[:,1] = binned_auto_original_debiased
            binned_reconstructed_clkk[:,2] = binned_auto_gmv_debiased_TTEETE
            binned_reconstructed_clkk[:,3] = binned_auto_original_debiased_TT
        #np.save(clkk_filename,reconstructed_clkk)
        #np.save(binned_clkk_filename,binned_reconstructed_clkk)

        reconstructed_clkk_gmv = reconstructed_clkk[:,0]
        binned_reconstructed_clkk_gmv = binned_reconstructed_clkk[:,0]
        reconstructed_clkk_sqe = reconstructed_clkk[:,1]
        binned_reconstructed_clkk_sqe = binned_reconstructed_clkk[:,1]
        reconstructed_clkk_gmv_TTEETE = reconstructed_clkk[:,2]
        binned_reconstructed_clkk_gmv_TTEETE = binned_reconstructed_clkk[:,2]
        reconstructed_clkk_sqe_TT = reconstructed_clkk[:,3]
        binned_reconstructed_clkk_sqe_TT = binned_reconstructed_clkk[:,3]

        # Get bias
        # Theory spectrum
        clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
        ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
        clkk = slpp * (l*(l+1))**2/4
        binned_clkk = [clkk[digitized == i].mean() for i in range(1, len(lbins))]
        reconstructed_dclkk_gmv = reconstructed_clkk_gmv - input_clkk
        binned_reconstructed_dclkk_gmv = [reconstructed_dclkk_gmv[digitized == i].mean() for i in range(1, len(lbins))]
        reconstructed_dclkk_sqe = reconstructed_clkk_sqe - input_clkk
        binned_reconstructed_dclkk_sqe = [reconstructed_dclkk_sqe[digitized == i].mean() for i in range(1, len(lbins))]
        reconstructed_dclkk_gmv_TTEETE = reconstructed_clkk_gmv_TTEETE - input_clkk
        binned_reconstructed_dclkk_gmv_TTEETE = [reconstructed_dclkk_gmv_TTEETE[digitized == i].mean() for i in range(1, len(lbins))]
        reconstructed_dclkk_sqe_TT = reconstructed_clkk_sqe_TT - input_clkk
        binned_reconstructed_dclkk_sqe_TT = [reconstructed_dclkk_sqe_TT[digitized == i].mean() for i in range(1, len(lbins))]

        bias_gmv[:,j] = reconstructed_dclkk_gmv
        bias_sqe[:,j] = reconstructed_dclkk_sqe
        bias_gmv_TTEETE[:,j] = reconstructed_dclkk_gmv_TTEETE
        bias_sqe_TT[:,j] = reconstructed_dclkk_sqe_TT
        binned_bias_gmv[:,j] = binned_reconstructed_dclkk_gmv
        binned_bias_sqe[:,j] = binned_reconstructed_dclkk_sqe
        binned_bias_gmv_TTEETE[:,j] = binned_reconstructed_dclkk_gmv_TTEETE
        binned_bias_sqe_TT[:,j] = binned_reconstructed_dclkk_sqe_TT

        # Get uncertainty
        uncertainty_gmv = np.load(dir_out+f'/agora_reconstruction/measurement_uncertainty_lmaxT{lmaxT}_{append_alt}.npy')[:,0]
        uncertainty_sqe = np.load(dir_out+f'/agora_reconstruction/measurement_uncertainty_lmaxT{lmaxT}_{append_alt}.npy')[:,1]
        uncertainty_gmv_TTEETE = np.load(dir_out+f'/agora_reconstruction/measurement_uncertainty_lmaxT{lmaxT}_{append_alt}.npy')[:,2]
        uncertainty_sqe_TT = np.load(dir_out+f'/agora_reconstruction/measurement_uncertainty_lmaxT{lmaxT}_{append_alt}.npy')[:,3]
        binned_uncertainty_gmv[:,j] = np.load(dir_out+f'/agora_reconstruction/binned_measurement_uncertainty_lmaxT{lmaxT}_{append_alt}.npy')[:,0]
        binned_uncertainty_sqe[:,j] = np.load(dir_out+f'/agora_reconstruction/binned_measurement_uncertainty_lmaxT{lmaxT}_{append_alt}.npy')[:,1]
        binned_uncertainty_gmv_TTEETE[:,j] = np.load(dir_out+f'/agora_reconstruction/binned_measurement_uncertainty_lmaxT{lmaxT}_{append_alt}.npy')[:,2]
        binned_uncertainty_sqe_TT[:,j] = np.load(dir_out+f'/agora_reconstruction/binned_measurement_uncertainty_lmaxT{lmaxT}_{append_alt}.npy')[:,3]

        bias_over_uncertainty_gmv[:,j] = reconstructed_dclkk_gmv / uncertainty_gmv
        binned_bias_over_uncertainty_gmv[:,j] = binned_reconstructed_dclkk_gmv / binned_uncertainty_gmv[:,j]
        bias_over_uncertainty_sqe[:,j] = reconstructed_dclkk_sqe / uncertainty_sqe
        binned_bias_over_uncertainty_sqe[:,j] = binned_reconstructed_dclkk_sqe / binned_uncertainty_sqe[:,j]
        bias_over_uncertainty_gmv_TTEETE[:,j] = reconstructed_dclkk_gmv_TTEETE / uncertainty_gmv_TTEETE
        binned_bias_over_uncertainty_gmv_TTEETE[:,j] = binned_reconstructed_dclkk_gmv_TTEETE / binned_uncertainty_gmv_TTEETE[:,j]
        bias_over_uncertainty_sqe_TT[:,j] = reconstructed_dclkk_sqe_TT / uncertainty_sqe_TT
        binned_bias_over_uncertainty_sqe_TT[:,j] = binned_reconstructed_dclkk_sqe_TT / binned_uncertainty_sqe_TT[:,j]

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
        

    # Plot
    plt.figure(0)
    plt.clf()
    plt.axhline(y=0, color='gray', alpha=0.5, linestyle='-')
    #plt.plot(l, bias_over_uncertainty_gmv[:,2], color='lightgreen', linestyle='-', label="Cross-ILC GMV (one component CIB)")
    #plt.plot(l, bias_over_uncertainty_gmv[:,3], color='thistle', linestyle='-', label="Cross-ILC GMV (two component CIB)")
    #plt.plot(l, bias_over_uncertainty_gmv[:,1], color='khaki', linestyle='-', label="MH GMV")
    #plt.plot(l, bias_over_uncertainty_gmv[:,4], color='bisque', linestyle='-', label="Profile Hardened GMV")
    #plt.plot(l, bias_over_uncertainty_sqe[:,0], color='lightcoral', linestyle='-', label=f'Standard SQE')
    #plt.plot(l, bias_over_uncertainty_gmv[:,0], color='cornflowerblue', linestyle='-', label="Standard GMV")

    plt.plot(bin_centers, binned_bias_over_uncertainty_gmv[:,2], color='forestgreen', marker='o', linestyle='--', ms=3, alpha=0.8, label="Cross-ILC GMV (one component CIB)")
    plt.plot(bin_centers, binned_bias_over_uncertainty_gmv[:,3], color='plum', marker='o', linestyle='--', ms=3, alpha=0.8, label="Cross-ILC GMV (two component CIB)")
    plt.plot(bin_centers, binned_bias_over_uncertainty_gmv[:,1], color='goldenrod', marker='o', linestyle='--', ms=3, alpha=0.8, label="MH GMV")
    plt.plot(bin_centers, binned_bias_over_uncertainty_gmv[:,4], color='darkorange', marker='o', linestyle='--', ms=3, alpha=0.8, label="Profile Hardened GMV")
    plt.plot(bin_centers, binned_bias_over_uncertainty_sqe[:,0], color='firebrick', marker='o', linestyle='--', ms=3, alpha=0.8, label=f'Standard SQE')
    plt.plot(bin_centers, binned_bias_over_uncertainty_gmv[:,0], color='darkblue', marker='o', linestyle='--', ms=3, alpha=0.8, label="Standard GMV")

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
    #plt.plot(l, bias_over_uncertainty_gmv_TTEETE[:,2], color='lightgreen', linestyle='-', label="Cross-ILC GMV (one component CIB)")
    #plt.plot(l, bias_over_uncertainty_gmv_TTEETE[:,3], color='thistle', linestyle='-', label="Cross-ILC GMV (two component CIB)")
    #plt.plot(l, bias_over_uncertainty_gmv_TTEETE[:,1], color='khaki', linestyle='-', label="MH GMV")
    #plt.plot(l, bias_over_uncertainty_gmv_TTEETE[:,4], color='bisque', linestyle='-', label="Profile Hardened GMV")
    #plt.plot(l, bias_over_uncertainty_sqe_TT[:,0], color='lightcoral', linestyle='-', label=f'Standard SQE')
    #plt.plot(l, bias_over_uncertainty_gmv_TTEETE[:,0], color='cornflowerblue', linestyle='-', label="Standard GMV")

    plt.plot(bin_centers, binned_bias_over_uncertainty_gmv_TTEETE[:,2], color='forestgreen', marker='o', linestyle='--', ms=3, alpha=0.8, label="Cross-ILC GMV (one component CIB)")
    plt.plot(bin_centers, binned_bias_over_uncertainty_gmv_TTEETE[:,3], color='plum', marker='o', linestyle='--', ms=3, alpha=0.8, label="Cross-ILC GMV (two component CIB)")
    plt.plot(bin_centers, binned_bias_over_uncertainty_gmv_TTEETE[:,1], color='goldenrod', marker='o', linestyle='--', ms=3, alpha=0.8, label="MH GMV")
    plt.plot(bin_centers, binned_bias_over_uncertainty_gmv_TTEETE[:,4], color='darkorange', marker='o', linestyle='--', ms=3, alpha=0.8, label="Profile Hardened GMV")
    plt.plot(bin_centers, binned_bias_over_uncertainty_sqe_TT[:,0], color='firebrick', marker='o', linestyle='--', ms=3, alpha=0.8, label=f'Standard SQE')
    plt.plot(bin_centers, binned_bias_over_uncertainty_gmv_TTEETE[:,0], color='darkblue', marker='o', linestyle='--', ms=3, alpha=0.8, label="Standard GMV")

    #plt.ylabel("$C_\ell^{\kappa\kappa}$")
    plt.xlabel('$\ell$')
    plt.title(f'Bias / Uncertainty (TTEETE only)')
    plt.legend(loc='upper left', fontsize='x-small', bbox_to_anchor=(1,0.5))
    plt.xscale('log')
    plt.xlim(10,lmax)
    plt.ylim(-0.5,0.5)
    plt.savefig(dir_out+f'/figs/bias_over_uncertainty_TTEETE.png',bbox_inches='tight')

    plt.figure(1)
    plt.clf()
    #plt.plot(l, bias_gmv[:,2]/clkk, color='lightgreen', linestyle='-', label="Cross-ILC GMV (one component CIB)")
    #plt.plot(l, bias_gmv[:,3]/clkk, color='thistle', linestyle='-', label="Cross-ILC GMV (two component CIB)")
    #plt.plot(l, bias_gmv[:,1]/clkk, color='khaki', linestyle='-', label="MH GMV")
    #plt.plot(l, bias_gmv[:,4]/clkk, color='bisque', linestyle='-', label="Profile Hardened GMV")
    #plt.plot(l, bias_sqe[:,0]/clkk, color='lightcoral', linestyle='-', label=f'Standard SQE')
    #plt.plot(l, bias_gmv[:,0]/clkk, color='cornflowerblue', linestyle='-', label="Standard GMV")
    plt.axhline(y=0, color='gray', alpha=0.5, linestyle='--')

    #plt.plot(bin_centers, binned_bias_gmv[:,2]/binned_input_clkk, color='forestgreen', marker='o', linestyle='--', ms=3, alpha=0.8, label="Cross-ILC GMV (one component CIB)")
    #plt.plot(bin_centers, binned_bias_gmv[:,3]/binned_input_clkk, color='plum', marker='o', linestyle='--', ms=3, alpha=0.8, label="Cross-ILC GMV (two component CIB)")
    #plt.plot(bin_centers, binned_bias_gmv[:,1]/binned_input_clkk, color='goldenrod', marker='o', linestyle='--', ms=3, alpha=0.8, label="MH GMV")
    #plt.plot(bin_centers, binned_bias_gmv[:,4]/binned_input_clkk, color='darkorange', marker='o', linestyle='--', ms=3, alpha=0.8, label="Profile Hardened GMV")
    #plt.plot(bin_centers, binned_bias_sqe[:,0]/binned_input_clkk, color='firebrick', marker='o', linestyle='--', ms=3, alpha=0.8, label=f'Standard SQE')
    #plt.plot(bin_centers, binned_bias_gmv[:,0]/binned_input_clkk, color='darkblue', marker='o', linestyle='--', ms=3, alpha=0.8, label="Standard GMV")

    plt.errorbar(bin_centers, binned_bias_gmv[:,2]/binned_input_clkk, yerr=binned_uncertainty_gmv[:,2]/binned_input_clkk, color='forestgreen', marker='o', linestyle='-', ms=3, alpha=0.8, label="Cross-ILC GMV (one component CIB)")
    plt.errorbar(bin_centers, binned_bias_gmv[:,3]/binned_input_clkk, yerr=binned_uncertainty_gmv[:,3]/binned_input_clkk, color='plum', marker='o', linestyle='-', ms=3, alpha=0.8, label="Cross-ILC GMV (two component CIB)")
    plt.errorbar(bin_centers, binned_bias_gmv[:,1]/binned_input_clkk, yerr=binned_uncertainty_gmv[:,1]/binned_input_clkk, color='goldenrod', marker='o', linestyle='-', ms=3, alpha=0.8, label="MH GMV")
    plt.errorbar(bin_centers, binned_bias_gmv[:,4]/binned_input_clkk, yerr=binned_uncertainty_gmv[:,4]/binned_input_clkk, color='darkorange', marker='o', linestyle='-', ms=3, alpha=0.8, label="Profile Hardened GMV")
    plt.errorbar(bin_centers, binned_bias_sqe[:,0]/binned_input_clkk, yerr=binned_uncertainty_sqe[:,0]/binned_input_clkk, color='firebrick', marker='o', linestyle='-', ms=3, alpha=0.8, label=f'Standard SQE')
    plt.errorbar(bin_centers, binned_bias_gmv[:,0]/binned_input_clkk, yerr=binned_uncertainty_gmv[:,0]/binned_input_clkk, color='darkblue', marker='o', linestyle='-', ms=3, alpha=0.8, label="Standard GMV")

    plt.ylabel("$C_\ell^{\kappa\kappa}$")
    plt.xlabel('$\ell$')
    plt.title(f'Bias from Agora Sim / Input Kappa Spectrum')
    plt.legend(loc='upper left', fontsize='x-small', bbox_to_anchor=(1,0.5))
    plt.xscale('log')
    plt.xlim(10,lmax)
    #plt.ylim(-0.3,0.3)
    plt.ylim(-2,2)
    plt.savefig(dir_out+f'/figs/bias_total.png',bbox_inches='tight')

    plt.clf()
    #plt.plot(l, bias_gmv_TTEETE[:,2]/clkk, color='lightgreen', linestyle='-', label="Cross-ILC GMV (one component CIB)")
    #plt.plot(l, bias_gmv_TTEETE[:,3]/clkk, color='thistle', linestyle='-', label="Cross-ILC GMV (two component CIB)")
    #plt.plot(l, bias_gmv_TTEETE[:,1]/clkk, color='khaki', linestyle='-', label="MH GMV")
    #plt.plot(l, bias_gmv_TTEETE[:,4]/clkk, color='bisque', linestyle='-', label="Profile Hardened GMV")
    #plt.plot(l, bias_sqe_TT[:,0]/clkk, color='lightcoral', linestyle='-', label=f'Standard SQE')
    #plt.plot(l, bias_gmv_TTEETE[:,0]/clkk, color='cornflowerblue', linestyle='-', label="Standard GMV")
    plt.axhline(y=0, color='gray', alpha=0.5, linestyle='--')

    #plt.plot(bin_centers, binned_bias_gmv_TTEETE[:,2]/binned_input_clkk, color='forestgreen', marker='o', linestyle='--', ms=3, alpha=0.8, label="Cross-ILC GMV (one component CIB)")
    #plt.plot(bin_centers, binned_bias_gmv_TTEETE[:,3]/binned_input_clkk, color='plum', marker='o', linestyle='--', ms=3, alpha=0.8, label="Cross-ILC GMV (two component CIB)")
    #plt.plot(bin_centers, binned_bias_gmv_TTEETE[:,1]/binned_input_clkk, color='goldenrod', marker='o', linestyle='--', ms=3, alpha=0.8, label="MH GMV")
    #plt.plot(bin_centers, binned_bias_gmv_TTEETE[:,4]/binned_input_clkk, color='darkorange', marker='o', linestyle='--', ms=3, alpha=0.8, label="Profile Hardened GMV")
    #plt.plot(bin_centers, binned_bias_sqe_TT[:,0]/binned_input_clkk, color='firebrick', marker='o', linestyle='--', ms=3, alpha=0.8, label=f'Standard SQE')
    #plt.plot(bin_centers, binned_bias_gmv_TTEETE[:,0]/binned_input_clkk, color='darkblue', marker='o', linestyle='--', ms=3, alpha=0.8, label="Standard GMV")

    #plt.errorbar(bin_centers, binned_bias_gmv_TTEETE[:,2]/binned_input_clkk, yerr=binned_uncertainty_gmv_TTEETE[:,2]/binned_input_clkk, color='forestgreen', marker='o', linestyle='-', ms=3, alpha=0.8, label="Cross-ILC GMV (one component CIB)")
    #plt.errorbar(bin_centers, binned_bias_gmv_TTEETE[:,3]/binned_input_clkk, yerr=binned_uncertainty_gmv_TTEETE[:,3]/binned_input_clkk, color='plum', marker='o', linestyle='-', ms=3, alpha=0.8, label="Cross-ILC GMV (two component CIB)")
    #plt.errorbar(bin_centers, binned_bias_gmv_TTEETE[:,1]/binned_input_clkk, yerr=binned_uncertainty_gmv_TTEETE[:,1]/binned_input_clkk, color='goldenrod', marker='o', linestyle='-', ms=3, alpha=0.8, label="MH GMV")
    #plt.errorbar(bin_centers, binned_bias_gmv_TTEETE[:,4]/binned_input_clkk, yerr=binned_uncertainty_gmv_TTEETE[:,4]/binned_input_clkk, color='darkorange', marker='o', linestyle='-', ms=3, alpha=0.8, label="Profile Hardened GMV")
    plt.errorbar(bin_centers, binned_bias_sqe_TT[:,0]/binned_input_clkk, yerr=binned_uncertainty_sqe_TT[:,0]/binned_input_clkk, color='firebrick', marker='o', linestyle='-', ms=3, alpha=0.8, label=f'Standard SQE')
    #plt.errorbar(bin_centers, binned_bias_gmv_TTEETE[:,0]/binned_input_clkk, yerr=binned_uncertainty_gmv_TTEETE[:,0]/binned_input_clkk, color='darkblue', marker='o', linestyle='-', ms=3, alpha=0.8, label="Standard GMV")

    plt.ylabel("$C_\ell^{\kappa\kappa}$")
    plt.xlabel('$\ell$')
    plt.title(f'Bias from Agora Sim / Input Kappa Spectrum (TTEETE only)')
    plt.legend(loc='upper left', fontsize='x-small', bbox_to_anchor=(1,0.5))
    plt.xscale('log')
    plt.xlim(10,lmax)
    #plt.ylim(-0.5,0.5)
    plt.ylim(-2,2)
    plt.savefig(dir_out+f'/figs/bias_TTEETE.png',bbox_inches='tight')

    plt.clf()
    # Uncertainty/input kappa test plots
    plt.plot(bin_centers, binned_uncertainty_sqe_TT[:,0], color='firebrick', marker='o', linestyle='-', ms=3, alpha=0.8, label="Standard SQE TT")
    plt.plot(bin_centers, binned_uncertainty_gmv[:,0], color='darkblue', marker='o', linestyle='-', ms=3, alpha=0.8, label="Standard GMV Total")
    plt.xlabel('$\ell$')
    plt.legend(loc='upper left', fontsize='x-small', bbox_to_anchor=(1,0.5))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,lmax)
    plt.savefig(dir_out+f'/figs/uncertainty_over_input_clkk.png',bbox_inches='tight')

    plt.figure(2)
    plt.clf()
    # Cross with input
    plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')
    plt.plot(l, input_clkk, 'gray',linestyle='--', label='Input $C_\ell^{\kappa\kappa}$')

    plt.plot(bin_centers, cross_gmv[:,2], color='forestgreen', marker='o', linestyle='None', ms=3, alpha=0.8, label="Cross-ILC GMV (one component CIB)")
    plt.plot(bin_centers, cross_gmv[:,3], color='plum', marker='o', linestyle='None', ms=3, alpha=0.8, label="Cross-ILC GMV (two component CIB)")
    plt.plot(bin_centers, cross_gmv[:,1], color='goldenrod', marker='o', linestyle='None', ms=3, alpha=0.8, label="MH GMV")
    plt.plot(bin_centers, cross_gmv[:,4], color='darkorange', marker='o', linestyle='None', ms=3, alpha=0.8, label="Profile Hardened GMV")
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

    plt.plot(bin_centers, cross_gmv_TTEETE[:,2], color='forestgreen', marker='o', linestyle='None', ms=3, alpha=0.8, label="Cross-ILC GMV (one component CIB)")
    plt.plot(bin_centers, cross_gmv_TTEETE[:,3], color='plum', marker='o', linestyle='None', ms=3, alpha=0.8, label="Cross-ILC GMV (two component CIB)")
    plt.plot(bin_centers, cross_gmv_TTEETE[:,1], color='goldenrod', marker='o', linestyle='None', ms=3, alpha=0.8, label="MH GMV")
    plt.plot(bin_centers, cross_gmv_TTEETE[:,4], color='darkorange', marker='o', linestyle='None', ms=3, alpha=0.8, label="Profile Hardened GMV")
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

    plt.plot(bin_centers, cross_gmv[:,2]/binned_input_clkk, color='forestgreen', marker='o', linestyle='--', ms=3, alpha=0.8, label="Cross-ILC GMV (one component CIB)")
    plt.plot(bin_centers, cross_gmv[:,3]/binned_input_clkk, color='plum', marker='o', linestyle='--', ms=3, alpha=0.8, label="Cross-ILC GMV (two component CIB)")
    plt.plot(bin_centers, cross_gmv[:,1]/binned_input_clkk, color='goldenrod', marker='o', linestyle='--', ms=3, alpha=0.8, label="MH GMV")
    plt.plot(bin_centers, cross_gmv[:,4]/binned_input_clkk, color='darkorange', marker='o', linestyle='--', ms=3, alpha=0.8, label="Profile Hardened GMV")
    plt.plot(bin_centers, cross_sqe[:,0]/binned_input_clkk, color='firebrick', marker='o', linestyle='--', ms=3, alpha=0.8, label=f'Standard SQE')
    plt.plot(bin_centers, cross_gmv[:,0]/binned_input_clkk, color='darkblue', marker='o', linestyle='--', ms=3, alpha=0.8, label="Standard GMV")

    plt.xlabel('$\ell$')
    plt.title(f'Agora Reconstruction x Input Kappa / Input Kappa')
    plt.legend(loc='upper left', fontsize='x-small', bbox_to_anchor=(1,0.5))
    plt.xscale('log')
    plt.xlim(10,lmax)
    plt.savefig(dir_out+f'/figs/agora_cross_total_ratio.png',bbox_inches='tight')

    plt.clf()
    plt.axhline(y=1, color='gray', alpha=0.5, linestyle='-')

    plt.plot(bin_centers, cross_gmv_TTEETE[:,2]/binned_input_clkk, color='forestgreen', marker='o', linestyle='--', ms=3, alpha=0.8, label="Cross-ILC GMV (one component CIB)")
    plt.plot(bin_centers, cross_gmv_TTEETE[:,3]/binned_input_clkk, color='plum', marker='o', linestyle='--', ms=3, alpha=0.8, label="Cross-ILC GMV (two component CIB)")
    plt.plot(bin_centers, cross_gmv_TTEETE[:,1]/binned_input_clkk, color='goldenrod', marker='o', linestyle='--', ms=3, alpha=0.8, label="MH GMV")
    plt.plot(bin_centers, cross_gmv_TTEETE[:,4]/binned_input_clkk, color='darkorange', marker='o', linestyle='--', ms=3, alpha=0.8, label="Profile Hardened GMV")
    plt.plot(bin_centers, cross_sqe_TT[:,0]/binned_input_clkk, color='firebrick', marker='o', linestyle='--', ms=3, alpha=0.8, label=f'Standard SQE')
    plt.plot(bin_centers, cross_gmv_TTEETE[:,0]/binned_input_clkk, color='darkblue', marker='o', linestyle='--', ms=3, alpha=0.8, label="Standard GMV")

    plt.xlabel('$\ell$')
    plt.title(f'Agora Reconstruction x Input Kappa / Input Kappa (TTEETE only)')
    plt.legend(loc='upper left', fontsize='x-small', bbox_to_anchor=(1,0.5))
    plt.xscale('log')
    plt.xlim(10,lmax)
    plt.savefig(dir_out+f'/figs/agora_cross_TTEETE_ratio.png',bbox_inches='tight')

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
    filename = dir_out+f'/n0/n0_{num}simpairs_healqest_{qetype}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_resp_from_sims.pkl'

    if os.path.isfile(filename):
        n0 = pickle.load(open(filename,'rb'))
    else:
        print("File doesn't exist!")

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
    filename = dir_out+f'/n1/n1_{num}simpairs_healqest_{qetype}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_resp_from_sims.pkl'

    if os.path.isfile(filename):
        n1 = pickle.load(open(filename,'rb'))
    else:
        print("File doesn't exist!")

    return n1

def get_sim_response(est,config,gmv,append,sims,filename=None):
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
        if gmv:
            fn += f'_gmv_est{est}'
        else:
            fn += f'_sqe_est{est}'
        fn += f'_lmaxT{lmaxT}_lmaxP{lmaxP}_lmin{lmin}_cltype{cltype}_{append}'
        filename = dir_out+f'/resp/sim_resp{fn}.npy'

    if os.path.isfile(filename):
        print('Loading from existing file!')
        sim_resp = np.load(filename)
    else:
        print("File doesn't exist!")
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
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
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
