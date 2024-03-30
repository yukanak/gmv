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

def analyze(sims=np.arange(99)+1,n0_n1_sims=np.arange(98)+1,
            config_file='test_yuka.yaml',
            append='mh',
            n0=True,n1=True,
            lbins=np.logspace(np.log10(50),np.log10(3000),20)):
    '''
    Compare with N0/N1 subtraction.
    ALWAYS USING RESPONSE FROM SIMS. Analytic response not implemented here.
    '''
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

    # Get SQE response
    ests = ['T1T2', 'T2T1', 'EE', 'TE', 'ET', 'TB', 'BT', 'EB', 'BE']
    resps_original = np.zeros((len(l),len(ests)), dtype=np.complex_)
    inv_resps_original = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
    for i, est in enumerate(ests):
        resps_original[:,i] = get_sim_response(est,config,gmv=False,append=append,sims=sims)
        inv_resps_original[1:,i] = 1/(resps_original)[1:,i]
    resp_original = 0.5*resps_original[:,0]+0.5*resps_original[:,1]+np.sum(resps_original[:,2:], axis=1)
    inv_resp_original = np.zeros_like(l,dtype=np.complex_); inv_resp_original[1:] = 1/(resp_original)[1:]

    # GMV response
    resp_gmv = get_sim_response('all',config,gmv=True,append=append,sims=sims)
    resp_gmv_TTEETE = get_sim_response('TTEETE',config,gmv=True,append=append,sims=sims)
    resp_gmv_TBEB = get_sim_response('TBEB',config,gmv=True,append=append,sims=sims)
    inv_resp_gmv = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv[1:] = 1./(resp_gmv)[1:]
    inv_resp_gmv_TTEETE = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TTEETE[1:] = 1./(resp_gmv_TTEETE)[1:]
    inv_resp_gmv_TBEB = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TBEB[1:] = 1./(resp_gmv_TBEB)[1:]

    if n0:
        # Get N0 correction (remember these still need (l*(l+1))**2/4 factor to convert to kappa)
        n0_gmv = get_n0(sims=n0_n1_sims,qetype='gmv',config=config,
                        append=append)
        n0_gmv_total = n0_gmv['total'] * (l*(l+1))**2/4
        n0_gmv_TTEETE = n0_gmv['TTEETE'] * (l*(l+1))**2/4
        n0_gmv_TBEB = n0_gmv['TBEB'] * (l*(l+1))**2/4
        n0_original = get_n0(sims=n0_n1_sims,qetype='sqe',config=config,
                             append=append)
        n0_original_total = n0_original['total'] * (l*(l+1))**2/4
        n0_original_TT = n0_original['TT'] * (l*(l+1))**2/4
        n0_original_EE = n0_original['EE'] * (l*(l+1))**2/4
        n0_original_TE = n0_original['TE'] * (l*(l+1))**2/4
        n0_original_ET = n0_original['ET'] * (l*(l+1))**2/4
        n0_original_TB = n0_original['TB'] * (l*(l+1))**2/4
        n0_original_BT = n0_original['BT'] * (l*(l+1))**2/4
        n0_original_EB = n0_original['EB'] * (l*(l+1))**2/4
        n0_original_BE = n0_original['BE'] * (l*(l+1))**2/4

    if n1:
        n1_gmv = get_n1(sims=n0_n1_sims,qetype='gmv',config=config,
                        append=append)
        n1_gmv_total = n1_gmv['total'] * (l*(l+1))**2/4
        n1_gmv_TTEETE = n1_gmv['TTEETE'] * (l*(l+1))**2/4
        n1_gmv_TBEB = n1_gmv['TBEB'] * (l*(l+1))**2/4
        n1_original = get_n1(sims=n0_n1_sims,qetype='sqe',config=config,
                             append=append)
        n1_original_total = n1_original['total'] * (l*(l+1))**2/4
        n1_original_T1T2 = n1_original['T1T2'] * (l*(l+1))**2/4
        n1_original_EE = n1_original['EE'] * (l*(l+1))**2/4
        n1_original_TE = n1_original['TE'] * (l*(l+1))**2/4
        n1_original_ET = n1_original['ET'] * (l*(l+1))**2/4
        n1_original_TB = n1_original['TB'] * (l*(l+1))**2/4
        n1_original_BT = n1_original['BT'] * (l*(l+1))**2/4
        n1_original_EB = n1_original['EB'] * (l*(l+1))**2/4
        n1_original_BT = n1_original['BE'] * (l*(l+1))**2/4

    # Get mean field
    #mf_gmv = get_mf('gmv', config, sims=np.arange(20)+1, append=append)
    #mf_gmv_total = mf_gmv['total']
    #mf_sqe = get_mf('sqe', config, sims=np.arange(20)+1, append=append)
    #mf_sqe_total = mf_sqe['total']

    auto_gmv_all = 0
    auto_gmv_all_TTEETE = 0
    auto_gmv_all_TBEB = 0
    auto_original_all = 0
    auto_original_all_T1T2 = 0
    auto_original_all_T2T1 = 0
    auto_original_all_EE = 0
    auto_original_all_TE = 0
    auto_original_all_ET = 0
    auto_original_all_TB = 0
    auto_original_all_BT = 0
    auto_original_all_EB = 0
    auto_original_all_BE = 0
    cross_gmv_all = 0
    cross_original_all = 0
    auto_gmv_debiased_all = 0
    auto_original_debiased_all = 0
    auto_original_debiased_all_TT = 0
    ratio_gmv = np.zeros((len(sims),len(lbins)-1),dtype=np.complex_)
    ratio_original = np.zeros((len(sims),len(lbins)-1),dtype=np.complex_)
    ratio_original_TT = np.zeros((len(sims),len(lbins)-1),dtype=np.complex_)
    temp_gmv = np.zeros((len(sims),len(l)),dtype=np.complex_)
    temp_sqe = np.zeros((len(sims),len(l)),dtype=np.complex_)
    temp_gmv_TTEETE = np.zeros((len(sims),len(l)),dtype=np.complex_)
    temp_sqe_TT = np.zeros((len(sims),len(l)),dtype=np.complex_)
    binned_temp_gmv = np.zeros((len(sims),len(l)),dtype=np.complex_)
    binned_temp_sqe = np.zeros((len(sims),len(l)),dtype=np.complex_)
    binned_temp_gmv_TTEETE = np.zeros((len(sims),len(l)),dtype=np.complex_)
    binned_temp_sqe_TT = np.zeros((len(sims),len(l)),dtype=np.complex_)

    for ii, sim in enumerate(sims):
        # Load GMV plms
        plm_gmv = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
        plm_gmv_TTEETE = np.load(dir_out+f'/plm_TTEETE_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
        plm_gmv_TBEB = np.load(dir_out+f'/plm_TBEB_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')

        # Load SQE plms
        plms_original = np.zeros((len(np.load(dir_out+f'/plm_T1T2_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')),len(ests)), dtype=np.complex_)
        for i, est in enumerate(ests):
            plms_original[:,i] = np.load(dir_out+f'/plm_{est}_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
        plm_original = 0.5*plms_original[:,0]+0.5*plms_original[:,1]+np.sum(plms_original[:,2:], axis=1)

        # MF subtract
        #plm_gmv -= mf_gmv_total
        #plm_original -= mf_sqe_total

        # Response correct
        plm_gmv_resp_corr = hp.almxfl(plm_gmv,inv_resp_gmv)
        plm_gmv_resp_corr_TTEETE = hp.almxfl(plm_gmv_TTEETE,inv_resp_gmv_TTEETE)
        plm_gmv_resp_corr_TBEB = hp.almxfl(plm_gmv_TBEB,inv_resp_gmv_TBEB)
        plm_original_resp_corr = hp.almxfl(plm_original,inv_resp_original)
        plm_original_resp_corr_T1T2 = hp.almxfl(plms_original[:,0],inv_resps_original[:,0])
        plm_original_resp_corr_T2T1 = hp.almxfl(plms_original[:,1],inv_resps_original[:,1])
        plm_original_resp_corr_EE = hp.almxfl(plms_original[:,2],inv_resps_original[:,2])
        plm_original_resp_corr_TE = hp.almxfl(plms_original[:,3],inv_resps_original[:,3])
        plm_original_resp_corr_ET = hp.almxfl(plms_original[:,4],inv_resps_original[:,4])
        plm_original_resp_corr_TB = hp.almxfl(plms_original[:,5],inv_resps_original[:,5])
        plm_original_resp_corr_BT = hp.almxfl(plms_original[:,6],inv_resps_original[:,6])
        plm_original_resp_corr_EB = hp.almxfl(plms_original[:,7],inv_resps_original[:,7])
        plm_original_resp_corr_BE = hp.almxfl(plms_original[:,8],inv_resps_original[:,8])

        # Get spectra
        auto_gmv = hp.alm2cl(plm_gmv_resp_corr, plm_gmv_resp_corr, lmax=lmax) * (l*(l+1))**2/4
        auto_gmv_TTEETE = hp.alm2cl(plm_gmv_resp_corr_TTEETE, plm_gmv_resp_corr_TTEETE, lmax=lmax) * (l*(l+1))**2/4
        auto_gmv_TBEB = hp.alm2cl(plm_gmv_resp_corr_TBEB, plm_gmv_resp_corr_TBEB, lmax=lmax) * (l*(l+1))**2/4
        auto_original = hp.alm2cl(plm_original_resp_corr, plm_original_resp_corr, lmax=lmax) * (l*(l+1))**2/4
        auto_original_T1T2 = hp.alm2cl(plm_original_resp_corr_T1T2, plm_original_resp_corr_T1T2, lmax=lmax) * (l*(l+1))**2/4
        auto_original_T2T1 = hp.alm2cl(plm_original_resp_corr_T2T1, plm_original_resp_corr_T2T1, lmax=lmax) * (l*(l+1))**2/4
        auto_original_EE = hp.alm2cl(plm_original_resp_corr_EE, plm_original_resp_corr_EE, lmax=lmax) * (l*(l+1))**2/4
        auto_original_TE = hp.alm2cl(plm_original_resp_corr_TE, plm_original_resp_corr_TE, lmax=lmax) * (l*(l+1))**2/4
        auto_original_ET = hp.alm2cl(plm_original_resp_corr_ET, plm_original_resp_corr_ET, lmax=lmax) * (l*(l+1))**2/4
        auto_original_TB = hp.alm2cl(plm_original_resp_corr_TB, plm_original_resp_corr_TB, lmax=lmax) * (l*(l+1))**2/4
        auto_original_BT = hp.alm2cl(plm_original_resp_corr_BT, plm_original_resp_corr_BT, lmax=lmax) * (l*(l+1))**2/4
        auto_original_EB = hp.alm2cl(plm_original_resp_corr_EB, plm_original_resp_corr_EB, lmax=lmax) * (l*(l+1))**2/4
        auto_original_BE = hp.alm2cl(plm_original_resp_corr_BE, plm_original_resp_corr_BE, lmax=lmax) * (l*(l+1))**2/4

        # N0 and N1 subtract
        if n0 and n1:
            auto_gmv_debiased = auto_gmv - n0_gmv_total - n1_gmv_total
            auto_original_debiased = auto_original - n0_original_total - n1_original_total
            auto_gmv_debiased_TTEETE = auto_gmv_TTEETE - n0_gmv_TTEETE - n1_gmv_TTEETE
            auto_original_debiased_TT = (auto_original_T1T2+auto_original_T2T1)*0.5 - n0_original_TT - n1_original_T1T2
        elif n0:
            auto_gmv_debiased = auto_gmv - n0_gmv_total
            auto_original_debiased = auto_original - n0_original_total

        # Sum the auto spectra
        auto_gmv_all += auto_gmv
        auto_gmv_all_TTEETE += auto_gmv_TTEETE
        auto_gmv_all_TBEB += auto_gmv_TBEB
        auto_original_all += auto_original
        auto_original_all_T1T2 += auto_original_T1T2
        auto_original_all_T2T1 += auto_original_T2T1
        auto_original_all_EE += auto_original_EE
        auto_original_all_TE += auto_original_TE
        auto_original_all_ET += auto_original_ET
        auto_original_all_TB += auto_original_TB
        auto_original_all_BT += auto_original_BT
        auto_original_all_EB += auto_original_EB
        auto_original_all_BE += auto_original_BE
        if n0:
            auto_gmv_debiased_all += auto_gmv_debiased
            auto_original_debiased_all += auto_original_debiased
            # Need this to compute uncertainty...
            temp_gmv[ii,:] = auto_gmv_debiased
            temp_sqe[ii,:] = auto_original_debiased
            temp_gmv_TTEETE[ii,:] = auto_gmv_debiased_TTEETE
            temp_sqe_TT[ii,:] = auto_original_debiased_TT
            binned_temp_gmv = [auto_gmv_debiased[digitized == i].mean() for i in range(1, len(lbins))]
            binned_temp_sqe = [auto_original_debiased[digitized == i].mean() for i in range(1, len(lbins))]
            binned_temp_gmv_TTEETE = [auto_gmv_debiased_TTEETE[digitized == i].mean() for i in range(1, len(lbins))]
            binned_temp_sqe_TT = [auto_original_debiased_TT[digitized == i].mean() for i in range(1, len(lbins))]

        # If debiasing, get the binned ratio against input
        if n0:
            input_plm = hp.read_alm(f'/oak/stanford/orgs/kipac//users/yukanaka/lensing19-20/inputcmb/phi/phi_lmax_{lmax}/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_phi1_seed{sim}_lmax{lmax}.alm')
            auto_input = hp.alm2cl(input_plm, input_plm, lmax=lmax) * (l*(l+1))**2/4
            # Bin!
            binned_auto_gmv_debiased = [auto_gmv_debiased[digitized == i].mean() for i in range(1, len(lbins))]
            binned_auto_original_debiased = [auto_original_debiased[digitized == i].mean() for i in range(1, len(lbins))]
            #binned_auto_original_debiased_TT = [auto_original_debiased_TT[digitized == i].mean() for i in range(1, len(lbins))]
            binned_auto_input = [auto_input[digitized == i].mean() for i in range(1, len(lbins))]
            # Get ratio
            ratio_gmv[ii,:] = np.array(binned_auto_gmv_debiased) / np.array(binned_auto_input)
            ratio_original[ii,:] = np.array(binned_auto_original_debiased) / np.array(binned_auto_input)
            #ratio_original_TT[ii,:] = np.array(binned_auto_original_debiased_TT) / np.array(binned_auto_input)

    # GET THE UNCERTAINTIES (error bar from spread of sims - measurement uncertainty of bandpowers)
    uncertainty = np.zeros((len(l),4), dtype=np.complex_)
    binned_uncertainty = np.zeros((len(bin_centers),4), dtype=np.complex_)
    uncertainty[:,0] = np.std(temp_gmv,axis=0)
    uncertainty[:,1] = np.std(temp_sqe,axis=0)
    uncertainty[:,2] = np.std(temp_gmv_TTEETE,axis=0)
    uncertainty[:,3] = np.std(temp_sqe_TT,axis=0)
    binned_uncertainty[:,0] = np.std(binned_temp_gmv,axis=0)
    binned_uncertainty[:,1] = np.std(binned_temp_sqe,axis=0)
    binned_uncertainty[:,2] = np.std(binned_temp_gmv_TTEETE,axis=0)
    binned_uncertainty[:,3] = np.std(binned_temp_sqe_TT,axis=0)
    np.save(dir_out+f'/agora_reconstruction/measurement_uncertainty_lmaxT{lmaxT}_{append}.npy',uncertainty)
    np.save(dir_out+f'/agora_reconstruction/binned_measurement_uncertainty_lmaxT{lmaxT}_{append}.npy',binned_uncertainty)

    # Average
    auto_gmv_avg = auto_gmv_all / num
    auto_gmv_avg_TTEETE = auto_gmv_all_TTEETE / num
    auto_gmv_avg_TBEB = auto_gmv_all_TBEB / num
    auto_original_avg = auto_original_all / num
    auto_original_avg_T1T2 = auto_original_all_T1T2 / num
    auto_original_avg_T2T1 = auto_original_all_T2T1 / num
    auto_original_avg_EE = auto_original_all_EE / num
    auto_original_avg_TE = auto_original_all_TE / num
    auto_original_avg_ET = auto_original_all_ET / num
    auto_original_avg_TB = auto_original_all_TB / num
    auto_original_avg_BT = auto_original_all_BT / num
    auto_original_avg_EB = auto_original_all_EB / num
    auto_original_avg_BE = auto_original_all_BE / num
    if n0:
        auto_gmv_debiased_avg = auto_gmv_debiased_all / num
        auto_original_debiased_avg = auto_original_debiased_all / num
        #auto_original_debiased_avg_TT = auto_original_debiased_all_TT / num
        # If debiasing, get the ratio points, error bars for the ratio points, and bin
        errorbars_gmv = np.std(ratio_gmv,axis=0)/np.sqrt(num)
        errorbars_original = np.std(ratio_original,axis=0)/np.sqrt(num)
        #errorbars_original_TT = np.std(ratio_original_TT,axis=0)/np.sqrt(num)
        ratio_gmv = np.mean(ratio_gmv,axis=0)
        ratio_original = np.mean(ratio_original,axis=0)
        #ratio_original_TT = np.mean(ratio_original_TT,axis=0)
        # Bin!
        binned_auto_gmv_debiased_avg = [auto_gmv_debiased_avg[digitized == i].mean() for i in range(1, len(lbins))]
        binned_auto_original_debiased_avg = [auto_original_debiased_avg[digitized == i].mean() for i in range(1, len(lbins))]
        #binned_auto_original_debiased_avg_TT = [auto_original_debiased_avg_TT[digitized == i].mean() for i in range(1, len(lbins))]

    # Theory spectrum
    clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
    ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
    clkk = slpp * (l*(l+1))**2/4

    # Plot
    plt.figure(0)
    plt.clf()
    plt.plot(l, auto_gmv_debiased_avg, color='cornflowerblue', linestyle='-', label="Auto Spectrum (GMV)")
    plt.plot(l, auto_original_debiased_avg, color='lightcoral', linestyle='-', label=f'Auto Spectrum (SQE)')

    plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')

    plt.plot(bin_centers, binned_auto_gmv_debiased_avg, color='darkblue', marker='o', linestyle='None', ms=3, label="Auto Spectrum (GMV)")
    plt.plot(bin_centers, binned_auto_original_debiased_avg, color='firebrick', marker='o', linestyle='None', ms=3, label="Auto Spectrum (SQE)")

    plt.ylabel("$C_\ell^{\kappa\kappa}$")
    plt.xlabel('$\ell$')
    plt.title(f'Spectra Averaged over {num} Sims, MH')
    plt.legend(loc='upper left', fontsize='x-small', bbox_to_anchor=(1,0.5))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,lmax)
    plt.ylim(1e-9,1e-6)
    plt.savefig(dir_out+f'/figs/{num}_sims_comparison_{append}_resp_from_sims_n0n1subtracted_lmaxT{lmaxT}.png',bbox_inches='tight')

    plt.figure(1)
    plt.clf()
    # Ratios with error bars
    plt.axhline(y=1, color='k', linestyle='--')
    plt.errorbar(bin_centers,ratio_gmv,yerr=errorbars_gmv,color='darkblue', marker='o', linestyle='None', ms=3, label="Ratio GMV/Input")
    plt.errorbar(bin_centers,ratio_original,yerr=errorbars_original,color='firebrick', marker='o', linestyle='None', ms=3, label="Ratio Original/Input")
    plt.xlabel('$\ell$')
    plt.title(f'Spectra Averaged over {num} Sims, MH')
    plt.legend(loc='lower left', fontsize='x-small')
    plt.xscale('log')
    plt.ylim(0.98,1.02)
    plt.xlim(10,lmax)
    if n1:
        plt.savefig(dir_out+f'/figs/{num}_sims_comparison_{append}_resp_from_sims_n0n1subtracted_binnedratio_lmaxT{lmaxT}.png',bbox_inches='tight')
    else:
        plt.savefig(dir_out+f'/figs/{num}_sims_comparison_{append}_resp_from_sims_n0subtracted_binnedratio.png',bbox_inches='tight')


def get_mf(qetype, config, sims=np.arange(20)+1, append='mh'):
    '''
    Get the mean field.
    qetype should be 'gmv' or 'sqe'.
    est should be 'all', 'TTEETE', 'TBEB', 'T1T2', 'T2T1', 'EE', 'TE', etc.
    '''
    lmaxT = config['lensrec']['lmaxT']
    lmaxP = config['lensrec']['lmaxP']
    nside = config['lensrec']['nside']
    dir_out = config['dir_out']
    num = len(sims)
    filename = dir_out+f'/mf/mf_{num}simpairs_{qetype}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.pkl'

    if os.path.isfile(filename):
        mf = pickle.load(open(filename,'rb'))
    elif qetype == 'gmv':
        qetype = '_gmv'
        mf = {'total':0, 'TTEETE':0, 'TBEB':0}
        for i, sim in enumerate(sims):
            # Load plms
            mf['total'] += np.load(dir_out+f'/plm_all_healqest{qetype}_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            mf['TTEETE'] += np.load(dir_out+f'/plm_TTEETE_healqest{qetype}_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            mf['TBEB'] += np.load(dir_out+f'/plm_TBEB_healqest{qetype}_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
        mf['total'] /= num
        mf['TTEETE'] /= num
        mf['TBEB'] /= num

        with open(filename, 'wb') as f:
            pickle.dump(mf, f)

    elif qetype == 'sqe':
        qetype = ''
        mf = {'T1T2':0, 'T2T1':0, 'EE':0, 'TE':0, 'ET':0, 'TB':0, 'BT':0, 'EB':0, 'BE':0}
        mf_total = 0
        for i, sim in enumerate(sims):
            # Load plms
            for key in mf:
                if key == 'T1T2' or key == 'T2T1':
                    nrm = 0.5
                else:
                    nrm = 1
                plm = np.load(dir_out+f'/plm_{key}_healqest{qetype}_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
                mf[key] += plm
                mf_total += nrm*plm
        mf['total'] = mf_total
        for key in mf:
            mf[key] /= num

        with open(filename, 'wb') as f:
            pickle.dump(mf, f)

    return mf

def get_n0(sims, qetype, config,append,cmbonly=False):
    '''
    Get N0 bias. qetype should be 'gmv' or 'sqe'.
    Returns dictionary containing keys 'total', 'TTEETE', and 'TBEB' for GMV.
    Similarly for SQE.
    '''
    lmax = config['lensrec']['Lmax']
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

    elif qetype == 'gmv':
        # GMV response
        resp_gmv = get_sim_response('all',config,gmv=True,append=append_original,sims=np.append(sims,num+1))
        resp_gmv_TTEETE = get_sim_response('TTEETE',config,gmv=True,append=append_original,sims=np.append(sims,num+1))
        resp_gmv_TBEB = get_sim_response('TBEB',config,gmv=True,append=append_original,sims=np.append(sims,num+1))
        inv_resp_gmv = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv[1:] = 1./(resp_gmv)[1:]
        inv_resp_gmv_TTEETE = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TTEETE[1:] = 1./(resp_gmv_TTEETE)[1:]
        inv_resp_gmv_TBEB = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TBEB[1:] = 1./(resp_gmv_TBEB)[1:]

        n0 = {'total':0, 'TTEETE':0, 'TBEB':0}
        for i, sim1 in enumerate(sims):
            sim2 = sim1 + 1

            # Get the lensed ij sims
            plm_gmv_ij = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            plm_gmv_TTEETE_ij = np.load(dir_out+f'/plm_TTEETE_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            plm_gmv_TBEB_ij = np.load(dir_out+f'/plm_TBEB_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')

            # Now get the ji sims
            plm_gmv_ji = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            plm_gmv_TTEETE_ji = np.load(dir_out+f'/plm_TTEETE_healqest_gmv_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            plm_gmv_TBEB_ji = np.load(dir_out+f'/plm_TBEB_healqest_gmv_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')

            # Response correct
            plm_gmv_resp_corr_ij = hp.almxfl(plm_gmv_ij,inv_resp_gmv)
            plm_gmv_resp_corr_TTEETE_ij = hp.almxfl(plm_gmv_TTEETE_ij,inv_resp_gmv_TTEETE)
            plm_gmv_resp_corr_TBEB_ij = hp.almxfl(plm_gmv_TBEB_ij,inv_resp_gmv_TBEB)

            plm_gmv_resp_corr_ji = hp.almxfl(plm_gmv_ji,inv_resp_gmv)
            plm_gmv_resp_corr_TTEETE_ji = hp.almxfl(plm_gmv_TTEETE_ji,inv_resp_gmv_TTEETE)
            plm_gmv_resp_corr_TBEB_ji = hp.almxfl(plm_gmv_TBEB_ji,inv_resp_gmv_TBEB)

            # Get ij auto spectra <ijij>
            auto = hp.alm2cl(plm_gmv_resp_corr_ij, plm_gmv_resp_corr_ij, lmax=lmax)
            auto_A = hp.alm2cl(plm_gmv_resp_corr_TTEETE_ij, plm_gmv_resp_corr_TTEETE_ij, lmax=lmax)
            auto_B = hp.alm2cl(plm_gmv_resp_corr_TBEB_ij, plm_gmv_resp_corr_TBEB_ij, lmax=lmax)

            # Get cross spectra <ijji>
            cross = hp.alm2cl(plm_gmv_resp_corr_ij, plm_gmv_resp_corr_ji, lmax=lmax)
            cross_A = hp.alm2cl(plm_gmv_resp_corr_TTEETE_ij, plm_gmv_resp_corr_TTEETE_ji, lmax=lmax)
            cross_B = hp.alm2cl(plm_gmv_resp_corr_TBEB_ij, plm_gmv_resp_corr_TBEB_ji, lmax=lmax)

            n0['total'] += auto + cross
            n0['TTEETE'] += auto_A + cross_A
            n0['TBEB'] += auto_B + cross_B

        n0['total'] *= 1/num
        n0['TTEETE'] *= 1/num
        n0['TBEB'] *= 1/num

        with open(filename, 'wb') as f:
            pickle.dump(n0, f)

    elif qetype == 'sqe':
        # SQE response
        ests = ['T1T2', 'EE', 'TE', 'ET', 'TB', 'BT', 'EB', 'BE']
        resps_original = np.zeros((len(l),len(ests)), dtype=np.complex_)
        inv_resps_original = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
        for i, est in enumerate(ests):
            resps_original[:,i] = get_sim_response(est,config,gmv=False,append=append_original,sims=np.append(sims,num+1))
            inv_resps_original[1:,i] = 1/(resps_original)[1:,i]
        resp_original = np.sum(resps_original, axis=1)
        resp_original_TTEETE = resps_original[:,0]+resps_original[:,1]+resps_original[:,2]+resps_original[:,3]
        resp_original_TBEB = resps_original[:,4]+resps_original[:,5]+resps_original[:,6]+resps_original[:,7]
        inv_resp_original = np.zeros_like(l,dtype=np.complex_); inv_resp_original[1:] = 1/(resp_original)[1:]
        inv_resp_original_TTEETE = np.zeros_like(l,dtype=np.complex_); inv_resp_original_TTEETE[1:] = 1/(resp_original_TTEETE)[1:]
        inv_resp_original_TBEB = np.zeros_like(l,dtype=np.complex_); inv_resp_original_TBEB[1:] = 1/(resp_original_TBEB)[1:]

        n0 = {'total':0, 'TT':0, 'EE':0, 'TE':0, 'ET':0, 'TB':0, 'BT':0, 'EB':0, 'BE':0, 'TTEETE':0, 'TBEB':0}
        for i, sim1 in enumerate(sims):
            sim2 = sim1 + 1

            # Get the lensed ij sims
            plms_ij = np.zeros((len(np.load(dir_out+f'/plm_T1T2_healqest_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')),len(ests)), dtype=np.complex_)
            for i, est in enumerate(ests):
                plms_ij[:,i] = np.load(dir_out+f'/plm_{est}_healqest_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            plm_total_ij = np.sum(plms_ij, axis=1)

            # Now get the ji sims
            plms_ji = np.zeros((len(np.load(dir_out+f'/plm_T1T2_healqest_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')),len(ests)), dtype=np.complex_)
            for i, est in enumerate(ests):
                plms_ji[:,i] = np.load(dir_out+f'/plm_{est}_healqest_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            plm_total_ji = np.sum(plms_ji, axis=1)

            # EIGHT estimators!!!
            plm_TTEETE_ij = plms_ij[:,0]+plms_ij[:,1]+plms_ij[:,2]+plms_ij[:,3]
            plm_TTEETE_ji = plms_ji[:,0]+plms_ji[:,1]+plms_ji[:,2]+plms_ji[:,3]
            plm_TBEB_ij = plms_ij[:,4]+plms_ij[:,5]+plms_ij[:,6]+plms_ij[:,7]
            plm_TBEB_ji = plms_ji[:,4]+plms_ji[:,5]+plms_ji[:,6]+plms_ji[:,7]

            # Response correct healqest
            plm_total_ij = hp.almxfl(plm_total_ij,inv_resp_original)
            plm_TTEETE_ij = hp.almxfl(plm_TTEETE_ij,inv_resp_original_TTEETE)
            plm_TBEB_ij = hp.almxfl(plm_TBEB_ij,inv_resp_original_TBEB)
            for i, est in enumerate(ests):
                plms_ij[:,i] = hp.almxfl(plms_ij[:,i],inv_resps_original[:,i])

            plm_total_ji = hp.almxfl(plm_total_ji,inv_resp_original)
            plm_TTEETE_ji = hp.almxfl(plm_TTEETE_ji,inv_resp_original_TTEETE)
            plm_TBEB_ji = hp.almxfl(plm_TBEB_ji,inv_resp_original_TBEB)
            for i, est in enumerate(ests):
                plms_ji[:,i] = hp.almxfl(plms_ji[:,i],inv_resps_original[:,i])

            # Get ij auto spectra <ijij>
            auto = hp.alm2cl(plm_total_ij, plm_total_ij, lmax=lmax)
            auto_TTEETE = hp.alm2cl(plm_TTEETE_ij, plm_TTEETE_ij, lmax=lmax)
            auto_TBEB = hp.alm2cl(plm_TBEB_ij, plm_TBEB_ij, lmax=lmax)
            auto_TT = hp.alm2cl(plms_ij[:,0], plms_ij[:,0], lmax=lmax)
            auto_EE = hp.alm2cl(plms_ij[:,1], plms_ij[:,1], lmax=lmax)
            auto_TE = hp.alm2cl(plms_ij[:,2], plms_ij[:,2], lmax=lmax)
            auto_ET = hp.alm2cl(plms_ij[:,3], plms_ij[:,3], lmax=lmax)
            auto_TB = hp.alm2cl(plms_ij[:,4], plms_ij[:,4], lmax=lmax)
            auto_BT = hp.alm2cl(plms_ij[:,5], plms_ij[:,5], lmax=lmax)
            auto_EB = hp.alm2cl(plms_ij[:,6], plms_ij[:,6], lmax=lmax)
            auto_BE = hp.alm2cl(plms_ij[:,7], plms_ij[:,7], lmax=lmax)

            # Get cross spectra <ijji>
            cross = hp.alm2cl(plm_total_ij, plm_total_ji, lmax=lmax)
            cross_TTEETE = hp.alm2cl(plm_TTEETE_ij, plm_TTEETE_ji, lmax=lmax)
            cross_TBEB = hp.alm2cl(plm_TBEB_ij, plm_TBEB_ji, lmax=lmax)
            cross_TT = hp.alm2cl(plms_ij[:,0], plms_ji[:,0], lmax=lmax)
            cross_EE = hp.alm2cl(plms_ij[:,1], plms_ji[:,1], lmax=lmax)
            cross_TE = hp.alm2cl(plms_ij[:,2], plms_ji[:,2], lmax=lmax)
            cross_ET = hp.alm2cl(plms_ij[:,3], plms_ji[:,3], lmax=lmax)
            cross_TB = hp.alm2cl(plms_ij[:,4], plms_ji[:,4], lmax=lmax)
            cross_BT = hp.alm2cl(plms_ij[:,5], plms_ji[:,5], lmax=lmax)
            cross_EB = hp.alm2cl(plms_ij[:,6], plms_ji[:,6], lmax=lmax)
            cross_BE = hp.alm2cl(plms_ij[:,7], plms_ji[:,7], lmax=lmax)

            n0['total'] += auto + cross
            n0['TTEETE'] += auto_TTEETE + cross_TTEETE
            n0['TBEB'] += auto_TBEB + cross_TBEB
            n0['TT'] += auto_TT + cross_TT
            n0['EE'] += auto_EE + cross_EE
            n0['TE'] += auto_TE + cross_TE
            n0['ET'] += auto_ET + cross_ET
            n0['TB'] += auto_TB + cross_TB
            n0['BT'] += auto_BT + cross_BT
            n0['EB'] += auto_EB + cross_EB
            n0['BE'] += auto_BE + cross_BE

        n0['total'] *= 1/num
        n0['TTEETE'] *= 1/num
        n0['TBEB'] *= 1/num
        n0['TT'] *= 1/num
        n0['EE'] *= 1/num
        n0['TE'] *= 1/num
        n0['ET'] *= 1/num
        n0['TB'] *= 1/num
        n0['BT'] *= 1/num
        n0['EB'] *= 1/num
        n0['BE'] *= 1/num

        with open(filename, 'wb') as f:
            pickle.dump(n0, f)

    else:
        print('Invalid argument qetype.')

    return n0

def get_n1(sims, qetype, config, append='mh'):
    '''
    Get N1 bias. qetype should be 'gmv' or 'sqe'.
    Returns dictionary containing keys 'total', 'TTEETE', and 'TBEB' for GMV.
    Similarly for SQE.
    '''
    lmax = config['lensrec']['Lmax']
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

    elif qetype == 'gmv':
        # GMV response
        resp_gmv = get_sim_response('all',config,gmv=True,append=append,sims=np.append(sims,num+1))
        resp_gmv_TTEETE = get_sim_response('TTEETE',config,gmv=True,append=append,sims=np.append(sims,num+1))
        resp_gmv_TBEB = get_sim_response('TBEB',config,gmv=True,append=append,sims=np.append(sims,num+1))
        inv_resp_gmv = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv[1:] = 1./(resp_gmv)[1:]
        inv_resp_gmv_TTEETE = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TTEETE[1:] = 1./(resp_gmv_TTEETE)[1:]
        inv_resp_gmv_TBEB = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TBEB[1:] = 1./(resp_gmv_TBEB)[1:]

        n1 = {'total':0, 'TTEETE':0, 'TBEB':0}
        for i, sim in enumerate(sims):
            # These are reconstructions using sims that were lensed with the same phi but different CMB realizations, no foregrounds
            # Get the lensed ij sims
            plm_gmv_ij = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu1tqu2.npy')
            plm_gmv_TTEETE_ij = np.load(dir_out+f'/plm_TTEETE_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu1tqu2.npy')
            plm_gmv_TBEB_ij = np.load(dir_out+f'/plm_TBEB_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu1tqu2.npy')

            # Now get the ji sims
            plm_gmv_ji = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu2tqu1.npy')
            plm_gmv_TTEETE_ji = np.load(dir_out+f'/plm_TTEETE_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu2tqu1.npy')
            plm_gmv_TBEB_ji = np.load(dir_out+f'/plm_TBEB_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu2tqu1.npy')

            # Response correct
            plm_gmv_resp_corr_ij = hp.almxfl(plm_gmv_ij,inv_resp_gmv)
            plm_gmv_resp_corr_TTEETE_ij = hp.almxfl(plm_gmv_TTEETE_ij,inv_resp_gmv_TTEETE)
            plm_gmv_resp_corr_TBEB_ij = hp.almxfl(plm_gmv_TBEB_ij,inv_resp_gmv_TBEB)

            plm_gmv_resp_corr_ji = hp.almxfl(plm_gmv_ji,inv_resp_gmv)
            plm_gmv_resp_corr_TTEETE_ji = hp.almxfl(plm_gmv_TTEETE_ji,inv_resp_gmv_TTEETE)
            plm_gmv_resp_corr_TBEB_ji = hp.almxfl(plm_gmv_TBEB_ji,inv_resp_gmv_TBEB)

            # Get ij auto spectra <ijij>
            auto = hp.alm2cl(plm_gmv_resp_corr_ij, plm_gmv_resp_corr_ij, lmax=lmax)
            auto_A = hp.alm2cl(plm_gmv_resp_corr_TTEETE_ij, plm_gmv_resp_corr_TTEETE_ij, lmax=lmax)
            auto_B = hp.alm2cl(plm_gmv_resp_corr_TBEB_ij, plm_gmv_resp_corr_TBEB_ij, lmax=lmax)

            # Get cross spectra <ijji>
            cross = hp.alm2cl(plm_gmv_resp_corr_ij, plm_gmv_resp_corr_ji, lmax=lmax)
            cross_A = hp.alm2cl(plm_gmv_resp_corr_TTEETE_ij, plm_gmv_resp_corr_TTEETE_ji, lmax=lmax)
            cross_B = hp.alm2cl(plm_gmv_resp_corr_TBEB_ij, plm_gmv_resp_corr_TBEB_ji, lmax=lmax)

            n1['total'] += auto + cross
            n1['TTEETE'] += auto_A + cross_A
            n1['TBEB'] += auto_B + cross_B

        n1['total'] *= 1/num
        n1['TTEETE'] *= 1/num
        n1['TBEB'] *= 1/num

        n0 = get_n0(sims=sims,qetype=qetype,config=config,
                    cmbonly=True,append=append)

        n1['total'] -= n0['total']
        n1['TTEETE'] -= n0['TTEETE']
        n1['TBEB'] -= n0['TBEB']

        with open(filename, 'wb') as f:
            pickle.dump(n1, f)

    elif qetype == 'sqe':
        # Get SQE response
        ests = ['T1T2','T2T1', 'EE', 'TE', 'ET', 'TB', 'BT', 'EB', 'BE']
        resps_original = np.zeros((len(l),len(ests)), dtype=np.complex_)
        inv_resps_original = np.zeros((len(l),len(ests)),dtype=np.complex_)
        for i, est in enumerate(ests):
            resps_original[:,i] = get_sim_response(est,config,gmv=False,append=append,sims=np.append(sims,num+1))
            inv_resps_original[1:,i] = 1/(resps_original)[1:,i]
        resp_original = 0.5*resps_original[:,0]+0.5*resps_original[:,1]+np.sum(resps_original[:,2:], axis=1)
        inv_resp_original = np.zeros_like(l,dtype=np.complex_); inv_resp_original[1:] = 1/(resp_original)[1:]

        n1 = {'total':0, 'T1T2':0, 'T2T1':0, 'EE':0, 'TE':0, 'ET':0, 'TB':0, 'BT':0, 'EB':0, 'BE':0}
        for i, sim in enumerate(sims):
            # Get the lensed ij sims
            plm_T1T2_ij = np.load(dir_out+f'/plm_T1T2_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu1tqu2.npy')
            plm_T2T1_ij = np.load(dir_out+f'/plm_T2T1_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu1tqu2.npy')
            plm_EE_ij = np.load(dir_out+f'/plm_EE_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu1tqu2.npy')
            plm_TE_ij = np.load(dir_out+f'/plm_TE_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu1tqu2.npy')
            plm_ET_ij = np.load(dir_out+f'/plm_ET_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu1tqu2.npy')
            plm_TB_ij = np.load(dir_out+f'/plm_TB_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu1tqu2.npy')
            plm_BT_ij = np.load(dir_out+f'/plm_BT_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu1tqu2.npy')
            plm_EB_ij = np.load(dir_out+f'/plm_EB_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu1tqu2.npy')
            plm_BE_ij = np.load(dir_out+f'/plm_BE_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu1tqu2.npy')

            # Now get the ji sims
            plm_T1T2_ji = np.load(dir_out+f'/plm_T1T2_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu2tqu1.npy')
            plm_T2T1_ji = np.load(dir_out+f'/plm_T2T1_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu2tqu1.npy')
            plm_EE_ji = np.load(dir_out+f'/plm_EE_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu2tqu1.npy')
            plm_TE_ji = np.load(dir_out+f'/plm_TE_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu2tqu1.npy')
            plm_ET_ji = np.load(dir_out+f'/plm_ET_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu2tqu1.npy')
            plm_TB_ji = np.load(dir_out+f'/plm_TB_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu2tqu1.npy')
            plm_BT_ji = np.load(dir_out+f'/plm_BT_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu2tqu1.npy')
            plm_EB_ji = np.load(dir_out+f'/plm_EB_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu2tqu1.npy')
            plm_BE_ji = np.load(dir_out+f'/plm_BE_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu2tqu1.npy')

            # NINE estimators!!!
            plm_total_ij = 0.5*plm_T1T2_ij + 0.5*plm_T2T1_ij + plm_EE_ij + plm_TE_ij + plm_ET_ij + plm_TB_ij + plm_BT_ij + plm_EB_ij + plm_BE_ij
            plm_total_ji = 0.5*plm_T1T2_ji + 0.5*plm_T2T1_ji + plm_EE_ji + plm_TE_ji + plm_ET_ji + plm_TB_ji + plm_BT_ji + plm_EB_ji + plm_BE_ji

            # Response correct healqest
            plm_total_ij = hp.almxfl(plm_total_ij,inv_resp_original)
            plm_T1T2_ij = hp.almxfl(plm_T1T2_ij,inv_resps_original[:,0])
            plm_T2T1_ij = hp.almxfl(plm_T2T1_ij,inv_resps_original[:,1])
            plm_EE_ij = hp.almxfl(plm_EE_ij,inv_resps_original[:,2])
            plm_TE_ij = hp.almxfl(plm_TE_ij,inv_resps_original[:,3])
            plm_ET_ij = hp.almxfl(plm_ET_ij,inv_resps_original[:,4])
            plm_TB_ij = hp.almxfl(plm_TB_ij,inv_resps_original[:,5])
            plm_BT_ij = hp.almxfl(plm_BT_ij,inv_resps_original[:,6])
            plm_EB_ij = hp.almxfl(plm_EB_ij,inv_resps_original[:,7])
            plm_BE_ij = hp.almxfl(plm_BE_ij,inv_resps_original[:,8])

            plm_total_ji = hp.almxfl(plm_total_ji,inv_resp_original)
            plm_T1T2_ji = hp.almxfl(plm_T1T2_ji,inv_resps_original[:,0])
            plm_T2T1_ji = hp.almxfl(plm_T2T1_ji,inv_resps_original[:,1])
            plm_EE_ji = hp.almxfl(plm_EE_ji,inv_resps_original[:,2])
            plm_TE_ji = hp.almxfl(plm_TE_ji,inv_resps_original[:,3])
            plm_ET_ji = hp.almxfl(plm_ET_ji,inv_resps_original[:,4])
            plm_TB_ji = hp.almxfl(plm_TB_ji,inv_resps_original[:,5])
            plm_BT_ji = hp.almxfl(plm_BT_ji,inv_resps_original[:,6])
            plm_EB_ji = hp.almxfl(plm_EB_ji,inv_resps_original[:,7])
            plm_BE_ji = hp.almxfl(plm_BE_ji,inv_resps_original[:,8])

            # Get ij auto spectra <ijij>
            auto = hp.alm2cl(plm_total_ij, plm_total_ij, lmax=lmax)
            auto_T1T2 = hp.alm2cl(plm_T1T2_ij, plm_T1T2_ij, lmax=lmax)
            auto_T2T1 = hp.alm2cl(plm_T2T1_ij, plm_T2T1_ij, lmax=lmax)
            auto_EE = hp.alm2cl(plm_EE_ij, plm_EE_ij, lmax=lmax)
            auto_TE = hp.alm2cl(plm_TE_ij, plm_TE_ij, lmax=lmax)
            auto_ET = hp.alm2cl(plm_ET_ij, plm_ET_ij, lmax=lmax)
            auto_TB = hp.alm2cl(plm_TB_ij, plm_TB_ij, lmax=lmax)
            auto_BT = hp.alm2cl(plm_BT_ij, plm_BT_ij, lmax=lmax)
            auto_EB = hp.alm2cl(plm_EB_ij, plm_EB_ij, lmax=lmax)
            auto_BE = hp.alm2cl(plm_BE_ij, plm_BE_ij, lmax=lmax)

            # Get cross spectra <ijji>
            cross = hp.alm2cl(plm_total_ij, plm_total_ji, lmax=lmax)
            cross_T1T2 = hp.alm2cl(plm_T1T2_ij, plm_T1T2_ji, lmax=lmax)
            cross_T2T1 = hp.alm2cl(plm_T2T1_ij, plm_T2T1_ji, lmax=lmax)
            cross_EE = hp.alm2cl(plm_EE_ij, plm_EE_ji, lmax=lmax)
            cross_TE = hp.alm2cl(plm_TE_ij, plm_TE_ji, lmax=lmax)
            cross_ET = hp.alm2cl(plm_ET_ij, plm_ET_ji, lmax=lmax)
            cross_TB = hp.alm2cl(plm_TB_ij, plm_TB_ji, lmax=lmax)
            cross_BT = hp.alm2cl(plm_BT_ij, plm_BT_ji, lmax=lmax)
            cross_EB = hp.alm2cl(plm_EB_ij, plm_EB_ji, lmax=lmax)
            cross_BE = hp.alm2cl(plm_BE_ij, plm_BE_ji, lmax=lmax)

            n1['total'] += auto + cross
            n1['T1T2'] += auto_T1T2 + cross_T1T2
            n1['T2T1'] += auto_T2T1 + cross_T2T1
            n1['EE'] += auto_EE + cross_EE
            n1['TE'] += auto_TE + cross_TE
            n1['ET'] += auto_ET + cross_ET
            n1['TB'] += auto_TB + cross_TB
            n1['BT'] += auto_BT + cross_BT
            n1['EB'] += auto_EB + cross_EB
            n1['BE'] += auto_BE + cross_BE

        n1['total'] *= 1/num
        n1['T1T2'] *= 1/num
        n1['T2T1'] *= 1/num
        n1['EE'] *= 1/num
        n1['TE'] *= 1/num
        n1['ET'] *= 1/num
        n1['TB'] *= 1/num
        n1['BT'] *= 1/num
        n1['EB'] *= 1/num
        n1['BE'] *= 1/num

        n0 = get_n0(sims=sims,qetype=qetype,config=config,
                    cmbonly=True,append=append)

        n1['total'] -= n0['total']
        n1['T1T2'] -= n0['TT']
        n1['T2T1'] -= n0['TT']
        n1['EE'] -= n0['EE']
        n1['TE'] -= n0['TE']
        n1['ET'] -= n0['ET']
        n1['TB'] -= n0['TB']
        n1['BT'] -= n0['BT']
        n1['EB'] -= n0['EB']
        n1['BE'] -= n0['BE']

        with open(filename, 'wb') as f:
            pickle.dump(n1, f)
    else:
        print('Invalid argument qetype.')

    return n1

def get_sim_response(est, config, gmv, append='mh', sims=np.arange(99)+1,filename=None):
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
        # File doesn't exist!
        cross_uncorrected_all = 0
        auto_input_all = 0
        for ii, sim in enumerate(sims):
            # Load plm
            if gmv:
                plm = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            else:
                plm = np.load(dir_out+f'/plm_{est}_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            input_plm = hp.read_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/phi/phi_lmax_{lmax}/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_phi1_seed{sim}_lmax{lmax}.alm')
            # Cross correlate with input plm
            # For response from sims, want to use plms that are not response corrected
            cross_uncorrected_all += hp.alm2cl(input_plm, plm, lmax=lmax)
            auto_input_all += hp.alm2cl(input_plm, input_plm, lmax=lmax)

        # Get "response from sims" calculated the same way as the MC response
        auto_input_avg = auto_input_all / num
        cross_uncorrected_avg = cross_uncorrected_all / num
        sim_resp = cross_uncorrected_avg / auto_input_avg
        np.save(filename, sim_resp)
    return sim_resp

####################

analyze()
