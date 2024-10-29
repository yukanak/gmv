#!/usr/bin/env python3
import numpy as np
import pickle
import healpy as hp
import camb
import os, sys
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import gmv_resp_alt as gmv_resp
import healqest_utils as utils
import matplotlib.pyplot as plt
import weights
import wignerd
import resp

def analyze(sims=np.arange(99)+1,n0_n1_sims=np.arange(98)+1,
            config_file='mh_yuka.yaml',
            save_fig=True,
            n0=True,n1=True,resp_from_sims=True,
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
    num = len(sims)
    bin_centers = (lbins[:-1] + lbins[1:]) / 2
    digitized = np.digitize(l, lbins)
    append = 'cmbonly'

    # Get SQE response
    ests = ['TT', 'EE', 'TE', 'TE', 'TB', 'TB', 'EB', 'EB']
    resps_original = np.zeros((len(l),len(ests)), dtype=np.complex_)
    inv_resps_original = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
    for i, est in enumerate(ests):
        resps_original[:,i] = get_sim_response(est,config,gmv=False,append=append,sims=sims)
        inv_resps_original[1:,i] = 1/(resps_original)[1:,i]
    resp_original = np.sum(resps_original, axis=1)
    inv_resp_original = np.zeros_like(l,dtype=np.complex_); inv_resp_original[1:] = 1/(resp_original)[1:]

    # GMV response
    if resp_from_sims:
        resp_gmv = get_sim_response('all',config,gmv=True,append=append,sims=sims)
        resp_gmv_TTEETE = get_sim_response('TTEETE',config,gmv=True,append=append,sims=sims)
        resp_gmv_TBEB = get_sim_response('TBEB',config,gmv=True,append=append,sims=sims)
    inv_resp_gmv = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv[1:] = 1./(resp_gmv)[1:]
    inv_resp_gmv_TTEETE = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TTEETE[1:] = 1./(resp_gmv_TTEETE)[1:]
    inv_resp_gmv_TBEB = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TBEB[1:] = 1./(resp_gmv_TBEB)[1:]

    if n0:
        # Get N0 correction (remember these still need (l*(l+1))**2/4 factor to convert to kappa)
        n0_gmv = get_n0(sims=n0_n1_sims,qetype='gmv',config=config,
                        resp_from_sims=resp_from_sims,append=append)
        n0_gmv_total = n0_gmv['total'] * (l*(l+1))**2/4
        n0_gmv_TTEETE = n0_gmv['TTEETE'] * (l*(l+1))**2/4
        n0_gmv_TBEB = n0_gmv['TBEB'] * (l*(l+1))**2/4
        n0_original = get_n0(sims=n0_n1_sims,qetype='sqe',config=config,
                             resp_from_sims=resp_from_sims,append=append)
        n0_original_total = n0_original['total'] * (l*(l+1))**2/4
        n0_original_T1T2 = n0_original['T1T2'] * (l*(l+1))**2/4
        n0_original_T2T1 = n0_original['T2T1'] * (l*(l+1))**2/4
        n0_original_EE = n0_original['EE'] * (l*(l+1))**2/4
        n0_original_TE = n0_original['TE'] * (l*(l+1))**2/4
        n0_original_ET = n0_original['ET'] * (l*(l+1))**2/4
        n0_original_TB = n0_original['TB'] * (l*(l+1))**2/4
        n0_original_BT = n0_original['BT'] * (l*(l+1))**2/4
        n0_original_EB = n0_original['EB'] * (l*(l+1))**2/4
        n0_original_BE = n0_original['BE'] * (l*(l+1))**2/4

    if n1:
        n1_gmv = get_n1(sims=n0_n1_sims,qetype='gmv',config=config,
                        resp_from_sims=resp_from_sims,append=append)
        n1_gmv_total = n1_gmv['total'] * (l*(l+1))**2/4
        n1_gmv_TTEETE = n1_gmv['TTEETE'] * (l*(l+1))**2/4
        n1_gmv_TBEB = n1_gmv['TBEB'] * (l*(l+1))**2/4
        n1_original = get_n1(sims=n0_n1_sims,qetype='sqe',config=config,
                             resp_from_sims=resp_from_sims,append=append)
        n1_original_total = n1_original['total'] * (l*(l+1))**2/4
        n1_original_T1T2 = n1_original['T1T2'] * (l*(l+1))**2/4
        n1_original_T2T1 = n1_original['T2T1'] * (l*(l+1))**2/4
        n1_original_EE = n1_original['EE'] * (l*(l+1))**2/4
        n1_original_TE = n1_original['TE'] * (l*(l+1))**2/4
        n1_original_ET = n1_original['ET'] * (l*(l+1))**2/4
        n1_original_TB = n1_original['TB'] * (l*(l+1))**2/4
        n1_original_BT = n1_original['BT'] * (l*(l+1))**2/4
        n1_original_EB = n1_original['EB'] * (l*(l+1))**2/4
        n1_original_BT = n1_original['BE'] * (l*(l+1))**2/4

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

    for ii, sim in enumerate(sims):
        # Load GMV plms
        plm_gmv = np.load(dir_out+f'/outputs_lensing19-20_no_foregrounds_with_ilc_noise/plm_all_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
        plm_gmv_TTEETE = np.load(dir_out+f'/outputs_lensing19-20_no_foregrounds_with_ilc_noise/plm_TTEETE_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
        plm_gmv_TBEB = np.load(dir_out+f'/outputs_lensing19-20_no_foregrounds_with_ilc_noise/plm_TBEB_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')

        # Load SQE plms
        plms_original = np.zeros((len(np.load(dir_out+f'/outputs_lensing19-20_no_foregrounds_with_ilc_noise/plm_TT_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')),len(ests)), dtype=np.complex_)
        for i, est in enumerate(ests):
            plms_original[:,i] = np.load(dir_out+f'/outputs_lensing19-20_no_foregrounds_with_ilc_noise/plm_{est}_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
        plm_original = np.sum(plms_original, axis=1)

        # Response correct
        plm_gmv_resp_corr = hp.almxfl(plm_gmv,inv_resp_gmv)
        plm_gmv_resp_corr_TTEETE = hp.almxfl(plm_gmv_TTEETE,inv_resp_gmv_TTEETE)
        plm_gmv_resp_corr_TBEB = hp.almxfl(plm_gmv_TBEB,inv_resp_gmv_TBEB)
        plm_original_resp_corr = hp.almxfl(plm_original,inv_resp_original)
        plm_original_resp_corr_T1T2 = hp.almxfl(plms_original[:,0],inv_resps_original[:,0])
        plm_original_resp_corr_EE = hp.almxfl(plms_original[:,1],inv_resps_original[:,1])
        plm_original_resp_corr_TE = hp.almxfl(plms_original[:,2],inv_resps_original[:,2])
        plm_original_resp_corr_ET = hp.almxfl(plms_original[:,3],inv_resps_original[:,3])
        plm_original_resp_corr_TB = hp.almxfl(plms_original[:,4],inv_resps_original[:,4])
        plm_original_resp_corr_BT = hp.almxfl(plms_original[:,5],inv_resps_original[:,5])
        plm_original_resp_corr_EB = hp.almxfl(plms_original[:,6],inv_resps_original[:,6])
        plm_original_resp_corr_BE = hp.almxfl(plms_original[:,7],inv_resps_original[:,7])

        # Get spectra
        auto_gmv = hp.alm2cl(plm_gmv_resp_corr, plm_gmv_resp_corr, lmax=lmax) * (l*(l+1))**2/4
        auto_gmv_TTEETE = hp.alm2cl(plm_gmv_resp_corr_TTEETE, plm_gmv_resp_corr_TTEETE, lmax=lmax) * (l*(l+1))**2/4
        auto_gmv_TBEB = hp.alm2cl(plm_gmv_resp_corr_TBEB, plm_gmv_resp_corr_TBEB, lmax=lmax) * (l*(l+1))**2/4
        auto_original = hp.alm2cl(plm_original_resp_corr, plm_original_resp_corr, lmax=lmax) * (l*(l+1))**2/4
        auto_original_T1T2 = hp.alm2cl(plm_original_resp_corr_T1T2, plm_original_resp_corr_T1T2, lmax=lmax) * (l*(l+1))**2/4
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
            #auto_original_debiased_TT = (auto_original_T1T2+auto_original_T2T1)*0.5 - n0_original_T1T2 - (n1_original_T1T2+n1_original_T2T1)*0.5
        elif n0:
            auto_gmv_debiased = auto_gmv - n0_gmv_total
            auto_original_debiased = auto_original - n0_original_total
            #auto_original_debiased_TT = (auto_original_T1T2+auto_original_T2T1)*0.5 - n0_original_T1T2

        # Sum the auto spectra
        auto_gmv_all += auto_gmv
        auto_gmv_all_TTEETE += auto_gmv_TTEETE
        auto_gmv_all_TBEB += auto_gmv_TBEB
        auto_original_all += auto_original
        auto_original_all_T1T2 += auto_original_T1T2
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
            #auto_original_debiased_all_TT += auto_original_debiased_TT

        input_plm = hp.read_alm(f'/oak/stanford/orgs/kipac//users/yukanaka/lensing19-20/inputcmb/phi/phi_lmax_{lmax}/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_phi1_seed{sim}_lmax{lmax}.alm')
        # Cross correlate with input plm
        cross_gmv_all += hp.alm2cl(input_plm, plm_gmv_resp_corr, lmax=lmax) * (l*(l+1))**2/4
        cross_original_all += hp.alm2cl(input_plm, plm_original_resp_corr, lmax=lmax) * (l*(l+1))**2/4
        # If debiasing, get the binned ratio against input
        if n0:
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

    # Average
    auto_gmv_avg = auto_gmv_all / num
    auto_gmv_avg_TTEETE = auto_gmv_all_TTEETE / num
    auto_gmv_avg_TBEB = auto_gmv_all_TBEB / num
    auto_original_avg = auto_original_all / num
    auto_original_avg_T1T2 = auto_original_all_T1T2 / num
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

    #auto_gmv_avg = {'total':auto_gmv_avg, 'TTEETE':auto_gmv_avg_TTEETE, 'TBEB':auto_gmv_avg_TBEB}
    #auto_sqe_avg = {'total':auto_original_avg, 'TT':auto_original_avg_T1T2, 'EE':auto_original_avg_EE, 'TE':auto_original_avg_TE, 'TB':auto_original_avg_TB, 'EB':auto_original_avg_EB}
    #with open('auto_gmv_avg_old_plms.pkl', 'wb') as f:
    #    pickle.dump(auto_gmv_avg, f)
    #with open('auto_sqe_avg_old_plms.pkl', 'wb') as f:
    #    pickle.dump(auto_sqe_avg, f)

    # Average the cross with input spectra
    cross_gmv_avg = cross_gmv_all / num
    cross_original_avg = cross_original_all / num

    # Theory spectrum
    clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
    ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
    clkk = slpp * (l*(l+1))**2/4

    # Plot
    plt.figure(1)
    plt.clf()
    plt.plot(l, auto_gmv_debiased_avg, color='cornflowerblue', linestyle='-', label="Auto Spectrum (GMV)")
    plt.plot(l, auto_original_debiased_avg, color='lightcoral', linestyle='-', label=f'Auto Spectrum (SQE)')
    #plt.plot(l, auto_original_debiased_avg_TT, color='thistle', linestyle='-', label=f'Auto Spectrum (SQE, TT)')

    plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')

    plt.plot(bin_centers, binned_auto_gmv_debiased_avg, color='darkblue', marker='o', linestyle='None', ms=3, label="Auto Spectrum (GMV)")
    plt.plot(bin_centers, binned_auto_original_debiased_avg, color='firebrick', marker='o', linestyle='None', ms=3, label="Auto Spectrum (SQE)")
    #plt.plot(bin_centers, binned_auto_original_debiased_avg_TT, color='mediumorchid', marker='o', linestyle='None', ms=3, label="Auto Spectrum (SQE, TT)")

    plt.ylabel("$C_\ell^{\kappa\kappa}$")
    plt.xlabel('$\ell$')
    plt.title(f'Spectra Averaged over {num} Sims, MH')
    plt.legend(loc='upper left', fontsize='x-small', bbox_to_anchor=(1,0.5))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,lmax)
    plt.ylim(1e-9,1e-6)
    if save_fig:
        if n1:
            if resp_from_sims:
                plt.savefig(dir_out+f'/figs/{num}_sims_comparison_{append}_plus_noise_from_old_resp_from_sims_n0n1subtracted.png',bbox_inches='tight')
        else:
            if resp_from_sims:
                plt.savefig(dir_out+f'/figs/{num}_sims_comparison_{append}_plus_noise_from_old_resp_from_sims_n0subtracted.png',bbox_inches='tight')

    plt.figure(2)
    plt.clf()
    # Ratios with error bars
    plt.axhline(y=1, color='k', linestyle='--')
    plt.errorbar(bin_centers,ratio_gmv,yerr=errorbars_gmv,color='darkblue', marker='o', linestyle='None', ms=3, label="Ratio GMV/Input")
    plt.errorbar(bin_centers,ratio_original,yerr=errorbars_original,color='firebrick', marker='o', linestyle='None', ms=3, label="Ratio Original/Input")
    #plt.errorbar(bin_centers,ratio_original_TT,yerr=errorbars_original_TT,color='mediumorchid', marker='o', linestyle='None', ms=3, label="Ratio Original/Input, TT")
    plt.xlabel('$\ell$')
    plt.title(f'Spectra Averaged over {num} Sims')
    plt.legend(loc='lower left', fontsize='x-small')
    plt.xscale('log')
    plt.ylim(0.98,1.02)
    #plt.ylim(0.95,1.10)
    plt.xlim(10,lmax)
    if save_fig:
        if n1:
            if resp_from_sims:
                plt.savefig(dir_out+f'/figs/{num}_sims_comparison_{append}_plus_noise_from_old_resp_from_sims_n0n1subtracted_binnedratio.png',bbox_inches='tight')
        else:
            if resp_from_sims:
                plt.savefig(dir_out+f'/figs/{num}_sims_comparison_{append}_plus_noise_from_old_resp_from_sims_n0subtracted_binnedratio.png',bbox_inches='tight')

    '''
    auto_gmv_avg = pickle.load(open('auto_gmv_avg_old_plms.pkl','rb'))
    auto_sqe_avg = pickle.load(open('auto_sqe_avg_old_plms.pkl','rb'))
    auto_gmv_avg_old = pickle.load(open('auto_gmv_avg_old_plms_old_code.pkl','rb'))
    auto_sqe_avg_old = pickle.load(open('auto_sqe_avg_old_plms_old_code.pkl','rb'))

    plt.figure(3)
    plt.clf()
    plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')

    plt.plot(l, auto_gmv_avg['total'], color='darkblue', linestyle='-', label="Auto Spectrum (GMV), recalculated")
    plt.plot(l, auto_sqe_avg['total'], color='firebrick', linestyle='-', label=f'Auto Spectrum (SQE), recalculated')

    plt.plot(l, auto_gmv_avg_old['total'], color='cornflowerblue', linestyle='--', label="Auto Spectrum (GMV), last year's code")
    plt.plot(l, auto_sqe_avg_old['total'], color='lightcoral', linestyle='--', label=f"Auto Spectrum (SQE), last year's code")

    plt.ylabel("$C_\ell^{\kappa\kappa}$")
    plt.xlabel('$\ell$')
    plt.title(f'Spectra')
    plt.legend(loc='upper left', fontsize='x-small', bbox_to_anchor=(1,0.5))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,lmax)
    plt.ylim(1e-9,1e-6)
    plt.savefig(dir_out+f'/figs/biased_autospec_comparison_{append}_plus_noise_from_old_resp_from_sims.png',bbox_inches='tight')

    plt.clf()
    plt.axhline(y=1, color='k', linestyle='--')

    plt.plot(l, auto_gmv_avg['total']/auto_gmv_avg_old['total'], color='darkblue', linestyle='-', label="Auto Spectrum Ratio (GMV), recalculated/from last year")
    plt.plot(l, auto_sqe_avg['total']/auto_sqe_avg_old['total'], color='firebrick', linestyle='-', label=f'Auto Spectrum Ratio (SQE), recalculated/from last year')

    plt.xlabel('$\ell$')
    plt.legend(loc='upper left', fontsize='x-small', bbox_to_anchor=(1,0.5))
    plt.xscale('log')
    plt.xlim(10,lmax)
    plt.savefig(dir_out+f'/figs/biased_autospec_comparison_{append}_plus_noise_from_old_resp_from_sims_ratio.png',bbox_inches='tight')

    filename = dir_out+f'/n0/n0_lensing19-20_no_foregrounds_with_ilc_noise/n0_99simpairs_healqest_gmv_lmaxT3000_lmaxP4096_nside2048_cmbonly_resp_from_sims.pkl'
    n0_gmv_old = pickle.load(open(filename,'rb'))
    n0_gmv_old_total = n0_gmv_old['total'] * (l*(l+1))**2/4
    filename = dir_out+f'/n0/n0_lensing19-20_no_foregrounds_with_ilc_noise/n0_99simpairs_healqest_sqe_lmaxT3000_lmaxP4096_nside2048_cmbonly_resp_from_sims.pkl'
    n0_original_old = pickle.load(open(filename,'rb'))
    n0_original_old_total = n0_original_old['total'] * (l*(l+1))**2/4
    filename = dir_out+f'/n1/n1_lensing19-20_no_foregrounds_with_ilc_noise/n1_99simpairs_healqest_gmv_lmaxT3000_lmaxP4096_nside2048_cmbonly_resp_from_sims.pkl'
    n1_gmv_old = pickle.load(open(filename,'rb'))
    n1_gmv_old_total = n1_gmv_old['total'] * (l*(l+1))**2/4
    filename = dir_out+f'/n1/n1_lensing19-20_no_foregrounds_with_ilc_noise/n1_99simpairs_healqest_sqe_lmaxT3000_lmaxP4096_nside2048_cmbonly_resp_from_sims.pkl'
    n1_original_old = pickle.load(open(filename,'rb'))
    n1_original_old_total = n1_original_old['total'] * (l*(l+1))**2/4

    plt.clf()
    plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')

    plt.plot(l, auto_gmv_avg['total'] - n0_gmv_total - n1_gmv_old_total, color='darkblue', linestyle='-', label="Auto Spectrum (GMV), recalculated WITH OLD N1")
    plt.plot(l, auto_sqe_avg['total'] - n0_original_total - n1_original_old_total, color='firebrick', linestyle='-', label=f'Auto Spectrum (SQE), recalculated WITH OLD N1')

    plt.plot(l, auto_gmv_avg_old['total'] - n0_gmv_old_total - n1_gmv_old_total, color='cornflowerblue', linestyle='--', label="Auto Spectrum (GMV), last year's code")
    plt.plot(l, auto_sqe_avg_old['total'] - n0_original_old_total - n1_original_old_total, color='lightcoral', linestyle='--', label=f"Auto Spectrum (SQE), last year's code")

    plt.ylabel("$C_\ell^{\kappa\kappa}$")
    plt.xlabel('$\ell$')
    plt.title(f'Spectra')
    plt.legend(loc='upper left', fontsize='x-small', bbox_to_anchor=(1,0.5))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,lmax)
    plt.ylim(1e-9,1e-6)
    plt.savefig(dir_out+f'/figs/unbiased_autospec_comparison_{append}_plus_noise_from_old_resp_from_sims.png',bbox_inches='tight')
    '''

def compare_resp(config_file='mh_yuka.yaml',
                 save_fig=True):
    config = utils.parse_yaml(config_file)
    lmax = config['lensrec']['Lmax']
    lmin = config['lensrec']['lminT']
    dir_out = config['dir_out']
    l = np.arange(0,lmax+1)
    append = 'cmbonly'
    sims=np.arange(40)+1

    # Get SQE response, computed last year
    ests = ['TT', 'EE', 'TE', 'TB', 'EB']
    resps_original_old = np.zeros((len(l),len(ests)), dtype=np.complex_)
    inv_resps_original_old = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
    for i, est in enumerate(ests):
        resps_original_old[:,i] = np.load(dir_out+f'/resp/sim_resp_sqe_est{est}_100sims_lmaxT3000_lmaxP4096_nside2048_cmbonly.npy')
        inv_resps_original_old[1:,i] = 1/(resps_original_old)[1:,i]
    resp_original_old = np.load(dir_out+f'/resp/sim_resp_sqe_estall_100sims_lmaxT3000_lmaxP4096_nside2048_cmbonly.npy')
    inv_resp_original_old = np.zeros_like(l,dtype=np.complex_); inv_resp_original_old[1:] = 1/(resp_original_old)[1:]

    #ests = ['TT', 'EE', 'TE', 'TE', 'TB', 'TB', 'EB', 'EB']
    #resps_original_old = np.zeros((len(l),len(ests)), dtype=np.complex_)
    #inv_resps_original_old = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
    #for i, est in enumerate(ests):
    #    resps_original_old[:,i] = get_sim_response(est,config,gmv=False,append=append,sims=sims)
    #    inv_resps_original_old[1:,i] = 1/(resps_original_old)[1:,i]
    #resp_original_old = np.sum(resps_original_old, axis=1)
    #inv_resp_original_old = np.zeros_like(l,dtype=np.complex_); inv_resp_original_old[1:] = 1/(resp_original_old)[1:]

    # GMV response, computed last year
    resp_gmv_old = get_sim_response('all',config,gmv=True,append=append,sims=sims)
    resp_gmv_TTEETE_old = get_sim_response('TTEETE',config,gmv=True,append=append,sims=sims)
    resp_gmv_TBEB_old = get_sim_response('TBEB',config,gmv=True,append=append,sims=sims)
    inv_resp_gmv_old = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_old[1:] = 1./(resp_gmv_old)[1:]
    inv_resp_gmv_TTEETE_old = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TTEETE_old[1:] = 1./(resp_gmv_TTEETE_old)[1:]
    inv_resp_gmv_TBEB_old = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TBEB_old[1:] = 1./(resp_gmv_TBEB_old)[1:]

    # SQE response, notmh_crossilcFalse
    new_ests = ['T1T2', 'EE', 'TE', 'ET', 'TB', 'BT', 'EB', 'BE']
    resps_original_new = np.zeros((len(l),len(new_ests)), dtype=np.complex_)
    inv_resps_original_new = np.zeros((len(l),len(new_ests)) ,dtype=np.complex_)
    for i, est in enumerate(new_ests):
        resps_original_new[:,i] = np.load(dir_out+f'/resp/sim_resp_sqe_est{est}_lmaxT3000_lmaxP4096_lmin300_cltypelcmb_notmh_crossilcFalse.npy')
        inv_resps_original_new[1:,i] = 1/(resps_original_new)[1:,i]
    resp_original_new = np.sum(resps_original_new, axis=1)
    inv_resp_original_new = np.zeros_like(l,dtype=np.complex_); inv_resp_original_new[1:] = 1/(resp_original_new)[1:]

    # GMV response, notmh_crossilcFalse
    resp_gmv_new = np.load(dir_out+f'/resp/sim_resp_gmv_estall_lmaxT3000_lmaxP4096_lmin300_cltypelcmb_notmh_crossilcFalse.npy')
    resp_gmv_TTEETE_new = np.load(dir_out+f'/resp/sim_resp_gmv_estTTEETE_lmaxT3000_lmaxP4096_lmin300_cltypelcmb_notmh_crossilcFalse.npy')
    resp_gmv_TBEB_new = np.load(dir_out+f'/resp/sim_resp_gmv_estTBEB_lmaxT3000_lmaxP4096_lmin300_cltypelcmb_notmh_crossilcFalse.npy')
    inv_resp_gmv_new = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_new[1:] = 1./(resp_gmv_new)[1:]
    inv_resp_gmv_TTEETE_new = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TTEETE_new[1:] = 1./(resp_gmv_TTEETE_new)[1:]
    inv_resp_gmv_TBEB_new = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TBEB_new[1:] = 1./(resp_gmv_TBEB_new)[1:]

    # Theory spectrum
    clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
    ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
    clkk = slpp * (l*(l+1))**2/4

    plt.figure(0)
    plt.clf()
    plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')

    plt.plot(l, inv_resp_original_new * (l*(l+1))**2/4, color='lightcoral', linestyle='--', label='$1/R$ (SQE new, notmh_crossilcFalse)')
    plt.plot(l, inv_resps_original_new[:,0] * (l*(l+1))**2/4, color='sandybrown', linestyle='--', label='$1/R$ (SQE new, T1T2)')
    plt.plot(l, inv_resps_original_new[:,1] * (l*(l+1))**2/4, color='plum', linestyle='--', label='$1/R$ (SQE new, EE)')
    plt.plot(l, 0.5*inv_resps_original_new[:,2] * (l*(l+1))**2/4, color='lightgreen', linestyle='--', label='$1/(2R)$ (SQE new, TE)')
    plt.plot(l, 0.5*inv_resps_original_new[:,3] * (l*(l+1))**2/4, color='silver', linestyle='--', label='$1/(2R)$ (SQE new, ET)')
    plt.plot(l, 0.5*inv_resps_original_new[:,4] * (l*(l+1))**2/4, color='palegoldenrod', linestyle='--', label='$1/(2R)$ (SQE new, TB)')
    plt.plot(l, 0.5*inv_resps_original_new[:,5] * (l*(l+1))**2/4, color='slateblue', linestyle='--', label='$1/(2R)$ (SQE new, BT)')
    plt.plot(l, 0.5*inv_resps_original_new[:,6] * (l*(l+1))**2/4, color='bisque', linestyle='--', label='$1/(2R$) (SQE new, EB)')
    plt.plot(l, 0.5*inv_resps_original_new[:,7] * (l*(l+1))**2/4, color='paleturquoise', linestyle='--', label='$1/(2R$) (SQE new, BE)')

    plt.plot(l, inv_resp_original_old * (l*(l+1))**2/4, color='firebrick', linestyle='--', label='$1/R$ (SQE old, calculated last year)')
    plt.plot(l, inv_resps_original_old[:,0] * (l*(l+1))**2/4, color='sienna', linestyle='--', label='$1/R$ (SQE old, TT)')
    plt.plot(l, inv_resps_original_old[:,1] * (l*(l+1))**2/4, color='mediumorchid', linestyle='--', label='$1/R$ (SQE old, EE)')
    plt.plot(l, 0.5*inv_resps_original_old[:,2] * (l*(l+1))**2/4, color='forestgreen', linestyle='--', label='$1/(2R)$ (SQE old, TE)')
    plt.plot(l, 0.5*inv_resps_original_old[:,3] * (l*(l+1))**2/4, color='gold', linestyle='--', label='$1/(2R)$ (SQE old, TB)')
    plt.plot(l, 0.5*inv_resps_original_old[:,4] * (l*(l+1))**2/4, color='orange', linestyle='--', label='$1/(2R$) (SQE old, EB)')

    plt.ylabel("$C_\ell^{\kappa\kappa}$")
    plt.xlabel('$\ell$')
    plt.title('$1/R$')
    plt.legend(loc='center left', fontsize='small', bbox_to_anchor=(1,0.5))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,lmax)
    plt.ylim(8e-9,1e-5)
    if save_fig:
        plt.savefig(dir_out+f'/figs/notmh_response_comparison_sqe_only.png',bbox_inches='tight')

    plt.figure(1)
    plt.clf()
    plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')
    plt.plot(l, inv_resp_gmv_new * (l*(l+1))**2/4, color='cornflowerblue', linestyle='--', label='$1/R$ (GMV new, notmh_crossilcFalse)')
    plt.plot(l, inv_resp_gmv_TTEETE_new * (l*(l+1))**2/4, color='lightgreen', linestyle='--', label='1/R (GMV new, TTEETE)')
    plt.plot(l, inv_resp_gmv_TBEB_new * (l*(l+1))**2/4, color='thistle', linestyle='--', label='1/R (GMV new, TBEB)')

    plt.plot(l, inv_resp_gmv_old * (l*(l+1))**2/4, color='darkblue', linestyle='--', label='$1/R$ (GMV old, calculated last year)')
    plt.plot(l, inv_resp_gmv_TTEETE_old * (l*(l+1))**2/4, color='forestgreen', linestyle='--', label='1/R (GMV old, TTEETE)')
    plt.plot(l, inv_resp_gmv_TBEB_old * (l*(l+1))**2/4, color='blueviolet', linestyle='--', label='1/R (GMV old, TBEB)')

    plt.ylabel("$C_\ell^{\kappa\kappa}$")
    plt.xlabel('$\ell$')
    plt.title('$1/R$')
    plt.legend(loc='center left', fontsize='small', bbox_to_anchor=(1,0.5))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,lmax)
    plt.ylim(8e-9,1e-6)
    if save_fig:
        plt.savefig(dir_out+f'/figs/notmh_response_comparison_gmv_only.png',bbox_inches='tight')

def compare_n1(config_file='mh_yuka.yaml',sims=np.arange(40)+1,n0_n1_sims=np.arange(39)+1,
               resp_from_sims=True,save_fig=True):

    config = utils.parse_yaml(config_file)
    lmax = config['lensrec']['Lmax']
    lmin = config['lensrec']['lminT']
    lmaxT = config['lensrec']['lmaxT']
    lmaxP = config['lensrec']['lmaxP']
    nside = config['lensrec']['nside']
    dir_out = config['dir_out']
    l = np.arange(0,lmax+1)
    num = len(n0_n1_sims)
    append = 'cmbonly'

    # Get SQE response
    ests = ['TT', 'EE', 'TE', 'TE', 'TB', 'TB', 'EB', 'EB']
    resps_original = np.zeros((len(l),len(ests)), dtype=np.complex_)
    inv_resps_original = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
    for i, est in enumerate(ests):
        resps_original[:,i] = get_sim_response(est,config,gmv=False,append=append,sims=sims)
        inv_resps_original[1:,i] = 1/(resps_original)[1:,i]
    resp_original = np.sum(resps_original, axis=1)
    inv_resp_original = np.zeros_like(l,dtype=np.complex_); inv_resp_original[1:] = 1/(resp_original)[1:]

    # GMV response
    if resp_from_sims:
        resp_gmv = get_sim_response('all',config,gmv=True,append=append,sims=sims)
        resp_gmv_TTEETE = get_sim_response('TTEETE',config,gmv=True,append=append,sims=sims)
        resp_gmv_TBEB = get_sim_response('TBEB',config,gmv=True,append=append,sims=sims)
    inv_resp_gmv = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv[1:] = 1./(resp_gmv)[1:]
    inv_resp_gmv_TTEETE = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TTEETE[1:] = 1./(resp_gmv_TTEETE)[1:]
    inv_resp_gmv_TBEB = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TBEB[1:] = 1./(resp_gmv_TBEB)[1:]

    # Get N1 correction (remember these still need (l*(l+1))**2/4 factor to convert to kappa)
    # GMV
    n1_gmv = get_n1(sims=n0_n1_sims,qetype='gmv',config=config,
                    resp_from_sims=resp_from_sims,append=append)
    n1_gmv_total = n1_gmv['total'] * (l*(l+1))**2/4
    n1_gmv_TTEETE = n1_gmv['TTEETE'] * (l*(l+1))**2/4
    n1_gmv_TBEB = n1_gmv['TBEB'] * (l*(l+1))**2/4

    # SQE
    n1_original = get_n1(sims=n0_n1_sims,qetype='sqe',config=config,
                         resp_from_sims=resp_from_sims,append=append)
    n1_original_total = n1_original['total'] * (l*(l+1))**2/4
    n1_original_T1T2 = n1_original['T1T2'] * (l*(l+1))**2/4
    n1_original_T2T1 = n1_original['T2T1'] * (l*(l+1))**2/4
    n1_original_EE = n1_original['EE'] * (l*(l+1))**2/4
    n1_original_TE = n1_original['TE'] * (l*(l+1))**2/4
    n1_original_ET = n1_original['ET'] * (l*(l+1))**2/4
    n1_original_TB = n1_original['TB'] * (l*(l+1))**2/4
    n1_original_BT = n1_original['BT'] * (l*(l+1))**2/4
    n1_original_EB = n1_original['EB'] * (l*(l+1))**2/4
    n1_original_BE = n1_original['BE'] * (l*(l+1))**2/4

    # GMV, notmh_crossilcFalse
    filename = dir_out+f'/n1/n1_39simpairs_healqest_gmv_lmaxT3000_lmaxP4096_nside2048_notmh_crossilcFalse_resp_from_sims.pkl'
    n1_gmv_new = pickle.load(open(filename,'rb'))
    n1_gmv_new_total = n1_gmv_new['total'] * (l*(l+1))**2/4
    n1_gmv_new_TTEETE = n1_gmv_new['TTEETE'] * (l*(l+1))**2/4
    n1_gmv_new_TBEB = n1_gmv_new['TBEB'] * (l*(l+1))**2/4

    # SQE, notmh_crossilcFalse
    filename = dir_out+f'/n1/n1_39simpairs_healqest_sqe_lmaxT3000_lmaxP4096_nside2048_notmh_crossilcFalse_resp_from_sims.pkl'
    n1_original_new = pickle.load(open(filename,'rb'))
    n1_original_new_total = n1_original_new['total'] * (l*(l+1))**2/4
    n1_original_new_T1T2 = n1_original_new['T1T2'] * (l*(l+1))**2/4
    n1_original_new_T2T1 = n1_original_new['T2T1'] * (l*(l+1))**2/4
    n1_original_new_EE = n1_original_new['EE'] * (l*(l+1))**2/4
    n1_original_new_TE = n1_original_new['TE'] * (l*(l+1))**2/4
    n1_original_new_ET = n1_original_new['ET'] * (l*(l+1))**2/4
    n1_original_new_TB = n1_original_new['TB'] * (l*(l+1))**2/4
    n1_original_new_BT = n1_original_new['BT'] * (l*(l+1))**2/4
    n1_original_new_EB = n1_original_new['EB'] * (l*(l+1))**2/4
    n1_original_new_BE = n1_original_new['BE'] * (l*(l+1))**2/4

    # GMV, calculated last year
    filename = dir_out+f'/n1/n1_lensing19-20_no_foregrounds_with_ilc_noise/n1_99simpairs_healqest_gmv_lmaxT3000_lmaxP4096_nside2048_cmbonly_resp_from_sims.pkl'
    n1_gmv_old = pickle.load(open(filename,'rb'))
    n1_gmv_old_total = n1_gmv_old['total'] * (l*(l+1))**2/4
    n1_gmv_old_TTEETE = n1_gmv_old['TTEETE'] * (l*(l+1))**2/4
    n1_gmv_old_TBEB = n1_gmv_old['TBEB'] * (l*(l+1))**2/4

    # SQE, calculated last year
    filename = dir_out+f'/n1/n1_lensing19-20_no_foregrounds_with_ilc_noise/n1_99simpairs_healqest_sqe_lmaxT3000_lmaxP4096_nside2048_cmbonly_resp_from_sims.pkl'
    n1_original_old = pickle.load(open(filename,'rb'))
    n1_original_old_total = n1_original_old['total'] * (l*(l+1))**2/4
    n1_original_old_TT = n1_original_old['TT'] * (l*(l+1))**2/4
    n1_original_old_EE = n1_original_old['EE'] * (l*(l+1))**2/4
    n1_original_old_TE = n1_original_old['TE'] * (l*(l+1))**2/4
    n1_original_old_TB = n1_original_old['TB'] * (l*(l+1))**2/4
    n1_original_old_EB = n1_original_old['EB'] * (l*(l+1))**2/4

    # Theory spectrum
    clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
    ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
    clkk = slpp * (l*(l+1))**2/4

    plt.figure(0)
    plt.clf()
    plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')

    plt.plot(l, n1_original_new_T1T2, color='sienna', linestyle='-',label='N1 (SQE new, TT)')
    plt.plot(l, n1_original_new_EE, color='mediumorchid', linestyle='-',label='N1 (SQE new, EE)')
    plt.plot(l, n1_original_new_TE, color='forestgreen', linestyle='-',label='N1 (SQE new, TE)')
    plt.plot(l, n1_original_new_TB, color='gold', linestyle='-',label='N1 (SQE new, TB)')
    plt.plot(l, n1_original_new_EB, color='orange', linestyle='-',label='N1 (SQE new, EB)')
    plt.plot(l, n1_original_new_total, color='firebrick', linestyle='-',label='N1 (SQE new, notmh_crossilcFalse)')

    plt.plot(l, n1_original_old_TT, color='chocolate', linestyle='--',label='N1 (SQE old, TT)')
    plt.plot(l, n1_original_old_EE, color='violet', linestyle='--',label='N1 (SQE old, EE)')
    plt.plot(l, n1_original_old_TE, color='darkseagreen', linestyle='--',label='N1 (SQE old, TE)')
    plt.plot(l, n1_original_old_TB, color='goldenrod', linestyle='--',label='N1 (SQE old, TB)')
    plt.plot(l, n1_original_old_EB, color='burlywood', linestyle='--',label='N1 (SQE old, EB)')
    plt.plot(l, n1_original_old_total, color='pink', linestyle='--',label='N1 (SQE old, calculated last year)')

    plt.ylabel("$C_\ell^{\kappa\kappa}$")
    plt.xlabel('$\ell$')
    plt.title(f'N1')
    plt.legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize='x-small')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,lmax)
    #plt.ylim(1e-8,1e-5)
    if save_fig:
        if resp_from_sims:
            plt.savefig(dir_out+f'/figs/n1_comparison_sqe_notmh_resp_from_sims.png',bbox_inches='tight')

    plt.clf()
    plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')

    plt.plot(l, n1_gmv_new_TTEETE, color='forestgreen', linestyle='-',label='N1 (GMV new, TTEETE)')
    plt.plot(l, n1_gmv_new_TBEB, color='blueviolet', linestyle='-',label='N1 (GMV new, TBEB)')
    plt.plot(l, n1_gmv_new_total, color='darkblue', linestyle='-',label='N1 (GMV new, notmh_crossilcFalse)')

    plt.plot(l, n1_gmv_old_TTEETE, color='olive', linestyle='--',label='N1 (GMV TTEETE, calculated last year)')
    plt.plot(l, n1_gmv_old_TBEB, color='rebeccapurple', linestyle='--',label='N1 (GMV TBEB, calculated last year)')
    plt.plot(l, n1_gmv_old_total, color='powderblue', linestyle='--',label='N1 (GMV old, calculated last year)')

    plt.ylabel("$C_\ell^{\kappa\kappa}$")
    plt.xlabel('$\ell$')
    plt.title(f'N1')
    plt.legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize='x-small')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,lmax)
    #plt.ylim(1e-8,1e-5)
    if save_fig:
        if resp_from_sims:
            plt.savefig(dir_out+f'/figs/n1_comparison_gmv_notmh_resp_from_sims.png',bbox_inches='tight')

    plt.clf()
    plt.axhline(y=1, color='k', linestyle='--')

    plt.plot(l, n1_gmv_new_total/n1_gmv_old_total, color='darkblue', linestyle='-', label="N1 Ratio (GMV), recalculated/from last year")
    plt.plot(l, n1_original_new_total/n1_original_old_total, color='firebrick', linestyle='-', label=f'N1 Ratio (SQE), recalculated/from last year')

    plt.xlabel('$\ell$')
    plt.legend(loc='upper left', fontsize='x-small', bbox_to_anchor=(1,0.5))
    plt.xscale('log')
    plt.xlim(10,lmax)
    plt.savefig(dir_out+f'/figs/n1_comparison_notmh_resp_from_sims_ratio.png',bbox_inches='tight')

def compare_n0(config_file='mh_yuka.yaml',sims=np.arange(40)+1,n0_n1_sims=np.arange(39)+1,
               resp_from_sims=True,save_fig=True):

    config = utils.parse_yaml(config_file)
    lmax = config['lensrec']['Lmax']
    lmin = config['lensrec']['lminT']
    lmaxT = config['lensrec']['lmaxT']
    lmaxP = config['lensrec']['lmaxP']
    nside = config['lensrec']['nside']
    dir_out = config['dir_out']
    l = np.arange(0,lmax+1)
    num = len(n0_n1_sims)
    append = 'cmbonly'

    # Get SQE response
    ests = ['TT', 'EE', 'TE', 'TE', 'TB', 'TB', 'EB', 'EB']
    resps_original = np.zeros((len(l),len(ests)), dtype=np.complex_)
    inv_resps_original = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
    for i, est in enumerate(ests):
        resps_original[:,i] = get_sim_response(est,config,gmv=False,append=append,sims=sims)
        inv_resps_original[1:,i] = 1/(resps_original)[1:,i]
    resp_original = np.sum(resps_original, axis=1)
    inv_resp_original = np.zeros_like(l,dtype=np.complex_); inv_resp_original[1:] = 1/(resp_original)[1:]

    # GMV response
    if resp_from_sims:
        resp_gmv = get_sim_response('all',config,gmv=True,append=append,sims=sims)
        resp_gmv_TTEETE = get_sim_response('TTEETE',config,gmv=True,append=append,sims=sims)
        resp_gmv_TBEB = get_sim_response('TBEB',config,gmv=True,append=append,sims=sims)
    inv_resp_gmv = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv[1:] = 1./(resp_gmv)[1:]
    inv_resp_gmv_TTEETE = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TTEETE[1:] = 1./(resp_gmv_TTEETE)[1:]
    inv_resp_gmv_TBEB = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TBEB[1:] = 1./(resp_gmv_TBEB)[1:]

    # Get N0 correction (remember these still need (l*(l+1))**2/4 factor to convert to kappa)
    # GMV
    n0_gmv = get_n0(sims=n0_n1_sims,qetype='gmv',config=config,
                    resp_from_sims=resp_from_sims,append=append)
    n0_gmv_total = n0_gmv['total'] * (l*(l+1))**2/4
    n0_gmv_TTEETE = n0_gmv['TTEETE'] * (l*(l+1))**2/4
    n0_gmv_TBEB = n0_gmv['TBEB'] * (l*(l+1))**2/4

    # SQE
    n0_original = get_n0(sims=n0_n1_sims,qetype='sqe',config=config,
                         resp_from_sims=resp_from_sims,append=append)
    n0_original_total = n0_original['total'] * (l*(l+1))**2/4
    n0_original_T1T2 = n0_original['T1T2'] * (l*(l+1))**2/4
    n0_original_T2T1 = n0_original['T2T1'] * (l*(l+1))**2/4
    n0_original_EE = n0_original['EE'] * (l*(l+1))**2/4
    n0_original_TE = n0_original['TE'] * (l*(l+1))**2/4
    n0_original_ET = n0_original['ET'] * (l*(l+1))**2/4
    n0_original_TB = n0_original['TB'] * (l*(l+1))**2/4
    n0_original_BT = n0_original['BT'] * (l*(l+1))**2/4
    n0_original_EB = n0_original['EB'] * (l*(l+1))**2/4
    n0_original_BE = n0_original['BE'] * (l*(l+1))**2/4

    # GMV, notmh_crossilcFalse
    filename = dir_out+f'/n0/n0_39simpairs_healqest_gmv_lmaxT3000_lmaxP4096_nside2048_notmh_crossilcFalse_resp_from_sims.pkl'
    n0_gmv_new = pickle.load(open(filename,'rb'))
    n0_gmv_new_total = n0_gmv_new['total'] * (l*(l+1))**2/4
    n0_gmv_new_TTEETE = n0_gmv_new['TTEETE'] * (l*(l+1))**2/4
    n0_gmv_new_TBEB = n0_gmv_new['TBEB'] * (l*(l+1))**2/4

    # SQE, notmh_crossilcFalse
    filename = dir_out+f'/n0/n0_39simpairs_healqest_sqe_lmaxT3000_lmaxP4096_nside2048_notmh_crossilcFalse_resp_from_sims.pkl'
    n0_original_new = pickle.load(open(filename,'rb'))
    n0_original_new_total = n0_original_new['total'] * (l*(l+1))**2/4
    n0_original_new_T1T2 = n0_original_new['T1T2'] * (l*(l+1))**2/4
    n0_original_new_T2T1 = n0_original_new['T2T1'] * (l*(l+1))**2/4
    n0_original_new_EE = n0_original_new['EE'] * (l*(l+1))**2/4
    n0_original_new_TE = n0_original_new['TE'] * (l*(l+1))**2/4
    n0_original_new_ET = n0_original_new['ET'] * (l*(l+1))**2/4
    n0_original_new_TB = n0_original_new['TB'] * (l*(l+1))**2/4
    n0_original_new_BT = n0_original_new['BT'] * (l*(l+1))**2/4
    n0_original_new_EB = n0_original_new['EB'] * (l*(l+1))**2/4
    n0_original_new_BE = n0_original_new['BE'] * (l*(l+1))**2/4

    # GMV, calculated last year
    filename = dir_out+f'/n0/n0_lensing19-20_no_foregrounds_with_ilc_noise/n0_99simpairs_healqest_gmv_lmaxT3000_lmaxP4096_nside2048_cmbonly_resp_from_sims.pkl'
    n0_gmv_old = pickle.load(open(filename,'rb'))
    n0_gmv_old_total = n0_gmv_old['total'] * (l*(l+1))**2/4
    n0_gmv_old_TTEETE = n0_gmv_old['TTEETE'] * (l*(l+1))**2/4
    n0_gmv_old_TBEB = n0_gmv_old['TBEB'] * (l*(l+1))**2/4

    # SQE, calculated last year
    filename = dir_out+f'/n0/n0_lensing19-20_no_foregrounds_with_ilc_noise/n0_99simpairs_healqest_sqe_lmaxT3000_lmaxP4096_nside2048_cmbonly_resp_from_sims.pkl'
    n0_original_old = pickle.load(open(filename,'rb'))
    n0_original_old_total = n0_original_old['total'] * (l*(l+1))**2/4
    n0_original_old_TT = n0_original_old['TT'] * (l*(l+1))**2/4
    n0_original_old_EE = n0_original_old['EE'] * (l*(l+1))**2/4
    n0_original_old_TE = n0_original_old['TE'] * (l*(l+1))**2/4
    n0_original_old_TB = n0_original_old['TB'] * (l*(l+1))**2/4
    n0_original_old_EB = n0_original_old['EB'] * (l*(l+1))**2/4

    # GMV, notmh_crossilcFalse, noiseless
    filename = dir_out+f'/n0/n0_39simpairs_healqest_gmv_lmaxT3000_lmaxP4096_nside2048_notmh_crossilcFalse_cmbonly_resp_from_sims.pkl'
    n0_gmv_new_noiseless = pickle.load(open(filename,'rb'))
    n0_gmv_new_noiseless_total = n0_gmv_new_noiseless['total'] * (l*(l+1))**2/4
    n0_gmv_new_noiseless_TTEETE = n0_gmv_new_noiseless['TTEETE'] * (l*(l+1))**2/4
    n0_gmv_new_noiseless_TBEB = n0_gmv_new_noiseless['TBEB'] * (l*(l+1))**2/4

    # SQE, notmh_crossilcFalse, noiseless
    filename = dir_out+f'/n0/n0_39simpairs_healqest_sqe_lmaxT3000_lmaxP4096_nside2048_notmh_crossilcFalse_cmbonly_resp_from_sims.pkl'
    n0_original_new_noiseless = pickle.load(open(filename,'rb'))
    n0_original_new_noiseless_total = n0_original_new_noiseless['total'] * (l*(l+1))**2/4
    n0_original_new_noiseless_T1T2 = n0_original_new_noiseless['T1T2'] * (l*(l+1))**2/4
    n0_original_new_noiseless_T2T1 = n0_original_new_noiseless['T2T1'] * (l*(l+1))**2/4
    n0_original_new_noiseless_EE = n0_original_new_noiseless['EE'] * (l*(l+1))**2/4
    n0_original_new_noiseless_TE = n0_original_new_noiseless['TE'] * (l*(l+1))**2/4
    n0_original_new_noiseless_ET = n0_original_new_noiseless['ET'] * (l*(l+1))**2/4
    n0_original_new_noiseless_TB = n0_original_new_noiseless['TB'] * (l*(l+1))**2/4
    n0_original_new_noiseless_BT = n0_original_new_noiseless['BT'] * (l*(l+1))**2/4
    n0_original_new_noiseless_EB = n0_original_new_noiseless['EB'] * (l*(l+1))**2/4
    n0_original_new_noiseless_BE = n0_original_new_noiseless['BE'] * (l*(l+1))**2/4

    # GMV, calculated last year, noiseless
    filename = dir_out+f'/n0/n0_lensing19-20_no_foregrounds_with_ilc_noise/n0_99simpairs_healqest_gmv_lmaxT3000_lmaxP4096_nside2048_noiseless_cmbonly_resp_from_sims.pkl'
    n0_gmv_old_noiseless = pickle.load(open(filename,'rb'))
    n0_gmv_old_noiseless_total = n0_gmv_old_noiseless['total'] * (l*(l+1))**2/4
    n0_gmv_old_noiseless_TTEETE = n0_gmv_old_noiseless['TTEETE'] * (l*(l+1))**2/4
    n0_gmv_old_noiseless_TBEB = n0_gmv_old_noiseless['TBEB'] * (l*(l+1))**2/4

    # SQE, calculated last year, noiseless
    filename = dir_out+f'/n0/n0_lensing19-20_no_foregrounds_with_ilc_noise/n0_99simpairs_healqest_sqe_lmaxT3000_lmaxP4096_nside2048_noiseless_cmbonly_resp_from_sims.pkl'
    n0_original_old_noiseless = pickle.load(open(filename,'rb'))
    n0_original_old_noiseless_total = n0_original_old_noiseless['total'] * (l*(l+1))**2/4
    n0_original_old_noiseless_TT = n0_original_old_noiseless['TT'] * (l*(l+1))**2/4
    n0_original_old_noiseless_EE = n0_original_old_noiseless['EE'] * (l*(l+1))**2/4
    n0_original_old_noiseless_TE = n0_original_old_noiseless['TE'] * (l*(l+1))**2/4
    n0_original_old_noiseless_TB = n0_original_old_noiseless['TB'] * (l*(l+1))**2/4
    n0_original_old_noiseless_EB = n0_original_old_noiseless['EB'] * (l*(l+1))**2/4

    ratio_original = n0_original_total/(inv_resp_original * (l*(l+1))**2/4)
    ratio_original_avg = float(np.nanmean(ratio_original))
    ratio_gmv = n0_gmv_total/(inv_resp_gmv * (l*(l+1))**2/4)
    ratio_gmv_avg = float(np.nanmean(ratio_gmv))

    # Theory spectrum
    clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
    ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
    clkk = slpp * (l*(l+1))**2/4

    plt.figure(0)
    plt.clf()
    plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')

    plt.plot(l, n0_original_new_noiseless_T1T2, color='sienna', linestyle='-',label='N0 (SQE new, TT)')
    plt.plot(l, n0_original_new_noiseless_EE, color='mediumorchid', linestyle='-',label='N0 (SQE new, EE)')
    plt.plot(l, n0_original_new_noiseless_TE, color='forestgreen', linestyle='-',label='N0 (SQE new, TE)')
    plt.plot(l, n0_original_new_noiseless_TB, color='gold', linestyle='-',label='N0 (SQE new, TB)')
    plt.plot(l, n0_original_new_noiseless_EB, color='orange', linestyle='-',label='N0 (SQE new, EB)')

    plt.plot(l, n0_original_old_noiseless_TT, color='chocolate', linestyle='--',label='N0 (SQE old, TT)')
    plt.plot(l, n0_original_old_noiseless_EE, color='violet', linestyle='--',label='N0 (SQE old, EE)')
    plt.plot(l, n0_original_old_noiseless_TE, color='darkseagreen', linestyle='--',label='N0 (SQE old, TE)')
    plt.plot(l, n0_original_old_noiseless_TB, color='goldenrod', linestyle='--',label='N0 (SQE old, TB)')
    plt.plot(l, n0_original_old_noiseless_EB, color='burlywood', linestyle='--',label='N0 (SQE old, EB)')

    plt.plot(l, n0_original_new_noiseless_total, color='firebrick', linestyle='-',label='N0 (SQE new, notmh_crossilcFalse)')
    plt.plot(l, n0_original_old_noiseless_total, color='pink', linestyle='--',label='N0 (SQE old, calculated last year)')

    plt.ylabel("$C_\ell^{\kappa\kappa}$")
    plt.xlabel('$\ell$')
    plt.title(f'N0')
    plt.legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize='x-small')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,lmax)
    plt.ylim(1e-8,1e-5)
    if save_fig:
        if resp_from_sims:
            plt.savefig(dir_out+f'/figs/n0_comparison_sqe_notmh_resp_from_sims.png',bbox_inches='tight')

    plt.clf()
    plt.axhline(y=1, color='k', linestyle='--')

    plt.plot(l, n0_original_new_noiseless_EB/n0_original_old_noiseless_EB, color='orange', linestyle='-', label=f'N0 Ratio (SQE EB), notmh_crossilcFalse/from last year')
    plt.plot(l, n0_original_new_noiseless_TB/n0_original_old_noiseless_TB, color='gold', linestyle='-', label=f'N0 Ratio (SQE TB), notmh_crossilcFalse/from last year')
    plt.plot(l, n0_original_new_noiseless_TE/n0_original_old_noiseless_TE, color='forestgreen', linestyle='-', label=f'N0 Ratio (SQE TE), notmh_crossilcFalse/from last year')
    plt.plot(l, n0_original_new_noiseless_EE/n0_original_old_noiseless_EE, color='mediumorchid', linestyle='-', label=f'N0 Ratio (SQE EE), notmh_crossilcFalse/from last year')
    plt.plot(l, n0_original_new_noiseless_T1T2/n0_original_old_noiseless_TT, color='sienna', linestyle='-', label=f'N0 Ratio (SQE TT), notmh_crossilcFalse/from last year')
    plt.plot(l, n0_original_new_noiseless_total/n0_original_old_noiseless_total, color='firebrick', linestyle='-', label=f'N0 Ratio (SQE), notmh_crossilcFalse/from last year')

    plt.xlabel('$\ell$')
    plt.legend(loc='upper left', fontsize='x-small', bbox_to_anchor=(1,0.5))
    plt.xscale('log')
    plt.xlim(10,lmax)
    plt.savefig(dir_out+f'/figs/n0_comparison_sqe_notmh_resp_from_sims_ratio.png',bbox_inches='tight')


    plt.clf()
    plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')

    plt.plot(l, n0_gmv_new_noiseless_total, color='darkblue', linestyle='-',label='N0 (GMV new, notmh_crossilcFalse)')
    plt.plot(l, n0_gmv_new_noiseless_TTEETE, color='forestgreen', linestyle='-',label='N0 (GMV new, TTEETE)')
    plt.plot(l, n0_gmv_new_noiseless_TBEB, color='blueviolet', linestyle='-',label='N0 (GMV new, TBEB)')

    plt.plot(l, n0_gmv_old_noiseless_total, color='powderblue', linestyle='--',label='N0 (GMV old, calculated last year)')
    plt.plot(l, n0_gmv_old_noiseless_TTEETE, color='olive', linestyle='--',label='N0 (GMV TTEETE, calculated last year)')
    plt.plot(l, n0_gmv_old_noiseless_TBEB, color='rebeccapurple', linestyle='--',label='N0 (GMV TBEB, calculated last year)')

    #plt.plot(l, inv_resp_gmv * (l*(l+1))**2/4, color='cornflowerblue', linestyle='--', label='1/R (GMV)')
    #plt.plot(l, inv_resp_gmv_TTEETE * (l*(l+1))**2/4, color='lightgreen', linestyle='--', label='1/R (GMV, TTEETE)')
    #plt.plot(l, inv_resp_gmv_TBEB * (l*(l+1))**2/4, color='thistle', linestyle='--', label='1/R (GMV, TBEB)')

    plt.ylabel("$C_\ell^{\kappa\kappa}$")
    plt.xlabel('$\ell$')
    plt.title(f'N0')
    plt.legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize='x-small')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,lmax)
    plt.ylim(1e-8,1e-5)
    if save_fig:
        if resp_from_sims:
            plt.savefig(dir_out+f'/figs/n0_comparison_gmv_notmh_resp_from_sims.png',bbox_inches='tight')

def get_n0(sims, qetype, config, resp_from_sims, cmbonly=False, append='cmbonly'):
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
        append = 'noiseless_cmbonly'
    if resp_from_sims:
        filename = dir_out+f'/n0/n0_lensing19-20_no_foregrounds_with_ilc_noise/n0_{num}simpairs_healqest_{qetype}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_resp_from_sims.pkl'

    if os.path.isfile(filename):
        n0 = pickle.load(open(filename,'rb'))

    elif qetype == 'gmv':
        # GMV response
        if resp_from_sims:
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
            plm_gmv_ij = np.load(dir_out+f'/outputs_lensing19-20_no_foregrounds_with_ilc_noise/plm_all_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            plm_gmv_TTEETE_ij = np.load(dir_out+f'/outputs_lensing19-20_no_foregrounds_with_ilc_noise/plm_TTEETE_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            plm_gmv_TBEB_ij = np.load(dir_out+f'/outputs_lensing19-20_no_foregrounds_with_ilc_noise/plm_TBEB_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')

            # Now get the ji sims
            plm_gmv_ji = np.load(dir_out+f'/outputs_lensing19-20_no_foregrounds_with_ilc_noise/plm_all_healqest_gmv_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            plm_gmv_TTEETE_ji = np.load(dir_out+f'/outputs_lensing19-20_no_foregrounds_with_ilc_noise/plm_TTEETE_healqest_gmv_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            plm_gmv_TBEB_ji = np.load(dir_out+f'/outputs_lensing19-20_no_foregrounds_with_ilc_noise/plm_TBEB_healqest_gmv_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')

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
        ests = ['TT', 'EE', 'TE', 'TE', 'TB', 'TB', 'EB', 'EB', 'TT']
        resps_original = np.zeros((len(l),len(ests)), dtype=np.complex_)
        inv_resps_original = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
        for i, est in enumerate(ests):
            if resp_from_sims:
                resps_original[:,i] = get_sim_response(est,config,gmv=False,append=append_original,sims=np.append(sims,num+1))
            inv_resps_original[1:,i] = 1/(resps_original)[1:,i]
        resp_original = np.sum(resps_original[:,:-1], axis=1)
        resp_original_TTEETE = resps_original[:,0]+resps_original[:,1]+resps_original[:,2]+resps_original[:,3]
        resp_original_TBEB = resps_original[:,4]+resps_original[:,5]+resps_original[:,6]+resps_original[:,7]
        inv_resp_original = np.zeros_like(l,dtype=np.complex_); inv_resp_original[1:] = 1/(resp_original)[1:]
        inv_resp_original_TTEETE = np.zeros_like(l,dtype=np.complex_); inv_resp_original_TTEETE[1:] = 1/(resp_original_TTEETE)[1:]
        inv_resp_original_TBEB = np.zeros_like(l,dtype=np.complex_); inv_resp_original_TBEB[1:] = 1/(resp_original_TBEB)[1:]

        n0 = {'total':0, 'T1T2':0, 'T2T1':0, 'EE':0, 'TE':0, 'ET':0, 'TB':0, 'BT':0, 'EB':0, 'BE':0, 'TTEETE':0, 'TBEB':0}
        ijij = {'T1T2':0, 'T2T1':0}
        for i, sim1 in enumerate(sims):
            sim2 = sim1 + 1

            # Get the lensed ij sims
            plms_ij = np.zeros((len(np.load(dir_out+f'/outputs_lensing19-20_no_foregrounds_with_ilc_noise/plm_TT_healqest_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')),len(ests)), dtype=np.complex_)
            for i, est in enumerate(ests):
                plms_ij[:,i] = np.load(dir_out+f'/outputs_lensing19-20_no_foregrounds_with_ilc_noise/plm_{est}_healqest_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            plm_total_ij = np.sum(plms_ij[:,:-1], axis=1)

            # Now get the ji sims
            plms_ji = np.zeros((len(np.load(dir_out+f'/outputs_lensing19-20_no_foregrounds_with_ilc_noise/plm_TT_healqest_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')),len(ests)), dtype=np.complex_)
            for i, est in enumerate(ests):
                plms_ji[:,i] = np.load(dir_out+f'/outputs_lensing19-20_no_foregrounds_with_ilc_noise/plm_{est}_healqest_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            plm_total_ji = np.sum(plms_ji[:,:-1], axis=1)

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
            auto_T1T2 = hp.alm2cl(plms_ij[:,0], plms_ij[:,0], lmax=lmax)
            auto_EE = hp.alm2cl(plms_ij[:,1], plms_ij[:,1], lmax=lmax)
            auto_TE = hp.alm2cl(plms_ij[:,2], plms_ij[:,2], lmax=lmax)
            auto_ET = hp.alm2cl(plms_ij[:,3], plms_ij[:,3], lmax=lmax)
            auto_TB = hp.alm2cl(plms_ij[:,4], plms_ij[:,4], lmax=lmax)
            auto_BT = hp.alm2cl(plms_ij[:,5], plms_ij[:,5], lmax=lmax)
            auto_EB = hp.alm2cl(plms_ij[:,6], plms_ij[:,6], lmax=lmax)
            auto_BE = hp.alm2cl(plms_ij[:,7], plms_ij[:,7], lmax=lmax)
            auto_T2T1 = hp.alm2cl(plms_ij[:,8], plms_ij[:,8], lmax=lmax)

            # Get cross spectra <ijji>
            cross = hp.alm2cl(plm_total_ij, plm_total_ji, lmax=lmax)
            cross_TTEETE = hp.alm2cl(plm_TTEETE_ij, plm_TTEETE_ji, lmax=lmax)
            cross_TBEB = hp.alm2cl(plm_TBEB_ij, plm_TBEB_ji, lmax=lmax)
            cross_T1T2 = hp.alm2cl(plms_ij[:,0], plms_ji[:,0], lmax=lmax)
            cross_EE = hp.alm2cl(plms_ij[:,1], plms_ji[:,1], lmax=lmax)
            cross_TE = hp.alm2cl(plms_ij[:,2], plms_ji[:,2], lmax=lmax)
            cross_ET = hp.alm2cl(plms_ij[:,3], plms_ji[:,3], lmax=lmax)
            cross_TB = hp.alm2cl(plms_ij[:,4], plms_ji[:,4], lmax=lmax)
            cross_BT = hp.alm2cl(plms_ij[:,5], plms_ji[:,5], lmax=lmax)
            cross_EB = hp.alm2cl(plms_ij[:,6], plms_ji[:,6], lmax=lmax)
            cross_BE = hp.alm2cl(plms_ij[:,7], plms_ji[:,7], lmax=lmax)
            cross_T2T1 = hp.alm2cl(plms_ij[:,8], plms_ji[:,8], lmax=lmax)

            n0['total'] += auto + cross
            n0['TTEETE'] += auto_TTEETE + cross_TTEETE
            n0['TBEB'] += auto_TBEB + cross_TBEB
            n0['T1T2'] += auto_T1T2 + cross_T1T2
            n0['T2T1'] += auto_T2T1 + cross_T2T1
            n0['EE'] += auto_EE + cross_EE
            n0['TE'] += auto_TE + cross_TE
            n0['ET'] += auto_ET + cross_ET
            n0['TB'] += auto_TB + cross_TB
            n0['BT'] += auto_BT + cross_BT
            n0['EB'] += auto_EB + cross_EB
            n0['BE'] += auto_BE + cross_BE

            ijij['T1T2'] += auto_T1T2
            ijij['T2T1'] += auto_T2T1

        n0['total'] *= 1/num
        n0['TTEETE'] *= 1/num
        n0['TBEB'] *= 1/num
        n0['T1T2'] *= 1/num
        n0['T2T1'] *= 1/num
        n0['EE'] *= 1/num
        n0['TE'] *= 1/num
        n0['ET'] *= 1/num
        n0['TB'] *= 1/num
        n0['BT'] *= 1/num
        n0['EB'] *= 1/num
        n0['BE'] *= 1/num
        ijij['T1T2'] *= 1/num
        ijij['T2T1'] *= 1/num

        with open(filename, 'wb') as f:
            pickle.dump(n0, f)

    else:
        print('Invalid argument qetype.')

    return n0

def get_n1(sims, qetype, config, resp_from_sims, append='cmbonly'):
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
    if resp_from_sims:
        filename = dir_out+f'/n1/n1_lensing19-20_no_foregrounds_with_ilc_noise/n1_{num}simpairs_healqest_{qetype}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_resp_from_sims.pkl'

    if os.path.isfile(filename):
        n1 = pickle.load(open(filename,'rb'))

    elif qetype == 'gmv':
        # GMV response
        if resp_from_sims:
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
            plm_gmv_ij = np.load(dir_out+f'/outputs_lensing19-20_no_foregrounds_with_ilc_noise/plm_all_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_phi1_tqu1tqu2.npy')
            plm_gmv_TTEETE_ij = np.load(dir_out+f'/outputs_lensing19-20_no_foregrounds_with_ilc_noise/plm_TTEETE_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_phi1_tqu1tqu2.npy')
            plm_gmv_TBEB_ij = np.load(dir_out+f'/outputs_lensing19-20_no_foregrounds_with_ilc_noise/plm_TBEB_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_phi1_tqu1tqu2.npy')

            # Now get the ji sims
            plm_gmv_ji = np.load(dir_out+f'/outputs_lensing19-20_no_foregrounds_with_ilc_noise/plm_all_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_phi1_tqu2tqu1.npy')
            plm_gmv_TTEETE_ji = np.load(dir_out+f'/outputs_lensing19-20_no_foregrounds_with_ilc_noise/plm_TTEETE_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_phi1_tqu2tqu1.npy')
            plm_gmv_TBEB_ji = np.load(dir_out+f'/outputs_lensing19-20_no_foregrounds_with_ilc_noise/plm_TBEB_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_phi1_tqu2tqu1.npy')

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
                    resp_from_sims=resp_from_sims,cmbonly=True,append=append)

        n1['total'] -= n0['total']
        n1['TTEETE'] -= n0['TTEETE']
        n1['TBEB'] -= n0['TBEB']

        with open(filename, 'wb') as f:
            pickle.dump(n1, f)

    elif qetype == 'sqe':
        # Get SQE response
        ests = ['TT', 'EE', 'TE', 'TE', 'TB', 'TB', 'EB', 'EB']
        resps_original = np.zeros((len(l),len(ests)), dtype=np.complex_)
        inv_resps_original = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
        for i, est in enumerate(ests):
            if resp_from_sims:
                resps_original[:,i] = get_sim_response(est,config,gmv=False,append=append,sims=np.append(sims,num+1))
            inv_resps_original[1:,i] = 1/(resps_original)[1:,i]
        resp_original = np.sum(resps_original, axis=1)
        inv_resp_original = np.zeros_like(l,dtype=np.complex_); inv_resp_original[1:] = 1/(resp_original)[1:]

        n1 = {'total':0, 'T1T2':0, 'T2T1':0, 'EE':0, 'TE':0, 'ET':0, 'TB':0, 'BT':0, 'EB':0, 'BE':0}
        for i, sim in enumerate(sims):
            # Get the lensed ij sims
            plm_T1T2_ij = np.load(dir_out+f'/outputs_lensing19-20_no_foregrounds_with_ilc_noise/plm_TT_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_phi1_tqu1tqu2.npy')
            plm_EE_ij = np.load(dir_out+f'/outputs_lensing19-20_no_foregrounds_with_ilc_noise/plm_EE_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_phi1_tqu1tqu2.npy')
            plm_TE_ij = np.load(dir_out+f'/outputs_lensing19-20_no_foregrounds_with_ilc_noise/plm_TE_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_phi1_tqu1tqu2.npy')
            plm_ET_ij = np.load(dir_out+f'/outputs_lensing19-20_no_foregrounds_with_ilc_noise/plm_TE_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_phi1_tqu1tqu2.npy')
            plm_TB_ij = np.load(dir_out+f'/outputs_lensing19-20_no_foregrounds_with_ilc_noise/plm_TB_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_phi1_tqu1tqu2.npy')
            plm_BT_ij = np.load(dir_out+f'/outputs_lensing19-20_no_foregrounds_with_ilc_noise/plm_TB_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_phi1_tqu1tqu2.npy')
            plm_EB_ij = np.load(dir_out+f'/outputs_lensing19-20_no_foregrounds_with_ilc_noise/plm_EB_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_phi1_tqu1tqu2.npy')
            plm_BE_ij = np.load(dir_out+f'/outputs_lensing19-20_no_foregrounds_with_ilc_noise/plm_EB_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_phi1_tqu1tqu2.npy')

            # Now get the ji sims
            plm_T1T2_ji = np.load(dir_out+f'/outputs_lensing19-20_no_foregrounds_with_ilc_noise/plm_TT_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_phi1_tqu2tqu1.npy')
            plm_EE_ji = np.load(dir_out+f'/outputs_lensing19-20_no_foregrounds_with_ilc_noise/plm_EE_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_phi1_tqu2tqu1.npy')
            plm_TE_ji = np.load(dir_out+f'/outputs_lensing19-20_no_foregrounds_with_ilc_noise/plm_TE_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_phi1_tqu2tqu1.npy')
            plm_ET_ji = np.load(dir_out+f'/outputs_lensing19-20_no_foregrounds_with_ilc_noise/plm_TE_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_phi1_tqu2tqu1.npy')
            plm_TB_ji = np.load(dir_out+f'/outputs_lensing19-20_no_foregrounds_with_ilc_noise/plm_TB_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_phi1_tqu2tqu1.npy')
            plm_BT_ji = np.load(dir_out+f'/outputs_lensing19-20_no_foregrounds_with_ilc_noise/plm_TB_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_phi1_tqu2tqu1.npy')
            plm_EB_ji = np.load(dir_out+f'/outputs_lensing19-20_no_foregrounds_with_ilc_noise/plm_EB_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_phi1_tqu2tqu1.npy')
            plm_BE_ji = np.load(dir_out+f'/outputs_lensing19-20_no_foregrounds_with_ilc_noise/plm_EB_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_phi1_tqu2tqu1.npy')

            plm_total_ij = plm_T1T2_ij + plm_EE_ij + plm_TE_ij + plm_ET_ij + plm_TB_ij + plm_BT_ij + plm_EB_ij + plm_BE_ij
            # TODO: omg.
            #plm_total_ji = plm_T1T2_ji + plm_EE_ji + plm_TE_ji + plm_ET_ji + plm_TB_ji + plm_BT_ji + plm_EB_ji + plm_BE_ij
            plm_total_ji = plm_T1T2_ji + plm_EE_ji + plm_TE_ji + plm_ET_ji + plm_TB_ji + plm_BT_ji + plm_EB_ji + plm_BE_ji

            # Response correct healqest
            plm_total_ij = hp.almxfl(plm_total_ij,inv_resp_original)
            plm_T1T2_ij = hp.almxfl(plm_T1T2_ij,inv_resps_original[:,0])
            plm_EE_ij = hp.almxfl(plm_EE_ij,inv_resps_original[:,1])
            plm_TE_ij = hp.almxfl(plm_TE_ij,inv_resps_original[:,2])
            plm_ET_ij = hp.almxfl(plm_ET_ij,inv_resps_original[:,3])
            plm_TB_ij = hp.almxfl(plm_TB_ij,inv_resps_original[:,4])
            plm_BT_ij = hp.almxfl(plm_BT_ij,inv_resps_original[:,5])
            plm_EB_ij = hp.almxfl(plm_EB_ij,inv_resps_original[:,6])
            plm_BE_ij = hp.almxfl(plm_BE_ij,inv_resps_original[:,7])

            plm_total_ji = hp.almxfl(plm_total_ji,inv_resp_original)
            plm_T1T2_ji = hp.almxfl(plm_T1T2_ji,inv_resps_original[:,0])
            plm_EE_ji = hp.almxfl(plm_EE_ji,inv_resps_original[:,1])
            plm_TE_ji = hp.almxfl(plm_TE_ji,inv_resps_original[:,2])
            plm_ET_ji = hp.almxfl(plm_ET_ji,inv_resps_original[:,3])
            plm_TB_ji = hp.almxfl(plm_TB_ji,inv_resps_original[:,4])
            plm_BT_ji = hp.almxfl(plm_BT_ji,inv_resps_original[:,5])
            plm_EB_ji = hp.almxfl(plm_EB_ji,inv_resps_original[:,6])
            plm_BE_ji = hp.almxfl(plm_BE_ji,inv_resps_original[:,7])

            # Get ij auto spectra <ijij>
            auto = hp.alm2cl(plm_total_ij, plm_total_ij, lmax=lmax)
            auto_T1T2 = hp.alm2cl(plm_T1T2_ij, plm_T1T2_ij, lmax=lmax)
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
            cross_EE = hp.alm2cl(plm_EE_ij, plm_EE_ji, lmax=lmax)
            cross_TE = hp.alm2cl(plm_TE_ij, plm_TE_ji, lmax=lmax)
            cross_ET = hp.alm2cl(plm_ET_ij, plm_ET_ji, lmax=lmax)
            cross_TB = hp.alm2cl(plm_TB_ij, plm_TB_ji, lmax=lmax)
            cross_BT = hp.alm2cl(plm_BT_ij, plm_BT_ji, lmax=lmax)
            cross_EB = hp.alm2cl(plm_EB_ij, plm_EB_ji, lmax=lmax)
            cross_BE = hp.alm2cl(plm_BE_ij, plm_BE_ji, lmax=lmax)

            n1['total'] += auto + cross
            n1['T1T2'] += auto_T1T2 + cross_T1T2
            n1['EE'] += auto_EE + cross_EE
            n1['TE'] += auto_TE + cross_TE
            n1['ET'] += auto_ET + cross_ET
            n1['TB'] += auto_TB + cross_TB
            n1['BT'] += auto_BT + cross_BT
            n1['EB'] += auto_EB + cross_EB
            n1['BE'] += auto_BE + cross_BE

        n1['total'] *= 1/num
        n1['T1T2'] *= 1/num
        n1['EE'] *= 1/num
        n1['TE'] *= 1/num
        n1['ET'] *= 1/num
        n1['TB'] *= 1/num
        n1['BT'] *= 1/num
        n1['EB'] *= 1/num
        n1['BE'] *= 1/num

        n0 = get_n0(sims=sims,qetype=qetype,config=config,
                    resp_from_sims=resp_from_sims,cmbonly=True,append=append)

        n1['total'] -= n0['total']
        n1['T1T2'] -= n0['T1T2']
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

def get_sim_response(est, config, gmv, append='cmbonly', sims=np.arange(40)+1,
                     filename=None):
    '''
    If gmv, est should be 'TTEETE'/'TBEB'/'all'.
    If not gmv, assume sqe and est should be 'TT'/'EE'/'TE'/'TB'/'EB'.
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
        #fn += f'_{num}sims_lmaxT{lmaxT}_lmaxP{lmaxP}_lmin{lmin}_cltype{cltype}_cmbonly'
        fn += f'_100sims_lmaxT3000_lmaxP4096_nside2048_cmbonly'
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
                plm = np.load(dir_out+f'/outputs_lensing19-20_no_foregrounds_with_ilc_noise/plm_{est}_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_cmbonly.npy')
            else:
                plm = np.load(dir_out+f'/outputs_lensing19-20_no_foregrounds_with_ilc_noise/plm_{est}_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_cmbonly.npy')

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

#compare_n0()
#compare_n1()
#compare_resp()
analyze()
