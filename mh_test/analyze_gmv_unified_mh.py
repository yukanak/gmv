#!/usr/bin/env python3
import numpy as np
import pickle
import healpy as hp
import camb
import os, sys
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import gmv_resp
import utils
import matplotlib.pyplot as plt
import weights
import qest
import wignerd
import resp

def analyze(sims=np.arange(40)+1,n0_n1_sims=np.arange(39)+1,
            config_file='mh_yuka.yaml',
            save_fig=True,
            unl=False,
            n0=True,n1=False,resp_from_sims=True,
            lbins=np.logspace(np.log10(50),np.log10(3000),20)):
    '''
    Compare with N0/N1 subtraction.
    '''
    config = utils.parse_yaml(config_file)
    lmax = config['Lmax']
    lmin = config['lminT']
    lmaxT = config['lmaxT']
    lmaxP = config['lmaxP']
    nside = config['nside']
    dir_out = config['dir_out']
    l = np.arange(0,lmax+1)
    num = len(sims)
    bin_centers = (lbins[:-1] + lbins[1:]) / 2
    digitized = np.digitize(l, lbins)
    if unl is False:
        append = 'mh'
    else:
        append = 'mh_unl'

    # Get SQE response
    ests = ['TT', 'EE', 'TE', 'ET', 'TB', 'BT', 'EB', 'BE']
    resps_original = np.zeros((len(l),len(ests)), dtype=np.complex_)
    inv_resps_original = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
    for i, est in enumerate(ests):
        if resp_from_sims:
            resps_original[:,i] = get_sim_response(est,config,gmv=False,sims=sims)
        else:
            resps_original[:,i] = get_analytic_response(est,config,gmv=False)
        inv_resps_original[1:,i] = 1/(resps_original)[1:,i]
    resp_original = np.sum(resps_original, axis=1)
    inv_resp_original = np.zeros_like(l,dtype=np.complex_); inv_resp_original[1:] = 1/(resp_original)[1:]

    # GMV response
    if resp_from_sims:
        resp_gmv = get_sim_response('all',config,gmv=True,sims=sims)
        resp_gmv_TTEETE = get_sim_response('TTEETE',config,gmv=True,sims=sims)
        resp_gmv_TBEB = get_sim_response('TBEB',config,gmv=True,sims=sims)
    else:
        resp_gmv = get_analytic_response('all',config,gmv=True)
        resp_gmv_TTEETE = get_analytic_response('TTEETE',config,gmv=True)
        resp_gmv_TBEB = get_analytic_response('TBEB',config,gmv=True)
    inv_resp_gmv = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv[1:] = 1./(resp_gmv)[1:]
    inv_resp_gmv_TTEETE = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TTEETE[1:] = 1./(resp_gmv_TTEETE)[1:]
    inv_resp_gmv_TBEB = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TBEB[1:] = 1./(resp_gmv_TBEB)[1:]

    if n0:
        # Get N0 correction (remember these still need (l*(l+1))**2/4 factor to convert to kappa)
        n0_gmv = get_n0(sims=n0_n1_sims,qetype='gmv',config=config,
                        resp_from_sims=resp_from_sims)
        n0_gmv_total = n0_gmv['total'] * (l*(l+1))**2/4
        n0_gmv_TTEETE = n0_gmv['TTEETE'] * (l*(l+1))**2/4
        n0_gmv_TBEB = n0_gmv['TBEB'] * (l*(l+1))**2/4
        n0_original = get_n0(sims=n0_n1_sims,qetype='sqe',config=config,
                             resp_from_sims=resp_from_sims)
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
                        resp_from_sims=resp_from_sims)
        n1_gmv_total = n1_gmv['total'] * (l*(l+1))**2/4
        n1_gmv_TTEETE = n1_gmv['TTEETE'] * (l*(l+1))**2/4
        n1_gmv_TBEB = n1_gmv['TBEB'] * (l*(l+1))**2/4
        n1_original = get_n1(sims=n0_n1_sims,qetype='sqe',config=config,
                             resp_from_sims=resp_from_sims)
        n1_original_total = n1_original['total'] * (l*(l+1))**2/4
        n1_original_TT = n1_original['TT'] * (l*(l+1))**2/4
        n1_original_EE = n1_original['EE'] * (l*(l+1))**2/4
        n1_original_TE = n1_original['TE'] * (l*(l+1))**2/4
        n1_original_ET = n1_original['ET'] * (l*(l+1))**2/4
        n1_original_TB = n1_original['TB'] * (l*(l+1))**2/4
        n1_original_BT = n1_original['BT'] * (l*(l+1))**2/4
        n1_original_EB = n1_original['EB'] * (l*(l+1))**2/4
        n1_original_BT = n1_original['BE'] * (l*(l+1))**2/4

    auto_gmv_all = 0
    auto_original_all = 0
    cross_gmv_all = 0
    cross_original_all = 0
    auto_gmv_debiased_all = 0
    auto_original_debiased_all = 0
    ratio_gmv = np.zeros((len(sims),len(lbins)-1),dtype=np.complex_)
    ratio_original = np.zeros((len(sims),len(lbins)-1),dtype=np.complex_)

    for ii, sim in enumerate(sims):
        # Load GMV plms
        plm_gmv = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
        plm_gmv_TTEETE = np.load(dir_out+f'/plm_TTEETE_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
        plm_gmv_TBEB = np.load(dir_out+f'/plm_TBEB_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')

        # Load SQE plms
        plms_original = np.zeros((len(np.load(dir_out+f'/plm_TT_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')),len(ests)), dtype=np.complex_)
        for i, est in enumerate(ests):
            plms_original[:,i] = np.load(dir_out+f'/plm_{est}_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
        plm_original = np.sum(plms_original, axis=1)

        # Response correct
        plm_gmv_resp_corr = hp.almxfl(plm_gmv,inv_resp_gmv)
        plm_original_resp_corr = hp.almxfl(plm_original,inv_resp_original)

        # Get spectra
        auto_gmv = hp.alm2cl(plm_gmv_resp_corr, plm_gmv_resp_corr, lmax=lmax) * (l*(l+1))**2/4
        auto_original = hp.alm2cl(plm_original_resp_corr, plm_original_resp_corr, lmax=lmax) * (l*(l+1))**2/4

        # N0 and N1 subtract
        if n0 and n1:
            auto_gmv_debiased = auto_gmv - n0_gmv_total - n1_gmv_total
            auto_original_debiased = auto_original - n0_original_total - n1_original_total
        elif n0:
            auto_gmv_debiased = auto_gmv - n0_gmv_total
            auto_original_debiased = auto_original - n0_original_total

        # Sum the auto spectra
        auto_gmv_all += auto_gmv
        auto_original_all += auto_original
        if n0:
            auto_gmv_debiased_all += auto_gmv_debiased
            auto_original_debiased_all += auto_original_debiased

        if not unl:
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
                binned_auto_input = [auto_input[digitized == i].mean() for i in range(1, len(lbins))]
                # Get ratio
                ratio_gmv[ii,:] = np.array(binned_auto_gmv_debiased) / np.array(binned_auto_input)
                ratio_original[ii,:] = np.array(binned_auto_original_debiased) / np.array(binned_auto_input)

    # Average
    auto_gmv_avg = auto_gmv_all / num
    auto_original_avg = auto_original_all / num
    if n0:
        auto_gmv_debiased_avg = auto_gmv_debiased_all / num
        auto_original_debiased_avg = auto_original_debiased_all / num
        # If debiasing, get the ratio points, error bars for the ratio points, and bin
        errorbars_gmv = np.std(ratio_gmv,axis=0)/np.sqrt(num)
        errorbars_original = np.std(ratio_original,axis=0)/np.sqrt(num)
        ratio_gmv = np.mean(ratio_gmv,axis=0)
        ratio_original = np.mean(ratio_original,axis=0)
        # Bin!
        binned_auto_gmv_debiased_avg = [auto_gmv_debiased_avg[digitized == i].mean() for i in range(1, len(lbins))]
        binned_auto_original_debiased_avg = [auto_original_debiased_avg[digitized == i].mean() for i in range(1, len(lbins))]

    if not unl:
        # Average the cross with input spectra
        cross_gmv_avg = cross_gmv_all / num
        cross_original_avg = cross_original_all / num

    # Theory spectrum
    clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
    ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
    clkk = slpp * (l*(l+1))**2/4

    # Plot
    plt.figure(0)
    plt.clf()
    plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')

    plt.plot(l, auto_gmv_avg, color='darkblue', linestyle='-', label="Auto Spectrum (GMV)")
    plt.plot(l, auto_original_avg, color='firebrick', linestyle='-', label=f'Auto Spectrum (SQE)')

    plt.plot(l, inv_resp_gmv * (l*(l+1))**2/4, color='cornflowerblue', linestyle='--', label='1/R (GMV)')
    plt.plot(l, inv_resp_original * (l*(l+1))**2/4, color='lightcoral', linestyle='--', label='1/R (Original)')

    plt.ylabel("$C_\ell^{\kappa\kappa}$")
    plt.xlabel('$\ell$')
    plt.title(f'Spectra Averaged over {num} Sims, MH')
    plt.legend(loc='upper left', fontsize='x-small', bbox_to_anchor=(1,0.5))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,lmax)
    plt.ylim(1e-9,1e-6)
    if save_fig:
        plt.savefig(dir_out+f'/figs/{num}_sims_comparison_{append}_resp_from_sims.png',bbox_inches='tight')

    if n0:
        plt.figure(0)
        plt.clf()
        plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')

        plt.plot(l, auto_gmv_debiased_avg, color='cornflowerblue', linestyle='-', label="Auto Spectrum (GMV)")
        plt.plot(l, auto_original_debiased_avg, color='lightcoral', linestyle='-', label=f'Auto Spectrum (SQE)')

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
        if save_fig:
            if n1:
                plt.savefig(dir_out+f'/figs/{num}_sims_comparison_{append}_resp_from_sims_n0n1subtracted.png',bbox_inches='tight')
            else:
                plt.savefig(dir_out+f'/figs/{num}_sims_comparison_{append}_resp_from_sims_n0subtracted.png',bbox_inches='tight')

        plt.figure(1)
        plt.clf()
        # Ratios with error bars
        plt.axhline(y=1, color='k', linestyle='--')
        plt.errorbar(bin_centers,ratio_gmv,yerr=errorbars_gmv,color='darkblue', marker='o', linestyle='None', ms=3, label="Ratio GMV/Input")
        plt.errorbar(bin_centers,ratio_original,yerr=errorbars_original,color='firebrick', marker='o', linestyle='None', ms=3, label="Ratio Original/Input")
        plt.xlabel('$\ell$')
        plt.title(f'Spectra Averaged over {num} Sims')
        plt.legend(loc='lower left', fontsize='x-small')
        plt.xscale('log')
        plt.ylim(0.99,1.01)
        plt.xlim(10,lmax)
        if save_fig:
            if n1:
                plt.savefig(dir_out+f'/figs/{num}_sims_comparison_{append}_resp_from_sims_n0n1subtracted_binnedratio.png',bbox_inches='tight')
            else:
                plt.savefig(dir_out+f'/figs/{num}_sims_comparison_{append}_resp_from_sims_n0subtracted_binnedratio.png',bbox_inches='tight')

    plt.figure(2)
    plt.clf()
    # Looking at cross with input spectra
    plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')
    plt.plot(l, cross_original_avg, color='firebrick', linestyle='-', label=f'Cross Spectrum with Input (SQE)')
    plt.plot(l, cross_gmv_avg, color='darkblue', linestyle='-', label="Cross Spectrum with Input (GMV)")
    plt.ylabel("$C_\ell^{\kappa\kappa}$")
    plt.xlabel('$\ell$')
    plt.title(f'Spectra Averaged over {num} Sims')
    plt.legend(loc='upper right', fontsize='small')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,lmax)
    plt.ylim(1e-9,1e-6)
    if save_fig:
        plt.savefig(dir_out+f'/figs/{num}_sims_cross_with_input_comparison_{append}_resp_from_sims.png')

    plt.figure(3)
    plt.clf()
    r_original = cross_original_avg/clkk
    r_gmv = cross_gmv_avg/clkk
    plt.axhline(y=1, color='k', linestyle='--')
    plt.plot(l, r_original, color='firebrick', linestyle='-', label="Ratio of Cross Spectrum with Input/$C_\ell^{\kappa\kappa}$ (SQE)")
    plt.plot(l, r_gmv, color='darkblue', linestyle='-', label="Ratio Cross Spectrum with Input/$C_\ell^{\kappa\kappa}$ (GMV)")
    plt.xlabel('$\ell$')
    plt.title(f'Spectra Averaged over {num} Sims')
    plt.legend(loc='upper right', fontsize='small')
    plt.xscale('log')
    plt.xlim(10,lmax)
    plt.ylim(0.95,1.05)
    if save_fig:
        plt.savefig(dir_out+f'/figs/{num}_sims_cross_with_input_comparison_{append}_resp_from_sims_ratio.png')

    if n0:
        plt.figure(4)
        plt.clf()
        # Getting GMV improvement
        plt.plot(l, (n0_gmv_total/n0_original_total)-1, color='maroon', linestyle='-')
        plt.ylabel("$(N_0^{GMV}/N_0^{healqest})-1$")
        plt.xlabel('$\ell$')
        plt.title('$N_0$ Comparison, Total')
        plt.xlim(10,lmax)
        plt.ylim(-0.2,0.2)
        if save_fig:
            plt.savefig(dir_out+f'/figs/n0_comparison_frac_diff_total_{append}_resp_from_sims.png',bbox_inches='tight')

def compare_resp(config_file='mh_yuka.yaml',
                 save_fig=True):
    config = utils.parse_yaml(config_file)
    lmax = config['Lmax']
    lmin = config['lminT']
    dir_out = config['dir_out']
    l = np.arange(0,lmax+1)

    # SQE response
    ests = ['TT', 'EE', 'TE', 'ET', 'TB', 'BT', 'EB', 'BE']
    resps_original = np.zeros((len(l),len(ests)), dtype=np.complex_)
    inv_resps_original = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
    for i, est in enumerate(ests):
        resps_original[:,i] = get_analytic_response(est,config,gmv=False)
        inv_resps_original[1:,i] = 1/(resps_original)[1:,i]
    resp_original = np.sum(resps_original, axis=1)
    inv_resp_original = np.zeros_like(l,dtype=np.complex_); inv_resp_original[1:] = 1/(resp_original)[1:]

    # GMV response
    resp_gmv = get_analytic_response('all',config,gmv=True)
    resp_gmv_TTEETE = get_analytic_response('TTEETE',config,gmv=True)
    resp_gmv_TBEB = get_analytic_response('TBEB',config,gmv=True)
    inv_resp_gmv = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv[1:] = 1./(resp_gmv)[1:]
    inv_resp_gmv_TTEETE = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TTEETE[1:] = 1./(resp_gmv_TTEETE)[1:]
    inv_resp_gmv_TBEB = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TBEB[1:] = 1./(resp_gmv_TBEB)[1:]

    # SQE response from before (2019/2020 ILC noise curves that are NOT correlated between frequencies, no foregrounds)
    old_ests = ['TT', 'EE', 'TE', 'TB', 'EB']
    resps_original_old = np.zeros((len(l),5), dtype=np.complex_)
    inv_resps_original_old = np.zeros((len(l),5) ,dtype=np.complex_)
    for i, est in enumerate(old_ests):
        resps_original_old[:,i] = np.load(dir_out+f'/resp/an_resp_sqe_est{est}_lmaxT3000_lmaxP4096_lmin300_cltypelcmb_added_noise_from_file.npy')
        inv_resps_original_old[1:,i] = 1/(resps_original_old)[1:,i]
    resp_original_old = resps_original_old[:,0]+resps_original_old[:,1]+2*resps_original_old[:,2]+2*resps_original_old[:,3]+2*resps_original_old[:,4]
    inv_resp_original_old = np.zeros_like(l,dtype=np.complex_); inv_resp_original_old[1:] = 1/(resp_original_old)[1:]

    # GMV response from before (2019/2020 ILC noise curves that are NOT correlated between frequencies, no foregrounds)
    resp_gmv_old = np.load(dir_out+f'/resp/an_resp_gmv_estall_lmaxT3000_lmaxP4096_lmin300_cltypelcmb_added_noise_from_file.npy')
    resp_gmv_TTEETE_old = resp_gmv_old[:,1]
    resp_gmv_TBEB_old = resp_gmv_old[:,2]
    resp_gmv_old = resp_gmv_old[:,3]
    inv_resp_gmv_old = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_old[1:] = 1./(resp_gmv_old)[1:]
    inv_resp_gmv_TTEETE_old = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TTEETE_old[1:] = 1./(resp_gmv_TTEETE_old)[1:]
    inv_resp_gmv_TBEB_old = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TBEB_old[1:] = 1./(resp_gmv_TBEB_old)[1:]

    # Theory spectrum
    clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
    ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
    clkk = slpp * (l*(l+1))**2/4

    plt.figure(0)
    plt.clf()
    plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')

    plt.plot(l, inv_resp_original * (l*(l+1))**2/4, color='firebrick', linestyle='-', label='$1/R$ (SQE)')
    #plt.plot(l, inv_resps_original[:,0] * (l*(l+1))**2/4, color='sienna', linestyle='-', label='$1/R$ (SQE, TT)')
    #plt.plot(l, inv_resps_original[:,1] * (l*(l+1))**2/4, color='mediumorchid', linestyle='-', label='$1/R$ (SQE, EE)')
    #plt.plot(l, 0.5*inv_resps_original[:,2] * (l*(l+1))**2/4, color='forestgreen', linestyle='-', label='$1/(2R)$ (SQE, TE)')
    #plt.plot(l, 0.5*inv_resps_original[:,4] * (l*(l+1))**2/4, color='gold', linestyle='-', label='$1/(2R)$ (SQE, TB)')
    #plt.plot(l, 0.5*inv_resps_original[:,6] * (l*(l+1))**2/4, color='orange', linestyle='-', label='$1/(2R$) (SQE, EB)')

    plt.plot(l, inv_resp_original_old * (l*(l+1))**2/4, color='lightcoral', linestyle='--', label='$1/R$ (SQE, old)')
    #plt.plot(l, inv_resps_original_old[:,0] * (l*(l+1))**2/4, color='sandybrown', linestyle='--', label='$1/R$ (SQE, TT old)')
    #plt.plot(l, inv_resps_original_old[:,1] * (l*(l+1))**2/4, color='plum', linestyle='--', label='$1/R$ (SQE, EE old)')
    #plt.plot(l, 0.5*inv_resps_original_old[:,2] * (l*(l+1))**2/4, color='lightgreen', linestyle='--', label='$1/(2R)$ (SQE, TE old)')
    #plt.plot(l, 0.5*inv_resps_original_old[:,3] * (l*(l+1))**2/4, color='palegoldenrod', linestyle='--', label='$1/(2R)$ (SQE, TB old)')
    #plt.plot(l, 0.5*inv_resps_original_old[:,4] * (l*(l+1))**2/4, color='bisque', linestyle='--', label='$1/(2R$) (SQE, EB old)')

    plt.plot(l, inv_resp_gmv * (l*(l+1))**2/4, color='darkblue', linestyle='-', label='$1/R$ (GMV)')
    #plt.plot(l, inv_resp_gmv_TTEETE * (l*(l+1))**2/4, color='forestgreen', linestyle='-', label='1/R (GMV, TTEETE)')
    #plt.plot(l, inv_resp_gmv_TBEB * (l*(l+1))**2/4, color='blueviolet', linestyle='-', label='1/R (GMV, TBEB)')

    plt.plot(l, inv_resp_gmv_old * (l*(l+1))**2/4, color='cornflowerblue', linestyle='--', label='$1/R$ (GMV, old)')
    #plt.plot(l, inv_resp_gmv_TTEETE_old * (l*(l+1))**2/4, color='lightgreen', linestyle='--', label='1/R (GMV, TTEETE old)')
    #plt.plot(l, inv_resp_gmv_TBEB_old * (l*(l+1))**2/4, color='thistle', linestyle='--', label='1/R (GMV, TBEB old)')

    plt.ylabel("$C_\ell^{\kappa\kappa}$")
    plt.xlabel('$\ell$')
    plt.title('$1/R$')
    plt.legend(loc='center left', fontsize='small', bbox_to_anchor=(1,0.5))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,lmax)
    #plt.ylim(8e-9,1e-5)
    plt.ylim(8e-9,1e-6)
    if save_fig:
        plt.savefig(dir_out+f'/figs/mh_response_comparison.png',bbox_inches='tight')
        #plt.savefig(dir_out+f'/figs/mh_response_comparison_sqe_only.png',bbox_inches='tight')
        #plt.savefig(dir_out+f'/figs/mh_response_comparison_gmv_only.png',bbox_inches='tight')

def get_n0(sims, qetype, config, resp_from_sims, cmbonly=False):
    '''
    Get N0 bias. qetype should be 'gmv' or 'sqe'.
    Returns dictionary containing keys 'total', 'TTEETE', and 'TBEB' for GMV.
    Similarly for SQE.
    '''
    lmax = config['Lmax']
    lmaxT = config['lmaxT']
    lmaxP = config['lmaxP']
    nside = config['nside']
    cltype = config['cltype']
    sl = {ee:config['cls'][cltype][ee] for ee in config['cls'][cltype].keys()}
    dir_out = config['dir_out']
    l = np.arange(0,lmax+1)
    num = len(sims)
    append = 'mh'
    if cmbonly:
        append += '_cmbonly'
    if resp_from_sims:
        filename = dir_out+f'/n0/n0_{num}simpairs_healqest_{qetype}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_resp_from_sims.pkl'
    else:
        filename = dir_out+f'/n0/n0_{num}simpairs_healqest_{qetype}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.pkl'

    if os.path.isfile(filename):
        n0 = pickle.load(open(filename,'rb'))

    elif qetype == 'gmv':
        # GMV response
        if resp_from_sims:
            resp_gmv = get_sim_response('all',config,gmv=True,sims=np.append(sims,num+1))
            resp_gmv_TTEETE = get_sim_response('TTEETE',config,gmv=True,sims=np.append(sims,num+1))
            resp_gmv_TBEB = get_sim_response('TBEB',config,gmv=True,sims=np.append(sims,num+1))
        else:
            resp_gmv = get_analytic_response('all',config,gmv=True)
            resp_gmv_TTEETE = get_analytic_response('TTEETE',config,gmv=True)
            resp_gmv_TBEB = get_analytic_response('TBEB',config,gmv=True)
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
        ests = ['TT', 'EE', 'TE', 'ET', 'TB', 'BT', 'EB', 'BE']
        resps_original = np.zeros((len(l),len(ests)), dtype=np.complex_)
        inv_resps_original = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
        for i, est in enumerate(ests):
            if resp_from_sims:
                resps_original[:,i] = get_sim_response(est,config,gmv=False,sims=np.append(sims,num+1))
            else:
                resps_original[:,i] = get_analytic_response(est,config,gmv=False)
            inv_resps_original[1:,i] = 1/(resps_original)[1:,i]
        resp_original = np.sum(resps_original, axis=1)
        inv_resp_original = np.zeros_like(l,dtype=np.complex_); inv_resp_original[1:] = 1/(resp_original)[1:]

        n0 = {'total':0, 'TT':0, 'EE':0, 'TE':0, 'ET':0, 'TB':0, 'BT':0, 'EB':0, 'BE':0, 'TTEETE':0, 'TBEB':0}
        for i, sim1 in enumerate(sims):
            sim2 = sim1 + 1

            # Get the lensed ij sims
            plm_TT_ij = np.load(dir_out+f'/plm_TT_healqest_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            plm_EE_ij = np.load(dir_out+f'/plm_EE_healqest_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            plm_TE_ij = np.load(dir_out+f'/plm_TE_healqest_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            plm_ET_ij = np.load(dir_out+f'/plm_ET_healqest_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            plm_TB_ij = np.load(dir_out+f'/plm_TB_healqest_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            plm_BT_ij = np.load(dir_out+f'/plm_BT_healqest_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            plm_EB_ij = np.load(dir_out+f'/plm_EB_healqest_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            plm_BE_ij = np.load(dir_out+f'/plm_BE_healqest_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')

            # Now get the ji sims
            plm_TT_ji = np.load(dir_out+f'/plm_TT_healqest_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            plm_EE_ji = np.load(dir_out+f'/plm_EE_healqest_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            plm_TE_ji = np.load(dir_out+f'/plm_TE_healqest_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            plm_ET_ji = np.load(dir_out+f'/plm_ET_healqest_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            plm_TB_ji = np.load(dir_out+f'/plm_TB_healqest_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            plm_BT_ji = np.load(dir_out+f'/plm_BT_healqest_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            plm_EB_ji = np.load(dir_out+f'/plm_EB_healqest_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            plm_BE_ji = np.load(dir_out+f'/plm_BE_healqest_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')

            # Eight estimators!!!
            plm_total_ij = plm_TT_ij + plm_EE_ij + plm_TE_ij + plm_ET_ij + plm_TB_ij + plm_BT_ij + plm_EB_ij + plm_BE_ij
            plm_total_ji = plm_TT_ji + plm_EE_ji + plm_TE_ji + plm_ET_ji + plm_TB_ji + plm_BT_ji + plm_EB_ji + plm_BE_ij
            plm_TTEETE_ij = plm_TT_ij + plm_EE_ij + plm_TE_ij + plm_ET_ij
            plm_TTEETE_ji = plm_TT_ji + plm_EE_ji + plm_TE_ji + plm_ET_ji
            plm_TBEB_ij = plm_TB_ij + plm_BT_ij + plm_EB_ij + plm_BE_ij
            plm_TBEB_ji = plm_TB_ji + plm_BT_ji + plm_EB_ji + plm_BE_ij

            # Response correct healqest
            plm_total_ij = hp.almxfl(plm_total_ij,inv_resp_original)
            plm_TTEETE_ij = hp.almxfl(plm_TTEETE_ij,inv_resp_original_TTEETE)
            plm_TBEB_ij = hp.almxfl(plm_TBEB_ij,inv_resp_original_TBEB)
            plm_TT_ij = hp.almxfl(plm_TT_ij,inv_resps_original[:,0])
            plm_EE_ij = hp.almxfl(plm_EE_ij,inv_resps_original[:,1])
            plm_TE_ij = hp.almxfl(plm_TE_ij,inv_resps_original[:,2])
            plm_ET_ij = hp.almxfl(plm_ET_ij,inv_resps_original[:,3])
            plm_TB_ij = hp.almxfl(plm_TB_ij,inv_resps_original[:,4])
            plm_BT_ij = hp.almxfl(plm_BT_ij,inv_resps_original[:,5])
            plm_EB_ij = hp.almxfl(plm_EB_ij,inv_resps_original[:,6])
            plm_BE_ij = hp.almxfl(plm_BE_ij,inv_resps_original[:,7])

            plm_total_ji = hp.almxfl(plm_total_ji,inv_resp_original)
            plm_TTEETE_ji = hp.almxfl(plm_TTEETE_ji,inv_resp_original_TTEETE)
            plm_TBEB_ji = hp.almxfl(plm_TBEB_ji,inv_resp_original_TBEB)
            plm_TT_ji = hp.almxfl(plm_TT_ji,inv_resps_original[:,0])
            plm_EE_ji = hp.almxfl(plm_EE_ji,inv_resps_original[:,1])
            plm_TE_ji = hp.almxfl(plm_TE_ji,inv_resps_original[:,2])
            plm_ET_ji = hp.almxfl(plm_ET_ji,inv_resps_original[:,3])
            plm_TB_ji = hp.almxfl(plm_TB_ji,inv_resps_original[:,4])
            plm_BT_ji = hp.almxfl(plm_BT_ji,inv_resps_original[:,5])
            plm_EB_ji = hp.almxfl(plm_EB_ji,inv_resps_original[:,6])
            plm_BE_ji = hp.almxfl(plm_BE_ji,inv_resps_original[:,7])

            # Get ij auto spectra <ijij>
            auto = hp.alm2cl(plm_total_ij, plm_total_ij, lmax=lmax)
            auto_TTEETE = hp.alm2cl(plm_TTEETE_ij, plm_TTEETE_ij, lmax=lmax)
            auto_TBEB = hp.alm2cl(plm_TBEB_ij, plm_TBEB_ij, lmax=lmax)
            auto_TT = hp.alm2cl(plm_TT_ij, plm_TT_ij, lmax=lmax)
            auto_EE = hp.alm2cl(plm_EE_ij, plm_EE_ij, lmax=lmax)
            auto_TE = hp.alm2cl(plm_TE_ij, plm_TE_ij, lmax=lmax)
            auto_ET = hp.alm2cl(plm_ET_ij, plm_ET_ij, lmax=lmax)
            auto_TB = hp.alm2cl(plm_TB_ij, plm_TB_ij, lmax=lmax)
            auto_BT = hp.alm2cl(plm_BT_ij, plm_BT_ij, lmax=lmax)
            auto_EB = hp.alm2cl(plm_EB_ij, plm_EB_ij, lmax=lmax)
            auto_BE = hp.alm2cl(plm_BE_ij, plm_BE_ij, lmax=lmax)

            # Get cross spectra <ijji>
            cross = hp.alm2cl(plm_total_ij, plm_total_ji, lmax=lmax)
            cross_TTEETE = hp.alm2cl(plm_TTEETE_ij, plm_TTEETE_ji, lmax=lmax)
            cross_TBEB = hp.alm2cl(plm_TBEB_ij, plm_TBEB_ji, lmax=lmax)
            cross_TT = hp.alm2cl(plm_TT_ij, plm_TT_ji, lmax=lmax)
            cross_EE = hp.alm2cl(plm_EE_ij, plm_EE_ji, lmax=lmax)
            cross_TE = hp.alm2cl(plm_TE_ij, plm_TE_ji, lmax=lmax)
            cross_ET = hp.alm2cl(plm_ET_ij, plm_ET_ji, lmax=lmax)
            cross_TB = hp.alm2cl(plm_TB_ij, plm_TB_ji, lmax=lmax)
            cross_BT = hp.alm2cl(plm_BT_ij, plm_BT_ji, lmax=lmax)
            cross_EB = hp.alm2cl(plm_EB_ij, plm_EB_ji, lmax=lmax)
            cross_BE = hp.alm2cl(plm_BE_ij, plm_BE_ji, lmax=lmax)

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

def get_n1(sims, qetype, config, resp_from_sims):
    '''
    Get N1 bias. qetype should be 'gmv' or 'sqe'.
    Returns dictionary containing keys 'total', 'TTEETE', and 'TBEB' for GMV.
    Similarly for SQE.
    '''
    lmax = config['Lmax']
    lmaxT = config['lmaxT']
    lmaxP = config['lmaxP']
    nside = config['nside']
    cltype = config['cltype']
    sl = {ee:config['cls'][cltype][ee] for ee in config['cls'][cltype].keys()}
    dir_out = config['dir_out']
    l = np.arange(0,lmax+1)
    num = len(sims)
    append = 'mh'
    if resp_from_sims:
        filename = dir_out+f'/n1/n1_{num}simpairs_healqest_{qetype}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_resp_from_sims.pkl'
    else:
        filename = dir_out+f'/n1/n1_{num}simpairs_healqest_{qetype}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.pkl'

    if os.path.isfile(filename):
        n1 = pickle.load(open(filename,'rb'))

    elif qetype == 'gmv':
        # GMV response
        if resp_from_sims:
            resp_gmv = get_sim_response('all',config,gmv=True,sims=np.append(sims,num+1))
            resp_gmv_TTEETE = get_sim_response('TTEETE',config,gmv=True,sims=np.append(sims,num+1))
            resp_gmv_TBEB = get_sim_response('TBEB',config,gmv=True,sims=np.append(sims,num+1))
        else:
            resp_gmv = get_analytic_response('all',config,gmv=True)
            resp_gmv_TTEETE = get_analytic_response('TTEETE',config,gmv=True)
            resp_gmv_TBEB = get_analytic_response('TBEB',config,gmv=True)
        inv_resp_gmv = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv[1:] = 1./(resp_gmv)[1:]
        inv_resp_gmv_TTEETE = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TTEETE[1:] = 1./(resp_gmv_TTEETE)[1:]
        inv_resp_gmv_TBEB = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TBEB[1:] = 1./(resp_gmv_TBEB)[1:]

        n1 = {'total':0, 'TTEETE':0, 'TBEB':0}
        for i, sim in enumerate(sims):
            # These are reconstructions using sims that were lensed with the same phi but different CMB realizations, no foregrounds
            # Get the lensed ij sims
            plm_gmv_ij = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_mh_cmbonly_phi1_tqu1tqu2.npy')
            plm_gmv_TTEETE_ij = np.load(dir_out+f'/plm_TTEETE_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_mh_cmbonly_phi1_tqu1tqu2.npy')
            plm_gmv_TBEB_ij = np.load(dir_out+f'/plm_TBEB_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_mh_cmbonly_phi1_tqu1tqu2.npy')

            # Now get the ji sims
            plm_gmv_ji = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_mh_cmbonly_phi1_tqu2tqu1.npy')
            plm_gmv_TTEETE_ji = np.load(dir_out+f'/plm_TTEETE_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_mh_cmbonly_phi1_tqu2tqu1.npy')
            plm_gmv_TBEB_ji = np.load(dir_out+f'/plm_TBEB_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_mh_cmbonly_phi1_tqu2tqu1.npy')

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
                    resp_from_sims=resp_from_sims,cmbonly=True)

        n1['total'] -= n0['total']
        n1['TTEETE'] -= n0['TTEETE']
        n1['TBEB'] -= n0['TBEB']

        with open(filename, 'wb') as f:
            pickle.dump(n1, f)

    elif qetype == 'sqe':
        # SQE response
        ests = ['TT', 'EE', 'TE', 'ET', 'TB', 'BT', 'EB', 'BE']
        resps_original = np.zeros((len(l),len(ests)), dtype=np.complex_)
        inv_resps_original = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
        for i, est in enumerate(ests):
            if resp_from_sims:
                resps_original[:,i] = get_sim_response(est,config,gmv=False,sims=np.append(sims,num+1))
            else:
                resps_original[:,i] = get_analytic_response(est,config,gmv=False)
            inv_resps_original[1:,i] = 1/(resps_original)[1:,i]
        resp_original = np.sum(resps_original, axis=1)
        inv_resp_original = np.zeros_like(l,dtype=np.complex_); inv_resp_original[1:] = 1/(resp_original)[1:]

        n1 = {'total':0, 'TT':0, 'EE':0, 'TE':0, 'ET':0, 'TB':0, 'BT':0, 'EB':0, 'BE':0}
        for i, sim in enumerate(sims):
            # Get the lensed ij sims
            plm_TT_ij = np.load(dir_out+f'/plm_TT_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_mh_cmbonly_phi1_tqu1tqu2.npy')
            plm_EE_ij = np.load(dir_out+f'/plm_EE_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_mh_cmbonly_phi1_tqu1tqu2.npy')
            plm_TE_ij = np.load(dir_out+f'/plm_TE_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_mh_cmbonly_phi1_tqu1tqu2.npy')
            plm_ET_ij = np.load(dir_out+f'/plm_ET_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_mh_cmbonly_phi1_tqu1tqu2.npy')
            plm_TB_ij = np.load(dir_out+f'/plm_TB_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_mh_cmbonly_phi1_tqu1tqu2.npy')
            plm_BT_ij = np.load(dir_out+f'/plm_BT_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_mh_cmbonly_phi1_tqu1tqu2.npy')
            plm_EB_ij = np.load(dir_out+f'/plm_EB_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_mh_cmbonly_phi1_tqu1tqu2.npy')
            plm_BE_ij = np.load(dir_out+f'/plm_BE_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_mh_cmbonly_phi1_tqu1tqu2.npy')

            # Now get the ji sims
            plm_TT_ji = np.load(dir_out+f'/plm_TT_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_mh_cmbonly_phi1_tqu2tqu1.npy')
            plm_EE_ji = np.load(dir_out+f'/plm_EE_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_mh_cmbonly_phi1_tqu2tqu1.npy')
            plm_TE_ji = np.load(dir_out+f'/plm_TE_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_mh_cmbonly_phi1_tqu2tqu1.npy')
            plm_ET_ji = np.load(dir_out+f'/plm_ET_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_mh_cmbonly_phi1_tqu2tqu1.npy')
            plm_TB_ji = np.load(dir_out+f'/plm_TB_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_mh_cmbonly_phi1_tqu2tqu1.npy')
            plm_BT_ji = np.load(dir_out+f'/plm_BT_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_mh_cmbonly_phi1_tqu2tqu1.npy')
            plm_EB_ji = np.load(dir_out+f'/plm_EB_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_mh_cmbonly_phi1_tqu2tqu1.npy')
            plm_BE_ji = np.load(dir_out+f'/plm_BE_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_mh_cmbonly_phi1_tqu2tqu1.npy')

            # Eight estimators!!!
            plm_total_ij = plm_TT_ij + plm_EE_ij + plm_TE_ij + plm_ET_ij + plm_TB_ij + plm_BT_ij + plm_EB_ij + plm_BE_ij
            plm_total_ji = plm_TT_ji + plm_EE_ji + plm_TE_ji + plm_ET_ji + plm_TB_ji + plm_BT_ji + plm_EB_ji + plm_BE_ij

            # Response correct healqest
            plm_total_ij = hp.almxfl(plm_total_ij,inv_resp_original)
            plm_TT_ij = hp.almxfl(plm_TT_ij,inv_resps_original[:,0])
            plm_EE_ij = hp.almxfl(plm_EE_ij,inv_resps_original[:,1])
            plm_TE_ij = hp.almxfl(plm_TE_ij,inv_resps_original[:,2])
            plm_ET_ij = hp.almxfl(plm_ET_ij,inv_resps_original[:,3])
            plm_TB_ij = hp.almxfl(plm_TB_ij,inv_resps_original[:,4])
            plm_BT_ij = hp.almxfl(plm_BT_ij,inv_resps_original[:,5])
            plm_EB_ij = hp.almxfl(plm_EB_ij,inv_resps_original[:,6])
            plm_BE_ij = hp.almxfl(plm_BE_ij,inv_resps_original[:,7])

            plm_total_ji = hp.almxfl(plm_total_ji,inv_resp_original)
            plm_TT_ji = hp.almxfl(plm_TT_ji,inv_resps_original[:,0])
            plm_EE_ji = hp.almxfl(plm_EE_ji,inv_resps_original[:,1])
            plm_TE_ji = hp.almxfl(plm_TE_ji,inv_resps_original[:,2])
            plm_ET_ji = hp.almxfl(plm_ET_ji,inv_resps_original[:,3])
            plm_TB_ji = hp.almxfl(plm_TB_ji,inv_resps_original[:,4])
            plm_BT_ji = hp.almxfl(plm_BT_ji,inv_resps_original[:,5])
            plm_EB_ji = hp.almxfl(plm_EB_ji,inv_resps_original[:,6])
            plm_BE_ji = hp.almxfl(plm_BE_ji,inv_resps_original[:,7])

            # Get ij auto spectra <ijij>
            auto = hp.alm2cl(plm_total_ij, plm_total_ij, lmax=lmax)
            auto_TT = hp.alm2cl(plm_TT_ij, plm_TT_ij, lmax=lmax)
            auto_EE = hp.alm2cl(plm_EE_ij, plm_EE_ij, lmax=lmax)
            auto_TE = hp.alm2cl(plm_TE_ij, plm_TE_ij, lmax=lmax)
            auto_ET = hp.alm2cl(plm_ET_ij, plm_ET_ij, lmax=lmax)
            auto_TB = hp.alm2cl(plm_TB_ij, plm_TB_ij, lmax=lmax)
            auto_BT = hp.alm2cl(plm_BT_ij, plm_BT_ij, lmax=lmax)
            auto_EB = hp.alm2cl(plm_EB_ij, plm_EB_ij, lmax=lmax)
            auto_BE = hp.alm2cl(plm_BE_ij, plm_BE_ij, lmax=lmax)

            # Get cross spectra <ijji>
            cross = hp.alm2cl(plm_total_ij, plm_total_ji, lmax=lmax)
            cross_TT = hp.alm2cl(plm_TT_ij, plm_TT_ji, lmax=lmax)
            cross_EE = hp.alm2cl(plm_EE_ij, plm_EE_ji, lmax=lmax)
            cross_TE = hp.alm2cl(plm_TE_ij, plm_TE_ji, lmax=lmax)
            cross_ET = hp.alm2cl(plm_ET_ij, plm_ET_ji, lmax=lmax)
            cross_TB = hp.alm2cl(plm_TB_ij, plm_TB_ji, lmax=lmax)
            cross_BT = hp.alm2cl(plm_BT_ij, plm_BT_ji, lmax=lmax)
            cross_EB = hp.alm2cl(plm_EB_ij, plm_EB_ji, lmax=lmax)
            cross_BE = hp.alm2cl(plm_BE_ij, plm_BE_ji, lmax=lmax)

            n1['total'] += auto + cross
            n1['TT'] += auto_TT + cross_TT
            n1['EE'] += auto_EE + cross_EE
            n1['TE'] += auto_TE + cross_TE
            n1['ET'] += auto_ET + cross_ET
            n1['TB'] += auto_TB + cross_TB
            n1['BT'] += auto_BT + cross_BT
            n1['EB'] += auto_EB + cross_EB
            n1['BE'] += auto_BE + cross_BE

        n1['total'] *= 1/num
        n1['TT'] *= 1/num
        n1['EE'] *= 1/num
        n1['TE'] *= 1/num
        n1['ET'] *= 1/num
        n1['TB'] *= 1/num
        n1['BT'] *= 1/num
        n1['EB'] *= 1/num
        n1['BE'] *= 1/num

        n0 = get_n0(sims=sims,qetype=qetype,config=config,
                    resp_from_sims=resp_from_sims,cmbonly=True)

        n1['total'] -= n0['total']
        n1['TT'] -= n0['TT']
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

def get_sim_response(est, config, gmv, sims=np.arange(40)+1,
                     filename=None):
    '''
    If gmv, est should be 'TTEETE'/'TBEB'/'all'.
    If not gmv, assume sqe and est should be 'TT'/'EE'/'TE'/'TB'/'EB'.
    Make sure the sims are lensed, not unlensed!
    '''
    lmax = config['Lmax']
    lmaxT = config['lmaxT']
    lmaxP = config['lmaxP']
    lmin = config['lminT']
    nside = config['nside']
    cltype = config['cltype']
    dir_out = config['dir_out']
    l = np.arange(0,lmax+1)
    num = len(sims)
    if filename is None:
        append = ''
        if gmv:
            append += '_gmv_est{est}'
        else:
            append += f'_sqe_est{est}'
        append += f'_lmaxT{lmaxT}_lmaxP{lmaxP}_lmin{lmin}_cltype{cltype}_mh'
        filename = dir_out+f'/resp/sim_resp{append}.npy'

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
                plm = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_mh.npy')
            else:
                plm = np.load(dir_out+f'/plm_{est}_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_mh.npy')

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

def get_analytic_response(est, config, gmv,
                          filename=None):
    '''
    If gmv, est should be 'TTEETE'/'TBEB'/'all'.
    If not gmv, assume sqe and est should be 'TT'/'EE'/'TE'/'TB'/'EB'.
    Also, we are taking lmax values from the config file, so make sure those are right.
    Note we are also assuming noise files used in the MH test, and just loading the saved totalcl file.
    '''
    print(f'Computing analytic response for est {est}')
    lmax = config['Lmax']
    lmaxT = config['lmaxT']
    lmaxP = config['lmaxP']
    lmin = config['lminT']
    nside = config['nside']
    cltype = config['cltype']
    cls = config['cls']
    sl = {ee:config['cls'][cltype][ee] for ee in config['cls'][cltype].keys()}
    ell = np.arange(lmax+1,dtype=np.float_)
    dir_out = config['dir_out']

    if filename is None:
        append = ''
        if gmv and (est=='all' or est=='TTEETE' or est=='TBEB'):
            append += '_gmv_estall'
        elif gmv:
            append += f'_gmv_est{est}'
        else:
            append += f'_sqe_est{est}'
        append += f'_lmaxT{lmaxT}_lmaxP{lmaxP}_lmin{lmin}_cltype{cltype}_mh'
        filename = dir_out+f'/resp/an_resp{append}.npy'

    if os.path.isfile(filename):
        print('Loading from existing file!')
        R = np.load(filename)
    else:
        # File doesn't exist!
        # Load total Cls; these are for the MH test, obtained from alm2cl and averaging over 40 sims
        totalcls = np.load(dir_out+f'totalcls/totalcls_average_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_mh.npy')
        cltt = totalcls[:,0]; clee = totalcls[:,1]; clbb = totalcls[:,2]; clte = totalcls[:,3]

        if not gmv:
            # Create 1/Nl filters
            flt = np.zeros(lmax+1); flt[lmin:] = 1./cltt[lmin:]
            fle = np.zeros(lmax+1); fle[lmin:] = 1./clee[lmin:]
            flb = np.zeros(lmax+1); flb[lmin:] = 1./clbb[lmin:]

            if est[0] == 'T': flX = flt
            if est[0] == 'E': flX = fle
            if est[0] == 'B': flX = flb

            if est[1] == 'T': flY = flt
            if est[1] == 'E': flY = fle
            if est[1] == 'B': flY = flb

            qeXY = weights.weights(est,cls[cltype],lmax,u=None)
            qeZA = None
            R = resp.fill_resp(qeXY,np.zeros(lmax+1, dtype=np.complex_),flX,flY,qeZA=qeZA)
            np.save(filename, R)
        else:
            # GMV response
            gmv_r = gmv_resp.gmv_resp(config,cltype,totalcls,u=None,save_path=filename)
            if est == 'TTEETE' or est == 'TBEB' or est == 'all':
                gmv_r.calc_tvar()
            elif est == 'TTEETEprf':
                gmv_r.calc_tvar_PRF(cross=False)
            elif est == 'TTEETETTEETEprf':
                gmv_r.calc_tvar_PRF(cross=True)
            R = np.load(filename)

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

#compare_resp()
analyze()

