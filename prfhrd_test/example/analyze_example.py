#!/usr/bin/env python3
import numpy as np
import healpy as hp
import pickle
import matplotlib.pyplot as plt
import os, sys
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import resp
import gmv_resp
import utils
import weights
import qest

def analyze(sims=np.arange(100)+1,n0_n1_sims=np.arange(99)+1,
            u=np.ones(4096+1, dtype=np.complex_),fluxlim=0.200,
            config_file='profhrd_yuka.yaml',
            noise_file=None,fsky_corr=1,
            n0=True,n1=True,
            lbins=np.logspace(np.log10(50),np.log10(3000),20)):
    '''
    Compare hardening effects with N0/N1 subtraction.
    '''
    config = utils.parse_yaml(config_file)
    dir_out = config['dir_out']
    lmax = config['Lmax']
    lmin = config['lmint']
    lmaxT = config['lmaxt']
    lmaxP = config['lmaxp']
    nside = config['nside']
    l = np.arange(0,lmax+1)
    num = len(sims)
    bin_centers = (lbins[:-1] + lbins[1:]) / 2
    digitized = np.digitize(l, lbins)
    append = f'tsrc_fluxlim{fluxlim:.3f}'

    # Get SQE response
    ests = ['TT', 'EE', 'TE', 'TB', 'EB']
    resps_original = np.zeros((len(l),len(ests)), dtype=np.complex_)
    for i, est in enumerate(ests):
        resps_original[:,i] = get_analytic_response(est,config,gmv=False,
                                                    fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,u=u,
                                                    noise_file=noise_file,fsky_corr=fsky_corr)
    resp_original = resps_original[:,0]+resps_original[:,1]+2*resps_original[:,2]+2*resps_original[:,3]+2*resps_original[:,4]
    inv_resp_original = np.zeros_like(l,dtype=np.complex_); inv_resp_original[1:] = 1/(resp_original)[1:]

    # Get GMV response
    resp_gmv = get_analytic_response('all',config,gmv=True,
                                     fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,u=u,
                                     noise_file=noise_file,fsky_corr=fsky_corr)
    inv_resp_gmv = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv[1:] = 1./(resp_gmv)[1:]

    if u is not None:
        # If we are hardening, get the profile response and weight
        # SQE
        resp_original_TT_ss = get_analytic_response('TTprf',config,gmv=False,
                                                    fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,u=u,
                                                    noise_file=noise_file,fsky_corr=fsky_corr)
        resp_original_TT_sk = get_analytic_response('TTTTprf',config,gmv=False,
                                                    fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,u=u,
                                                    noise_file=noise_file,fsky_corr=fsky_corr)
        weight_original = -1 * resp_original_TT_sk / resp_original_TT_ss
        resp_original_hrd = resp_original + weight_original*resp_original_TT_sk
        inv_resp_original_hrd = np.zeros_like(l,dtype=np.complex_); inv_resp_original_hrd[1:] = 1/(resp_original_hrd)[1:]

        # GMV
        resp_gmv_TTEETE_ss = get_analytic_response('TTEETEprf',config,gmv=True,
                                                   fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,u=u[lmin:],
                                                   noise_file=noise_file,fsky_corr=fsky_corr)
        resp_gmv_TTEETE_sk = get_analytic_response('TTEETETTEETEprf',config,gmv=True,
                                                   fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,u=u[lmin:],
                                                   noise_file=noise_file,fsky_corr=fsky_corr)
        weight_gmv = -1 * resp_gmv_TTEETE_sk / resp_gmv_TTEETE_ss
        resp_gmv_hrd = resp_gmv + weight_gmv*resp_gmv_TTEETE_sk # Equivalent to resp_gmv_TTEETE (hardened) + resp_gmv_TBEB (unhardened)
        inv_resp_gmv_hrd = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_hrd[1:] = 1./(resp_gmv_hrd)[1:]

    if n0:
        # Get N0 correction (remember these still need (l*(l+1))**2/4 factor to convert to kappa)
        n0_gmv = get_n0(sims=n0_n1_sims,qetype='gmv',config=config,dir_out=dir_out,u=u,fluxlim=fluxlim,
                        fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,
                        noise_file=noise_file,fsky_corr=fsky_corr,harden=False)
        n0_gmv_total = n0_gmv['total'] * (l*(l+1))**2/4
        n0_original = get_n0(sims=n0_n1_sims,qetype='sqe',config=config,dir_out=dir_out,u=u,fluxlim=fluxlim,
                             fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,
                             noise_file=noise_file,fsky_corr=fsky_corr,harden=False)
        n0_original_total = n0_original['total'] * (l*(l+1))**2/4
        if u is not None:
            n0_gmv_hrd = get_n0(sims=n0_n1_sims,qetype='gmv',config=config,dir_out=dir_out,u=u,fluxlim=fluxlim,
                                fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,
                                noise_file=noise_file,fsky_corr=fsky_corr,harden=True)
            n0_gmv_total_hrd = n0_gmv_hrd['total'] * (l*(l+1))**2/4
            n0_original_hrd = get_n0(sims=n0_n1_sims,qetype='sqe',config=config,dir_out=dir_out,u=u,fluxlim=fluxlim,
                                     fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,
                                     noise_file=noise_file,fsky_corr=fsky_corr,harden=True)
            n0_original_total_hrd = n0_original_hrd['total'] * (l*(l+1))**2/4

    if n1:
        # Get N1 correction (remember these still need (l*(l+1))**2/4 factor to convert to kappa)
        n1_gmv = get_n1(sims=n0_n1_sims,qetype='gmv',config=config,dir_out=dir_out,u=u,fluxlim=fluxlim,
                        fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,
                        noise_file=noise_file,fsky_corr=fsky_corr,harden=False)
        n1_gmv_total = n1_gmv['total'] * (l*(l+1))**2/4
        n1_original = get_n1(sims=n0_n1_sims,qetype='sqe',config=config,dir_out=dir_out,u=u,fluxlim=fluxlim,
                             fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,
                             noise_file=noise_file,fsky_corr=fsky_corr,harden=False)
        n1_original_total = n1_original['total'] * (l*(l+1))**2/4

    auto_gmv_all = 0
    auto_original_all = 0
    auto_original_all_hrd = 0
    auto_gmv_all_hrd = 0
    auto_gmv_debiased_all = 0
    auto_original_debiased_all = 0
    auto_gmv_debiased_all_hrd = 0
    auto_original_debiased_all_hrd = 0
    ratio_gmv = np.zeros((len(sims),len(lbins)-1),dtype=np.complex_)
    ratio_original = np.zeros((len(sims),len(lbins)-1),dtype=np.complex_)
    ratio_gmv_hrd = np.zeros((len(sims),len(lbins)-1),dtype=np.complex_)
    ratio_original_hrd = np.zeros((len(sims),len(lbins)-1),dtype=np.complex_)

    for ii, sim in enumerate(sims):
        # Load GMV plms
        plm_gmv = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
        plm_gmv_TTEETE = np.load(dir_out+f'/plm_TTEETE_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
        plm_gmv_TBEB = np.load(dir_out+f'/plm_TBEB_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')

        # Load SQE plms
        plms_original = np.zeros((len(np.load(dir_out+f'/plm_TT_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')),5), dtype=np.complex_)
        for i, est in enumerate(ests):
            plms_original[:,i] = np.load(dir_out+f'/plm_{est}_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
        plm_original = plms_original[:,0]+plms_original[:,1]+2*plms_original[:,2]+2*plms_original[:,3]+2*plms_original[:,4]

        if u is not None:
            # Harden!
            # GMV
            glm_prf_TTEETE = np.load(dir_out+f'/plm_TTEETEprf_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            plm_gmv_hrd = plm_gmv + hp.almxfl(glm_prf_TTEETE, weight_gmv) # Equivalent to plm_gmv_TTEETE_hrd + plm_gmv_TBEB

            # SQE
            glm_prf_TT = np.load(dir_out+f'/plm_TTprf_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            plm_original_TT_hrd = plms_original[:,0] + hp.almxfl(glm_prf_TT, weight_original)
            plm_original_hrd = plm_original_TT_hrd + plms_original[:,1]+2*plms_original[:,2]+2*plms_original[:,3]+2*plms_original[:,4]

        # Response correct
        plm_gmv_resp_corr = hp.almxfl(plm_gmv,inv_resp_gmv)
        plm_original_resp_corr = hp.almxfl(plm_original,inv_resp_original)
        if u is not None:
            plm_gmv_resp_corr_hrd = hp.almxfl(plm_gmv_hrd,inv_resp_gmv_hrd)
            plm_original_resp_corr_hrd = hp.almxfl(plm_original_hrd,inv_resp_original_hrd)

        # Get spectra
        auto_gmv = hp.alm2cl(plm_gmv_resp_corr, plm_gmv_resp_corr, lmax=lmax) * (l*(l+1))**2/4
        auto_original = hp.alm2cl(plm_original_resp_corr, plm_original_resp_corr, lmax=lmax) * (l*(l+1))**2/4
        if u is not None:
            auto_gmv_hrd = hp.alm2cl(plm_gmv_resp_corr_hrd, plm_gmv_resp_corr_hrd, lmax=lmax) * (l*(l+1))**2/4
            auto_original_hrd = hp.alm2cl(plm_original_resp_corr_hrd, plm_original_resp_corr_hrd, lmax=lmax) * (l*(l+1))**2/4

        # N0 and N1 subtract
        if n0 and n1:
            auto_gmv_debiased = auto_gmv - n0_gmv_total - n1_gmv_total
            auto_original_debiased = auto_original - n0_original_total - n1_original_total
            if u is not None:
                auto_gmv_debiased_hrd = auto_gmv_hrd - n0_gmv_total_hrd - n1_gmv_total
                auto_original_debiased_hrd = auto_original_hrd - n0_original_total_hrd - n1_original_total
        elif n0:
            auto_gmv_debiased = auto_gmv - n0_gmv_total
            auto_original_debiased = auto_original - n0_original_total
            if u is not None:
                auto_gmv_debiased_hrd = auto_gmv_hrd - n0_gmv_total_hrd
                auto_original_debiased_hrd = auto_original_hrd - n0_original_total_hrd

        auto_gmv_all += auto_gmv
        auto_original_all += auto_original
        if u is not None:
            auto_original_all_hrd += auto_original_hrd
            auto_gmv_all_hrd += auto_gmv_hrd
        if n0:
            auto_gmv_debiased_all += auto_gmv_debiased
            auto_original_debiased_all += auto_original_debiased
            if u is not None:
                auto_gmv_debiased_all_hrd += auto_gmv_debiased_hrd
                auto_original_debiased_all_hrd += auto_original_debiased_hrd
    
        if not unl:
            input_plm = hp.read_alm(f'/scratch/users/yukanaka/lensing19-20/inputcmb/phi/phi_lmax_{lmax}/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_phi1_seed{sim}_lmax{lmax}.alm')
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
                if u is not None:
                    # Bin!
                    binned_auto_gmv_debiased = [auto_gmv_debiased_hrd[digitized == i].mean() for i in range(1, len(lbins))]
                    binned_auto_original_debiased = [auto_original_debiased_hrd[digitized == i].mean() for i in range(1, len(lbins))]
                    # Get ratio
                    ratio_gmv_hrd[ii,:] = np.array(binned_auto_gmv_debiased) / np.array(binned_auto_input)
                    ratio_original_hrd[ii,:] = np.array(binned_auto_original_debiased) / np.array(binned_auto_input)

    # Average
    auto_gmv_avg = auto_gmv_all / num
    auto_original_avg = auto_original_all / num
    if u is not None:
        auto_original_avg_hrd = auto_original_all_hrd / num
        auto_gmv_avg_hrd = auto_gmv_all_hrd / num
    if n0:
        auto_gmv_debiased_avg = auto_gmv_debiased_all / num
        auto_original_debiased_avg = auto_original_debiased_all / num
        if u is not None:
            auto_gmv_debiased_avg_hrd = auto_gmv_debiased_all_hrd / num
            auto_original_debiased_avg_hrd = auto_original_debiased_all_hrd / num

    if n0:
        # If debiasing, get the ratio points, error bars for the ratio points, and bin
        errorbars_gmv = np.std(ratio_gmv,axis=0)/np.sqrt(num)
        errorbars_original = np.std(ratio_original,axis=0)/np.sqrt(num)
        ratio_gmv = np.mean(ratio_gmv,axis=0)
        ratio_original = np.mean(ratio_original,axis=0)
        # Bin!
        binned_auto_gmv_debiased_avg = [auto_gmv_debiased_avg[digitized == i].mean() for i in range(1, len(lbins))]
        binned_auto_original_debiased_avg = [auto_original_debiased_avg[digitized == i].mean() for i in range(1, len(lbins))]
        if u is not None:
            errorbars_gmv_hrd = np.std(ratio_gmv_hrd,axis=0)/np.sqrt(num)
            errorbars_original_hrd = np.std(ratio_original_hrd,axis=0)/np.sqrt(num)
            ratio_gmv_hrd = np.mean(ratio_gmv_hrd,axis=0)
            ratio_original_hrd = np.mean(ratio_original_hrd,axis=0)
            # Bin!
            binned_auto_gmv_debiased_avg_hrd = [auto_gmv_debiased_avg_hrd[digitized == i].mean() for i in range(1, len(lbins))]
            binned_auto_original_debiased_avg_hrd = [auto_original_debiased_avg_hrd[digitized == i].mean() for i in range(1, len(lbins))]

    # Theory spectrum
    clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
    ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
    clkk = slpp * (l*(l+1))**2/4

    # Plot
    plt.figure(0)
    plt.clf()

    #plt.plot(l, auto_gmv_avg, color='darkblue', linestyle='-', label="Auto Spectrum (GMV)")
    plt.plot(l, auto_gmv_debiased_avg, color='darkblue', linestyle='-', label="Auto Spectrum (GMV)")

    #plt.plot(l, auto_original_avg, color='firebrick', linestyle='-', label=f'Auto Spectrum (SQE)')
    plt.plot(l, auto_original_debiased_avg, color='firebrick', linestyle='-', label=f'Auto Spectrum (SQE)')

    #plt.plot(l, auto_gmv_avg_hrd, color='powderblue', linestyle='--', label="Auto Spectrum (GMV, hardened)")
    plt.plot(l, auto_gmv_debiased_avg_hrd, color='cornflowerblue', linestyle='--', label="Auto Spectrum (GMV, hardened)")

    #plt.plot(l, auto_original_avg_hrd, color='pink', linestyle='--', label=f'Auto Spectrum (SQE, hardened)')
    plt.plot(l, auto_original_debiased_avg_hrd, color='lightcoral', linestyle='--', label=f'Auto Spectrum (SQE, hardened)')

    plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')

    #plt.plot(bin_centers, binned_auto_gmv_debiased_avg, color='darkblue', marker='o', linestyle='None', ms=3, label="Auto Spectrum (GMV)")
    plt.plot(bin_centers, binned_auto_gmv_debiased_avg_hrd, color='darkblue', marker='o', linestyle='None', ms=3, label="Auto Spectrum (GMV, hardened)")

    #plt.plot(bin_centers, binned_auto_original_debiased_avg, color='firebrick', marker='o', linestyle='None', ms=3, label="Auto Spectrum (SQE)")
    plt.plot(bin_centers, binned_auto_original_debiased_avg_hrd, color='firebrick', marker='o', linestyle='None', ms=3, label="Auto Spectrum (SQE, hardened)")

    #plt.plot(l, n0_gmv_total, color='powderblue', linestyle='-',label='N0 (GMV)')
    #plt.plot(l, n0_original_total, color='pink', linestyle='-',label='N0 (Original)')

    #plt.plot(l, inv_resp_gmv_hrd * (l*(l+1))**2/4, color='cornflowerblue', linestyle='--', label='1/R (GMV, hardened)')
    #plt.plot(l, inv_resp_gmv * (l*(l+1))**2/4, color='cornflowerblue', linestyle='--', label='1/R (GMV)')

    #plt.plot(l, inv_resp_original_hrd * (l*(l+1))**2/4, color='lightcoral', linestyle='--', label='1/R (SQE, hardened)')
    #plt.plot(l, inv_resp_original * (l*(l+1))**2/4, color='lightcoral', linestyle='--', label='1/R (SQE)')

    plt.ylabel("$C_\ell^{\kappa\kappa}$")
    plt.xlabel('$\ell$')
    plt.title(f'Spectra Averaged over {num} Sims')
    plt.legend(loc='lower left', fontsize='x-small')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,lmax)
    plt.ylim(1e-9,1e-6)
    #plt.savefig(dir_out+f'/figs/{num}_sims_comparison_{append}.png',bbox_inches='tight')
    plt.savefig(dir_out+f'/figs/{num}_sims_comparison_{append}_n0n1subtracted.png',bbox_inches='tight')

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
    plt.xlim(10,lmax)
    plt.savefig(dir_out+f'/figs/{num}_sims_comparison_{append}_n0n1subtracted_binnedratio_nohrd.png',bbox_inches='tight')

    plt.clf()
    plt.axhline(y=1, color='k', linestyle='--')
    plt.errorbar(bin_centers,ratio_gmv_hrd,yerr=errorbars_gmv_hrd,color='darkblue', marker='o', linestyle='None', ms=3, label="Ratio GMV/Input, Hardened")
    plt.errorbar(bin_centers,ratio_original_hrd,yerr=errorbars_original_hrd,color='firebrick', marker='o', linestyle='None', ms=3, label="Ratio Original/Input, Hardened")
    plt.xlabel('$\ell$')
    plt.title(f'Spectra Averaged over {num} Sims')
    plt.legend(loc='lower left', fontsize='x-small')
    plt.xscale('log')
    plt.xlim(10,lmax)
    plt.savefig(dir_out+f'/figs/{num}_sims_comparison_{append}_n0n1subtracted_binnedratio_hrd.png',bbox_inches='tight')

def get_n0(sims,qetype,config,
           fwhm=0,nlev_t=0,nlev_p=0,
           noise_file='nl_cmbmv_20192020.dat',fsky_corr=25.308939726920805,
           dir_out='/scratch/users/yukanaka/gmv/',u=None,fluxlim=0.200,noiseless=False,
           harden=False):
    '''
    Get N0 bias. qetype should be 'gmv' or 'sqe'.
    Returns dictionary containing keys 'total', 'TTEETE', and 'TBEB' for GMV, but note that
    if we are hardening, 'total' and 'TTEETE' will be hardened but 'TBEB' will not.
    Similarly for SQE.
    '''
    lmax = config['Lmax']
    lmaxT = config['lmaxt']
    lmaxP = config['lmaxp']
    lmin = config['lmint']
    nside = config['nside']
    cltype = config['cltype']
    sl = {ee:config['cls'][cltype][ee] for ee in config['cls'][cltype].keys()}
    l = np.arange(lmax+1,dtype=np.float_)
    num = len(sims)
    if u is not None and not noiseless:
        append = f'tsrc_fluxlim{fluxlim:.3f}'
    elif noiseless:
        append = 'noiseless_cmbonly'
    filename = dir_out + f'/n0/n0_{num}simpairs_healqest_{qetype}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}'
    if harden:
        filename += '_hardened'
    filename += '.pkl'

    if os.path.isfile(filename):
        n0 = pickle.load(open(filename,'rb'))

    elif qetype == 'gmv':
        # Get GMV analytic response
        resp_gmv = get_analytic_response('all',config,gmv=True,
                                         fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,u=u,
                                         noise_file=noise_file,fsky_corr=fsky_corr)
        resp_gmv_TTEETE = get_analytic_response('TTEETE',config,gmv=True,u=u,
                                                fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,
                                                noise_file=noise_file,fsky_corr=fsky_corr)
        resp_gmv_TBEB = get_analytic_response('TBEB',config,gmv=True,u=u,
                                              fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,
                                              noise_file=noise_file,fsky_corr=fsky_corr)
        if harden:
            # If we are hardening, get the profile response and weight
            u = u[lmin:]
            resp_gmv_TTEETE_ss = get_analytic_response('TTEETEprf',config,gmv=True,
                                                       fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,u=u,
                                                       noise_file=noise_file,fsky_corr=fsky_corr)
            resp_gmv_TTEETE_sk = get_analytic_response('TTEETETTEETEprf',config,gmv=True,
                                                       fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,u=u,
                                                       noise_file=noise_file,fsky_corr=fsky_corr)
            weight_gmv = -1 * resp_gmv_TTEETE_sk / resp_gmv_TTEETE_ss
            resp_gmv = resp_gmv + weight_gmv*resp_gmv_TTEETE_sk # Equivalent to resp_gmv_TTEETE (hardened) + resp_gmv_TBEB (unhardened)
            resp_gmv_TTEETE = resp_gmv_TTEETE + weight_gmv*resp_gmv_TTEETE_sk
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

            if harden:
                # Harden!
                glm_prf_TTEETE_ij = np.load(dir_out+f'/plm_TTEETEprf_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
                plm_gmv_ij = plm_gmv_ij + hp.almxfl(glm_prf_TTEETE_ij, weight_gmv) # Equivalent to plm_gmv_TTEETE_ij (hardened) + plm_gmv_TBEB_ij (unhardened)
                plm_gmv_TTEETE_ij = plm_gmv_TTEETE_ij + hp.almxfl(glm_prf_TTEETE_ij, weight_gmv)

                glm_prf_TTEETE_ji = np.load(dir_out+f'/plm_TTEETEprf_healqest_gmv_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
                plm_gmv_ji = plm_gmv_ji + hp.almxfl(glm_prf_TTEETE_ji, weight_gmv) # Equivalent to plm_gmv_TTEETE_ji (hardened) + plm_gmv_TBEB_ji (unhardened)
                plm_gmv_TTEETE_ji = plm_gmv_TTEETE_ji + hp.almxfl(glm_prf_TTEETE_ji, weight_gmv)

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
        # Get SQE analytic response
        ests = ['TT', 'EE', 'TE', 'TB', 'EB']
        resps_original = np.zeros((len(l),len(ests)), dtype=np.complex_)
        inv_resps_original = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
        for i, est in enumerate(ests):
            resps_original[:,i] = get_analytic_response(est,config,gmv=False,
                                                        fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,u=u,
                                                        noise_file=noise_file,fsky_corr=fsky_corr)
            inv_resps_original[1:,i] = 1/(resps_original)[1:,i]
        resp_original = resps_original[:,0]+resps_original[:,1]+2*resps_original[:,2]+2*resps_original[:,3]+2*resps_original[:,4]

        if harden:
            # If we are hardening, get the profile response and weight
            resp_original_TT_ss = get_analytic_response('TTprf',config,gmv=False,
                                                        fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,u=u,
                                                        noise_file=noise_file,fsky_corr=fsky_corr)
            resp_original_TT_sk = get_analytic_response('TTTTprf',config,gmv=False,
                                                        fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,u=u,
                                                        noise_file=noise_file,fsky_corr=fsky_corr)
            weight_original = -1 * resp_original_TT_sk / resp_original_TT_ss
            resp_original = resp_original + weight_original*resp_original_TT_sk
            resps_original[:,0] = resps_original[:,0] + weight_original*resp_original_TT_sk
            inv_resps_original[1:,0] = 1/(resps_original)[1:,0]
        inv_resp_original = np.zeros_like(l,dtype=np.complex_); inv_resp_original[1:] = 1/(resp_original)[1:]
        resp_original_TTEETE = resps_original[:,0]+resps_original[:,1]+2*resps_original[:,2]
        resp_original_TBEB = 2*resps_original[:,3]+2*resps_original[:,4]
        inv_resp_original_TTEETE = np.zeros_like(l,dtype=np.complex_); inv_resp_original_TTEETE[1:] = 1/(resp_original_TTEETE)[1:]
        inv_resp_original_TBEB = np.zeros_like(l,dtype=np.complex_); inv_resp_original_TBEB[1:] = 1/(resp_original_TBEB)[1:]

        n0 = {'total':0, 'TT':0, 'EE':0, 'TE':0, 'TB':0, 'EB':0, 'TTEETE':0, 'TBEB':0}
        for i, sim1 in enumerate(sims):
            sim2 = sim1 + 1

            # Get the lensed ij sims
            plm_TT_ij = np.load(dir_out+f'/plm_TT_healqest_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            plm_EE_ij = np.load(dir_out+f'/plm_EE_healqest_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            plm_TE_ij = np.load(dir_out+f'/plm_TE_healqest_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            plm_TB_ij = np.load(dir_out+f'/plm_TB_healqest_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            plm_EB_ij = np.load(dir_out+f'/plm_EB_healqest_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')

            # Now get the ji sims
            plm_TT_ji = np.load(dir_out+f'/plm_TT_healqest_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            plm_EE_ji = np.load(dir_out+f'/plm_EE_healqest_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            plm_TE_ji = np.load(dir_out+f'/plm_TE_healqest_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            plm_TB_ji = np.load(dir_out+f'/plm_TB_healqest_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            plm_EB_ji = np.load(dir_out+f'/plm_EB_healqest_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')

            if harden:
                # Harden!
                glm_prf_TT_ij = np.load(dir_out+f'/plm_TTprf_healqest_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
                plm_TT_ij = plm_TT_ij + hp.almxfl(glm_prf_TT_ij, weight_original)

                glm_prf_TT_ji = np.load(dir_out+f'/plm_TTprf_healqest_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
                plm_TT_ji = plm_TT_ji + hp.almxfl(glm_prf_TT_ji, weight_original)

            # Eight estimators!!!
            plm_total_ij = plm_TT_ij + plm_EE_ij + 2*plm_TE_ij + 2*plm_TB_ij + 2*plm_EB_ij
            plm_total_ji = plm_TT_ji + plm_EE_ji + 2*plm_TE_ji + 2*plm_TB_ji + 2*plm_EB_ji
            plm_TTEETE_ij = plm_TT_ij + plm_EE_ij + 2*plm_TE_ij
            plm_TTEETE_ji = plm_TT_ji + plm_EE_ji + 2*plm_TE_ji
            plm_TBEB_ij = 2*plm_TB_ij + 2*plm_EB_ij
            plm_TBEB_ji = 2*plm_TB_ji + 2*plm_EB_ji

            # Response correct healqest
            plm_total_ij = hp.almxfl(plm_total_ij,inv_resp_original)
            plm_TTEETE_ij = hp.almxfl(plm_TTEETE_ij,inv_resp_original_TTEETE)
            plm_TBEB_ij = hp.almxfl(plm_TBEB_ij,inv_resp_original_TBEB)
            plm_TT_ij = hp.almxfl(plm_TT_ij,inv_resps_original[:,0])
            plm_EE_ij = hp.almxfl(plm_EE_ij,inv_resps_original[:,1])
            plm_TE_ij = hp.almxfl(plm_TE_ij,inv_resps_original[:,2])
            plm_TB_ij = hp.almxfl(plm_TB_ij,inv_resps_original[:,3])
            plm_EB_ij = hp.almxfl(plm_EB_ij,inv_resps_original[:,4])

            plm_total_ji = hp.almxfl(plm_total_ji,inv_resp_original)
            plm_TTEETE_ji = hp.almxfl(plm_TTEETE_ji,inv_resp_original_TTEETE)
            plm_TBEB_ji = hp.almxfl(plm_TBEB_ji,inv_resp_original_TBEB)
            plm_TT_ji = hp.almxfl(plm_TT_ji,inv_resps_original[:,0])
            plm_EE_ji = hp.almxfl(plm_EE_ji,inv_resps_original[:,1])
            plm_TE_ji = hp.almxfl(plm_TE_ji,inv_resps_original[:,2])
            plm_TB_ji = hp.almxfl(plm_TB_ji,inv_resps_original[:,3])
            plm_EB_ji = hp.almxfl(plm_EB_ji,inv_resps_original[:,4])

            # Get ij auto spectra <ijij>
            auto = hp.alm2cl(plm_total_ij, plm_total_ij, lmax=lmax)
            auto_TTEETE = hp.alm2cl(plm_TTEETE_ij, plm_TTEETE_ij, lmax=lmax)
            auto_TBEB = hp.alm2cl(plm_TBEB_ij, plm_TBEB_ij, lmax=lmax)
            auto_TT = hp.alm2cl(plm_TT_ij, plm_TT_ij, lmax=lmax)
            auto_TE = hp.alm2cl(plm_TE_ij, plm_TE_ij, lmax=lmax)
            auto_EE = hp.alm2cl(plm_EE_ij, plm_EE_ij, lmax=lmax)
            auto_TB = hp.alm2cl(plm_TB_ij, plm_TB_ij, lmax=lmax)
            auto_EB = hp.alm2cl(plm_EB_ij, plm_EB_ij, lmax=lmax)

            # Get cross spectra <ijji>
            cross = hp.alm2cl(plm_total_ij, plm_total_ji, lmax=lmax)
            cross_TTEETE = hp.alm2cl(plm_TTEETE_ij, plm_TTEETE_ji, lmax=lmax)
            cross_TBEB = hp.alm2cl(plm_TBEB_ij, plm_TBEB_ji, lmax=lmax)
            cross_TT = hp.alm2cl(plm_TT_ij, plm_TT_ji, lmax=lmax)
            cross_EE = hp.alm2cl(plm_EE_ij, plm_EE_ji, lmax=lmax)
            cross_TE = hp.alm2cl(plm_TE_ij, plm_TE_ji, lmax=lmax)
            cross_TB = hp.alm2cl(plm_TB_ij, plm_TB_ji, lmax=lmax)
            cross_EB = hp.alm2cl(plm_EB_ij, plm_EB_ji, lmax=lmax)

            n0['total'] += auto + cross
            n0['TTEETE'] += auto_TTEETE + cross_TTEETE
            n0['TBEB'] += auto_TBEB + cross_TBEB
            n0['TT'] += auto_TT + cross_TT
            n0['EE'] += auto_EE + cross_EE
            n0['TE'] += auto_TE + cross_TE
            n0['TB'] += auto_TB + cross_TB
            n0['EB'] += auto_EB + cross_EB

        n0['total'] *= 1/num
        n0['TTEETE'] *= 1/num
        n0['TBEB'] *= 1/num
        n0['TT'] *= 1/num
        n0['EE'] *= 1/num
        n0['TE'] *= 1/num
        n0['TB'] *= 1/num
        n0['EB'] *= 1/num

        with open(filename, 'wb') as f:
            pickle.dump(n0, f)
    else:
        print('Invalid argument qetype.')

    return n0

def get_n1(sims,qetype,config,
           fwhm=0,nlev_t=0,nlev_p=0,
           noise_file='nl_cmbmv_20192020.dat',fsky_corr=25.308939726920805,
           dir_out='/scratch/users/yukanaka/gmv/',
           u=None,fluxlim=0.200,
           harden=True):
    '''
    Get N1 bias. qetype should be 'gmv' or 'sqe'.
    No foregrounds in sims used in N1 calculation.
    Returns dictionary containing keys 'total', 'TTEETE', and 'TBEB' for GMV.
    Similarly for SQE.
    '''
    lmax = config['Lmax']
    lmaxT = config['lmaxt']
    lmaxP = config['lmaxp']
    lmin = config['lmint']
    nside = config['nside']
    cltype = config['cltype']
    sl = {ee:config['cls'][cltype][ee] for ee in config['cls'][cltype].keys()}
    l = np.arange(lmax+1,dtype=np.float_)
    num = len(sims)
    filename = f'/scratch/users/yukanaka/gmv/n1/n1_{num}simpairs_healqest_{qetype}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_cmbonly'
    if not harden:
        filename += '_no_hrd'
    filename += '.pkl'

    if os.path.isfile(filename):
        n1 = pickle.load(open(filename,'rb'))

    elif qetype == 'gmv':
        # Get GMV analytic response
        resp_gmv = get_analytic_response('all',config,gmv=True,
                                         fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,u=u,
                                         noise_file=noise_file,fsky_corr=fsky_corr)
        resp_gmv_TTEETE = get_analytic_response('TTEETE',config,gmv=True,
                                                fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,u=u,
                                                noise_file=noise_file,fsky_corr=fsky_corr)
        resp_gmv_TBEB = get_analytic_response('TBEB',config,gmv=True,
                                              fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,u=u,
                                              noise_file=noise_file,fsky_corr=fsky_corr)

        if harden:
            # If we are hardening, get the profile response and weight
            u = u[lmin:]
            resp_gmv_TTEETE_ss = get_analytic_response('TTEETEprf',config,gmv=True,
                                                       fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,u=u,
                                                       noise_file=noise_file,fsky_corr=fsky_corr)
            resp_gmv_TTEETE_sk = get_analytic_response('TTEETETTEETEprf',config,gmv=True,
                                                       fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,u=u,
                                                       noise_file=noise_file,fsky_corr=fsky_corr)
            weight_gmv = -1 * resp_gmv_TTEETE_sk / resp_gmv_TTEETE_ss
            resp_gmv = resp_gmv + weight_gmv*resp_gmv_TTEETE_sk # Equivalent to resp_gmv_TTEETE (hardened) + resp_gmv_TBEB (unhardened)
            resp_gmv_TTEETE = resp_gmv_TTEETE + weight_gmv*resp_gmv_TTEETE_sk
        inv_resp_gmv = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv[1:] = 1./(resp_gmv)[1:]
        inv_resp_gmv_TTEETE = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TTEETE[1:] = 1./(resp_gmv_TTEETE)[1:]
        inv_resp_gmv_TBEB = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TBEB[1:] = 1./(resp_gmv_TBEB)[1:]

        n1 = {'total':0, 'TTEETE':0, 'TBEB':0}
        for i, sim in enumerate(sims):
            # These are reconstructions using sims that were lensed with the same phi but different CMB realizations, no foregrounds
            # Get the lensed ij sims
            plm_gmv_ij = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_cmbonly_phi1_tqu1tqu2.npy')
            plm_gmv_TTEETE_ij = np.load(dir_out+f'/plm_TTEETE_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_cmbonly_phi1_tqu1tqu2.npy')
            plm_gmv_TBEB_ij = np.load(dir_out+f'/plm_TBEB_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_cmbonly_phi1_tqu1tqu2.npy')

            # Now get the ji sims
            plm_gmv_ji = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_cmbonly_phi1_tqu2tqu1.npy')
            plm_gmv_TTEETE_ji = np.load(dir_out+f'/plm_TTEETE_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_cmbonly_phi1_tqu2tqu1.npy')
            plm_gmv_TBEB_ji = np.load(dir_out+f'/plm_TBEB_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_cmbonly_phi1_tqu2tqu1.npy')

            if harden:
                # Harden!
                glm_prf_TTEETE_ij = np.load(dir_out+f'/plm_TTEETEprf_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_cmbonly_phi1_tqu1tqu2.npy')
                plm_gmv_ij = plm_gmv_ij + hp.almxfl(glm_prf_TTEETE_ij, weight_gmv) # Equivalent to plm_gmv_TTEETE_ij (hardened) + plm_gmv_TBEB_ij (unhardened)
                plm_gmv_TTEETE_ij = plm_gmv_TTEETE_ij + hp.almxfl(glm_prf_TTEETE_ij, weight_gmv)

                glm_prf_TTEETE_ji = np.load(dir_out+f'/plm_TTEETEprf_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_cmbonly_phi1_tqu2tqu1.npy')
                plm_gmv_ji = plm_gmv_ji + hp.almxfl(glm_prf_TTEETE_ji, weight_gmv) # Equivalent to plm_gmv_TTEETE_ji (hardened) + plm_gmv_TBEB_ji (unhardened)
                plm_gmv_TTEETE_ji = plm_gmv_TTEETE_ji + hp.almxfl(glm_prf_TTEETE_ji, weight_gmv)

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
                    fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,
                    noise_file=noise_file,fsky_corr=fsky_corr,
                    dir_out=dir_out,u=u,noiseless=True,harden=harden)

        n1['total'] -= n0['total']
        n1['TTEETE'] -= n0['TTEETE']
        n1['TBEB'] -= n0['TBEB']

        with open(filename, 'wb') as f:
            pickle.dump(n1, f)

    elif qetype == 'sqe':
        # Get SQE analytic response
        ests = ['TT', 'EE', 'TE', 'TB', 'EB']
        resps_original = np.zeros((len(l),len(ests)), dtype=np.complex_)
        inv_resps_original = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
        for i, est in enumerate(ests):
            resps_original[:,i] = get_analytic_response(est,config,gmv=False,
                                                        fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,u=u,
                                                        noise_file=noise_file,fsky_corr=fsky_corr)
            inv_resps_original[1:,i] = 1/(resps_original)[1:,i]
        resp_original = resps_original[:,0]+resps_original[:,1]+2*resps_original[:,2]+2*resps_original[:,3]+2*resps_original[:,4]

        if harden:
            # If we are hardening, get the profile response and weight
            resp_original_TT_ss = get_analytic_response('TTprf',config,gmv=False,
                                                        fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,u=u,
                                                        noise_file=noise_file,fsky_corr=fsky_corr)
            resp_original_TT_sk = get_analytic_response('TTTTprf',config,gmv=False,
                                                        fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,u=u,
                                                        noise_file=noise_file,fsky_corr=fsky_corr)
            weight_original = -1 * resp_original_TT_sk / resp_original_TT_ss
            resp_original = resp_original + weight_original*resp_original_TT_sk # Equivalent to resp_original_TT (hardened) + np.sum(resps_original[:,1:], axis=1)
            resps_original[:,0] = resps_original[:,0] + weight_original*resp_original_TT_sk
            inv_resps_original[1:,0] = 1/(resps_original)[1:,0]
        inv_resp_original = np.zeros_like(l,dtype=np.complex_); inv_resp_original[1:] = 1/(resp_original)[1:]
        resp_original_TTEETE = resps_original[:,0]+resps_original[:,1]+2*resps_original[:,2]
        resp_original_TBEB = 2*resps_original[:,3]+2*resps_original[:,4]
        inv_resp_original_TTEETE = np.zeros_like(l,dtype=np.complex_); inv_resp_original_TTEETE[1:] = 1/(resp_original_TTEETE)[1:]
        inv_resp_original_TBEB = np.zeros_like(l,dtype=np.complex_); inv_resp_original_TBEB[1:] = 1/(resp_original_TBEB)[1:]

        n1 = {'total':0, 'TT':0, 'EE':0, 'TE':0, 'TB':0, 'EB':0}
        for i, sim in enumerate(sims):
            # Get the lensed ij sims
            plm_TT_ij = np.load(dir_out+f'/plm_TT_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_cmbonly_phi1_tqu1tqu2.npy')
            plm_EE_ij = np.load(dir_out+f'/plm_EE_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_cmbonly_phi1_tqu1tqu2.npy')
            plm_TE_ij = np.load(dir_out+f'/plm_TE_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_cmbonly_phi1_tqu1tqu2.npy')
            plm_TB_ij = np.load(dir_out+f'/plm_TB_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_cmbonly_phi1_tqu1tqu2.npy')
            plm_EB_ij = np.load(dir_out+f'/plm_EB_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_cmbonly_phi1_tqu1tqu2.npy')

            # Now get the ji sims
            plm_TT_ji = np.load(dir_out+f'/plm_TT_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_cmbonly_phi1_tqu2tqu1.npy')
            plm_EE_ji = np.load(dir_out+f'/plm_EE_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_cmbonly_phi1_tqu2tqu1.npy')
            plm_TE_ji = np.load(dir_out+f'/plm_TE_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_cmbonly_phi1_tqu2tqu1.npy')
            plm_TB_ji = np.load(dir_out+f'/plm_TB_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_cmbonly_phi1_tqu2tqu1.npy')
            plm_EB_ji = np.load(dir_out+f'/plm_EB_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_cmbonly_phi1_tqu2tqu1.npy')

            if harden:
                # Harden!
                glm_prf_TT_ij = np.load(dir_out+f'/plm_TTprf_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_cmbonly_phi1_tqu1tqu2.npy')
                plm_TT_ij = plm_TT_ij + hp.almxfl(glm_prf_TT_ij, weight_original)

                glm_prf_TT_ji = np.load(dir_out+f'/plm_TTprf_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_cmbonly_phi1_tqu2tqu1.npy')
                plm_TT_ji = plm_TT_ji + hp.almxfl(glm_prf_TT_ji, weight_original)

            # Eight estimators!!!
            plm_total_ij = plm_TT_ij + plm_EE_ij + 2*plm_TE_ij + 2*plm_TB_ij + 2*plm_EB_ij
            plm_total_ji = plm_TT_ji + plm_EE_ji + 2*plm_TE_ji + 2*plm_TB_ji + 2*plm_EB_ji

            # Response correct healqest
            plm_total_ij = hp.almxfl(plm_total_ij,inv_resp_original)
            plm_TT_ij = hp.almxfl(plm_TT_ij,inv_resps_original[:,0])
            plm_EE_ij = hp.almxfl(plm_EE_ij,inv_resps_original[:,1])
            plm_TE_ij = hp.almxfl(plm_TE_ij,inv_resps_original[:,2])
            plm_TB_ij = hp.almxfl(plm_TB_ij,inv_resps_original[:,3])
            plm_EB_ij = hp.almxfl(plm_EB_ij,inv_resps_original[:,4])

            plm_total_ji = hp.almxfl(plm_total_ji,inv_resp_original)
            plm_TT_ji = hp.almxfl(plm_TT_ji,inv_resps_original[:,0])
            plm_EE_ji = hp.almxfl(plm_EE_ji,inv_resps_original[:,1])
            plm_TE_ji = hp.almxfl(plm_TE_ji,inv_resps_original[:,2])
            plm_TB_ji = hp.almxfl(plm_TB_ji,inv_resps_original[:,3])
            plm_EB_ji = hp.almxfl(plm_EB_ji,inv_resps_original[:,4])

            # Get ij auto spectra <ijij>
            auto = hp.alm2cl(plm_total_ij, plm_total_ij, lmax=lmax)
            auto_TT = hp.alm2cl(plm_TT_ij, plm_TT_ij, lmax=lmax)
            auto_TE = hp.alm2cl(plm_TE_ij, plm_TE_ij, lmax=lmax)
            auto_EE = hp.alm2cl(plm_EE_ij, plm_EE_ij, lmax=lmax)
            auto_TB = hp.alm2cl(plm_TB_ij, plm_TB_ij, lmax=lmax)
            auto_EB = hp.alm2cl(plm_EB_ij, plm_EB_ij, lmax=lmax)

            # Get cross spectra <ijji>
            cross = hp.alm2cl(plm_total_ij, plm_total_ji, lmax=lmax)
            cross_TT = hp.alm2cl(plm_TT_ij, plm_TT_ji, lmax=lmax)
            cross_TE = hp.alm2cl(plm_TE_ij, plm_TE_ji, lmax=lmax)
            cross_EE = hp.alm2cl(plm_EE_ij, plm_EE_ji, lmax=lmax)
            cross_TB = hp.alm2cl(plm_TB_ij, plm_TB_ji, lmax=lmax)
            cross_EB = hp.alm2cl(plm_EB_ij, plm_EB_ji, lmax=lmax)

            n1['total'] += auto + cross
            n1['TT'] += auto_TT + cross_TT
            n1['EE'] += auto_EE + cross_EE
            n1['TE'] += auto_TE + cross_TE
            n1['TB'] += auto_TB + cross_TB
            n1['EB'] += auto_EB + cross_EB

        n1['total'] *= 1/num
        n1['TT'] *= 1/num
        n1['EE'] *= 1/num
        n1['TE'] *= 1/num
        n1['TB'] *= 1/num
        n1['EB'] *= 1/num

        n0 = get_n0(sims=sims,qetype=qetype,config=config,
                    fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,
                    noise_file=noise_file,fsky_corr=fsky_corr,
                    dir_out=dir_out,u=u,noiseless=True,harden=harden)

        n1['total'] -= n0['total']
        n1['TT'] -= n0['TT']
        n1['EE'] -= n0['EE']
        n1['TE'] -= n0['TE']
        n1['TB'] -= n0['TB']
        n1['EB'] -= n0['EB']

        with open(filename, 'wb') as f:
            pickle.dump(n1, f)
    else:
        print('Invalid argument qetype.')

    return n1

def get_analytic_response(est, config, gmv,
                          fwhm=0, nlev_t=0, nlev_p=0, u=None,
                          noise_file=None, fsky_corr=1,
                          filename=None):
    '''
    If gmv, est should be 'TTEETE'/'TBEB'/'all'/'TTEETEprf'/'TTEETETTEETEprf'.
    If not gmv, assume sqe and est should be 'TT'/'EE'/'TE'/'TB'/'EB'/'TTprf'/'TTTTprf'.
    Also, we are taking lmax values from the config file, so make sure those are right.
    Note we are also assuming that if we are loading from a noise file, we won't also add
    noise according to nlev_t and nlev_p.
    '''
    print(f'Computing analytic response for est {est}')
    dir_out = config['dir_out']
    lmax = config['Lmax']
    lmaxT = config['lmaxt']
    lmaxP = config['lmaxp']
    lmin = config['lmint']
    cltype = config['cltype']
    cls = config['cls']
    sl = {ee:config['cls'][cltype][ee] for ee in config['cls'][cltype].keys()}
    ell = np.arange(lmax+1,dtype=np.float_)

    if filename is None:
        append = ''
        if gmv and (est=='all' or est=='TTEETE' or est=='TBEB'):
            append += '_gmv_estall'
        elif gmv:
            append += f'_gmv_est{est}'
        else:
            append += f'_sqe_est{est}'
        append += f'_lmaxT{lmaxT}_lmaxP{lmaxP}_lmin{lmin}_cltype{cltype}'
        if noise_file is not None:
            print('Loading from noise file!')
            append += '_added_noise_from_file'
        else:
            append += f'_fwhm{fwhm}_nlevt{nlev_t}_nlevp{nlev_p}'
        if u is not None:
            append += '_with_fg'
        filename = dir_out + f'/resp/an_resp{append}.npy'

    if os.path.isfile(filename):
        print('Loading from existing file!')
        R = np.load(filename)
    else:
        # File doesn't exist! Calculate from scratch.
        if noise_file is not None:
            noise_curves = np.loadtxt(noise_file) # lmax 6000
            # With fsky correction
            nltt = fsky_corr * noise_curves[:,1]
            nlee = fsky_corr * noise_curves[:,2]
            nlbb = fsky_corr * noise_curves[:,2]
        else:
            bl = hp.gauss_beam(fwhm=fwhm*0.00029088,lmax=lmax)
            nltt = (np.pi/180./60.*nlev_t)**2 / bl**2
            nlee=nlbb = (np.pi/180./60.*nlev_p)**2 / bl**2

        if u is not None:
            # Point source maps have a flat Cl power spectrum at 2.18e-05 uK^2
            fgtt =  np.ones(lmax+1) * 2.18e-5
        else:
            fgtt = np.zeros(lmax+1)
    
        # Signal + noise spectra
        artificial_noise = np.zeros(lmax+1)
        if lmaxT < lmaxP:
            artificial_noise[lmaxT+2:] = 1.e10
        cltt = sl['tt'][:lmax+1] + nltt[:lmax+1] + fgtt + artificial_noise
        clee = sl['ee'][:lmax+1] + nlee[:lmax+1]
        clbb = sl['bb'][:lmax+1] + nlbb[:lmax+1]
        clte = sl['te'][:lmax+1]

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

            if est == 'TTTTprf':
                qeXY = weights.weights(config,cls[cltype],'TT',u=u)
                qeZA = weights.weights(config,cls[cltype],'TTprf',u=u)
            else:
                qeXY = weights.weights(config,cls[cltype],est,u=u)
                qeZA = None
            R = resp.fill_resp(qeXY,np.zeros(lmax+1, dtype=np.complex_),flX,flY,qeZA=qeZA)
            np.save(filename, R)
        else:
            # GMV response
            totalcls = np.vstack((cltt,clee,clbb,clte)).T
            gmv_r = gmv_resp.gmv_resp(config,cltype,totalcls,u=u,save_path=filename)
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
