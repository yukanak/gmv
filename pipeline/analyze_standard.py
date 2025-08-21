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

def analyze(sims=np.arange(250)+1,n0_n1_sims=np.arange(249)+1,
            config_file='test_yuka.yaml',
            append='standard',
            n0=True,n1=True,
            lbins=np.logspace(np.log10(50),np.log10(3000),20),
            compare=False,sqe=False,fg_model='agora'):
    '''
    Default is cinv-style.
    If compare is True, compare to non-cinv GMV OR SQE.
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
    u = None
    # Theory spectrum
    clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
    ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
    clkk = slpp * (l*(l+1))**2/4
    binned_clkk = [clkk[digitized == i].mean() for i in range(1, len(lbins))]
    # Input kappa
    if fg_model == 'agora':
        #klm = hp.read_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_spt3g_input_klm_lmax{lmax}.fits')
        klm = hp.almxfl(hp.read_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_spt3g_input_plm_lmax{lmax}.fits'),(l*(l+1))/2)
        klm = utils.reduce_lmax(klm,lmax=lmax)
    else:
        kap = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/websky/kap.fits')
        klm = hp.map2alm(kap)
        klm = utils.reduce_lmax(klm,lmax=lmax)
    input_clkk = hp.alm2cl(klm,klm,lmax=lmax)
    binned_input_clkk = np.array([input_clkk[digitized == i].mean() for i in range(1, len(lbins))])

    # Get response
    ests = ['TT', 'EE', 'TE', 'ET', 'TB', 'BT', 'EB', 'BE']
    resps = np.zeros((len(l),len(ests)), dtype=np.complex_)
    inv_resps = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
    for i, est in enumerate(ests):
        resps[:,i] = get_sim_response(est,config,cinv=True,append=append,sims=sims,gmv=True,fg_model=fg_model)
        inv_resps[1:,i] = 1/(resps)[1:,i]
    resp = np.sum(resps, axis=1)
    inv_resp = np.zeros_like(l,dtype=np.complex_); inv_resp[1:] = 1/(resp)[1:]
    if compare and sqe:
        resps_sqe = np.zeros((len(l),len(ests)), dtype=np.complex_)
        inv_resps_sqe = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
        for i, est in enumerate(ests):
            resps_sqe[:,i] = get_sim_response(est,config,cinv=False,gmv=False,append=append,sims=sims,fg_model=fg_model)
            inv_resps_sqe[1:,i] = 1/(resps_sqe)[1:,i]
        resp_sqe = np.sum(resps_sqe, axis=1)
        inv_resp_sqe = np.zeros_like(l,dtype=np.complex_); inv_resp_sqe[1:] = 1/(resp_sqe)[1:]
    elif compare:
        # Original (not cinv-style) GMV response
        resp_gmv = get_sim_response('all',config,cinv=False,append=append,sims=sims,fg_model=fg_model)
        #resp_gmv_TTEETE = get_sim_response('TTEETE',config,cinv=False,append=append,sims=sims)
        #resp_gmv_TBEB = get_sim_response('TBEB',config,cinv=False,append=append,sims=sims)
        inv_resp_gmv = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv[1:] = 1./(resp_gmv)[1:]
        #inv_resp_gmv_TTEETE = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TTEETE[1:] = 1./(resp_gmv_TTEETE)[1:]
        #inv_resp_gmv_TBEB = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TBEB[1:] = 1./(resp_gmv_TBEB)[1:]

    if n0:
        # Get N0 correction (remember these still need (l*(l+1))**2/4 factor to convert to kappa)
        n0_cinv = get_n0(sims=n0_n1_sims,qetype='gmv_cinv',config=config,
                         append=append,fg_model=fg_model)
        n0_cinv_total = n0_cinv['total'] * (l*(l+1))**2/4
        if compare and sqe:
            n0_sqe = get_n0(sims=n0_n1_sims,qetype='sqe',config=config,append=append,fg_model=fg_model)
            n0_sqe_total = n0_sqe['total'] * (l*(l+1))**2/4
        elif compare:
            n0_gmv = get_n0(sims=n0_n1_sims,qetype='gmv',config=config,append=append,fg_model=fg_model)
            n0_gmv_total = n0_gmv['total'] * (l*(l+1))**2/4

    if n1:
        n1_cinv = get_n1(sims=n0_n1_sims,qetype='gmv_cinv',config=config,
                         append=append,fg_model=fg_model)
        n1_cinv_total = n1_cinv['total'] * (l*(l+1))**2/4
        if compare and sqe:
            n1_sqe = get_n1(sims=n0_n1_sims,qetype='sqe',config=config,append=append,fg_model=fg_model)
            n1_sqe_total = n1_sqe['total'] * (l*(l+1))**2/4
        elif compare:
            n1_gmv = get_n1(sims=n0_n1_sims,qetype='gmv',config=config,append=append,fg_model=fg_model)
            n1_gmv_total = n1_gmv['total'] * (l*(l+1))**2/4

    auto_gmv_all = 0
    auto_cinv_all = 0
    auto_sqe_all = 0
    auto_gmv_debiased_all = 0
    auto_cinv_debiased_all = 0
    auto_sqe_debiased_all = 0
    ratio_gmv = np.zeros((len(sims),len(lbins)-1),dtype=np.complex_)
    ratio_cinv = np.zeros((len(sims),len(lbins)-1),dtype=np.complex_)
    ratio_sqe = np.zeros((len(sims),len(lbins)-1),dtype=np.complex_)
    temp_gmv = np.zeros((len(sims),len(l)),dtype=np.complex_)
    temp_cinv = np.zeros((len(sims),len(l)),dtype=np.complex_)
    temp_sqe = np.zeros((len(sims),len(l)),dtype=np.complex_)
    binned_temp_gmv = np.zeros((len(sims),len(lbins)-1),dtype=np.complex_)
    binned_temp_cinv = np.zeros((len(sims),len(lbins)-1),dtype=np.complex_)
    binned_temp_sqe = np.zeros((len(sims),len(lbins)-1),dtype=np.complex_)
    cross_cinv_all = 0

    for ii, sim in enumerate(sims):
        # Load cinv-style GMV plms
        if os.path.isfile(dir_out+f'/plm_summed_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv.npy'):
            plm = np.load(dir_out+f'/plm_summed_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv.npy')
        else:
            plms = np.zeros((len(np.load(dir_out+f'/plm_TT_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv.npy')),len(ests)), dtype=np.complex_)
            for i, est in enumerate(ests):
                plms[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv.npy')
            plm = np.sum(plms, axis=1)
            np.save(dir_out+f'/plm_summed_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv.npy',plm)
            for i, est in enumerate(ests):
                os.remove(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv.npy')

        if np.any(np.isnan(plm)):
            print(f'Sim {sim} is bad!')
            num -= 1
            continue

        # Response correct
        plm_resp_corr = hp.almxfl(plm,inv_resp)

        # Get spectra
        auto = hp.alm2cl(plm_resp_corr, plm_resp_corr, lmax=lmax) * (l*(l+1))**2/4

        # N0 and N1 subtract
        if n0 and n1:
            auto_debiased = auto - n0_cinv_total - n1_cinv_total
        elif n0:
            auto_debiased = auto - n0_cinv_total

        # Sum the auto spectra
        auto_cinv_all += auto

        if n0:
            auto_cinv_debiased_all += auto_debiased

            # Need this to compute uncertainty...
            temp_cinv[ii,:] = auto_debiased
            binned_temp_cinv[ii,:] = [auto_debiased[digitized == i].mean() for i in range(1, len(lbins))]

            # If debiasing, get the binned ratio against input
            input_plm = hp.read_alm(f'/oak/stanford/orgs/kipac//users/yukanaka/lensing19-20/inputcmb/phi/phi_lmax_{lmax}/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_phi1_seed{sim}_lmax{lmax}.alm')
            auto_input = hp.alm2cl(input_plm, input_plm, lmax=lmax) * (l*(l+1))**2/4
            # Bin!
            binned_auto_cinv_debiased = [auto_debiased[digitized == i].mean() for i in range(1, len(lbins))]
            binned_auto_input = [auto_input[digitized == i].mean() for i in range(1, len(lbins))]
            # Get ratio
            ratio_cinv[ii,:] = np.array(binned_auto_cinv_debiased) / np.array(binned_auto_input)

            if fg_model == 'websky':
                cross_cinv_all += hp.alm2cl(klm, plm_resp_corr, lmax=lmax) * (l*(l+1))/2

        if compare and sqe:
            # Load SQE plms
            if os.path.isfile(dir_out+f'/plm_summed_healqest_sqe_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy'):
                plm_sqe = np.load(dir_out+f'/plm_summed_healqest_sqe_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy')
            else:
                plms_sqe = np.zeros((len(np.load(dir_out+f'/plm_TT_healqest_sqe_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy')),len(ests)), dtype=np.complex_)
                for i, est in enumerate(ests):
                    plms_sqe[:,i] = np.load(dir_out+f'/plm_{est}_healqest_sqe_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy')
                plm_sqe = np.sum(plms_sqe, axis=1)
                np.save(dir_out+f'/plm_summed_healqest_sqe_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy',plm_sqe)
                for i, est in enumerate(ests):
                    os.remove(dir_out+f'/plm_{est}_healqest_sqe_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy')

            # Response correct
            plm_sqe_resp_corr = hp.almxfl(plm_sqe,inv_resp_sqe)

            # Get spectra
            auto_sqe = hp.alm2cl(plm_sqe_resp_corr, plm_sqe_resp_corr, lmax=lmax) * (l*(l+1))**2/4

            # N0 and N1 subtract
            if n0 and n1:
                auto_sqe_debiased = auto_sqe - n0_sqe_total - n1_sqe_total
            elif n0:
                auto_sqe_debiased = auto_sqe - n0_sqe_total

            # Sum the auto spectra
            auto_sqe_all += auto_sqe

            if n0:
                auto_sqe_debiased_all += auto_sqe_debiased

                # Need this to compute uncertainty...
                temp_sqe[ii,:] = auto_sqe_debiased
                binned_temp_sqe[ii,:] = [auto_sqe_debiased[digitized == i].mean() for i in range(1, len(lbins))]

                # If debiasing, get the binned ratio against input
                input_plm = hp.read_alm(f'/oak/stanford/orgs/kipac//users/yukanaka/lensing19-20/inputcmb/phi/phi_lmax_{lmax}/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_phi1_seed{sim}_lmax{lmax}.alm')
                auto_input = hp.alm2cl(input_plm, input_plm, lmax=lmax) * (l*(l+1))**2/4
                # Bin!
                binned_auto_sqe_debiased = [auto_sqe_debiased[digitized == i].mean() for i in range(1, len(lbins))]
                binned_auto_input = [auto_input[digitized == i].mean() for i in range(1, len(lbins))]
                # Get ratio
                ratio_sqe[ii,:] = np.array(binned_auto_sqe_debiased) / np.array(binned_auto_input)

        elif compare:
            # Load GMV plms
            plm_gmv = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy')

            # Response correct
            plm_gmv_resp_corr = hp.almxfl(plm_gmv,inv_resp_gmv)

            # Get spectra
            auto_gmv = hp.alm2cl(plm_gmv_resp_corr, plm_gmv_resp_corr, lmax=lmax) * (l*(l+1))**2/4

            # N0 and N1 subtract
            if n0 and n1:
                auto_gmv_debiased = auto_gmv - n0_gmv_total - n1_gmv_total
            elif n0:
                auto_gmv_debiased = auto_gmv - n0_gmv_total

            # Sum the auto spectra
            auto_gmv_all += auto_gmv

            if n0:
                auto_gmv_debiased_all += auto_gmv_debiased

                # Need this to compute uncertainty...
                temp_gmv[ii,:] = auto_gmv_debiased
                binned_temp_gmv[ii,:] = [auto_gmv_debiased[digitized == i].mean() for i in range(1, len(lbins))]

                # If debiasing, get the binned ratio against input
                input_plm = hp.read_alm(f'/oak/stanford/orgs/kipac//users/yukanaka/lensing19-20/inputcmb/phi/phi_lmax_{lmax}/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_phi1_seed{sim}_lmax{lmax}.alm')
                auto_input = hp.alm2cl(input_plm, input_plm, lmax=lmax) * (l*(l+1))**2/4
                # Bin!
                binned_auto_gmv_debiased = [auto_gmv_debiased[digitized == i].mean() for i in range(1, len(lbins))]
                binned_auto_input = [auto_input[digitized == i].mean() for i in range(1, len(lbins))]
                # Get ratio
                ratio_gmv[ii,:] = np.array(binned_auto_gmv_debiased) / np.array(binned_auto_input)

    # GET THE UNCERTAINTIES (error bar from spread of sims -- measurement uncertainty of bandpowers)
    uncertainty_cinv = np.std(temp_cinv,axis=0)
    binned_uncertainty_cinv = np.std(binned_temp_cinv,axis=0)
    if not os.path.isfile(dir_out+f'/measurement_uncertainty/measurement_uncertainty_lmaxT{lmaxT}_{fg_model}_{append}_cinv.npy'):
        np.save(dir_out+f'/measurement_uncertainty/measurement_uncertainty_lmaxT{lmaxT}_{fg_model}_{append}_cinv.npy',uncertainty_cinv)
        np.save(dir_out+f'/measurement_uncertainty/binned_measurement_uncertainty_lmaxT{lmaxT}_{fg_model}_{append}_cinv.npy',binned_uncertainty_cinv)
    if compare and sqe:
        uncertainty_sqe = np.std(temp_sqe,axis=0)
        binned_uncertainty_sqe = np.std(binned_temp_sqe,axis=0)
        if not os.path.isfile(dir_out+f'/measurement_uncertainty/measurement_uncertainty_lmaxT{lmaxT}_{fg_model}_{append}_sqe.npy'):
            np.save(dir_out+f'/measurement_uncertainty/measurement_uncertainty_lmaxT{lmaxT}_{fg_model}_{append}_sqe.npy',uncertainty_sqe)
            np.save(dir_out+f'/measurement_uncertainty/binned_measurement_uncertainty_lmaxT{lmaxT}_{fg_model}_{append}_sqe.npy',binned_uncertainty_sqe)
    elif compare:
        uncertainty_gmv = np.std(temp_gmv,axis=0)
        binned_uncertainty_gmv = np.std(binned_temp_gmv,axis=0)
        if not os.path.isfile(dir_out+f'/measurement_uncertainty/measurement_uncertainty_lmaxT{lmaxT}_{fg_model}_{append}_gmv.npy'):
            np.save(dir_out+f'/measurement_uncertainty/measurement_uncertainty_lmaxT{lmaxT}_{fg_model}_{append}_gmv.npy',uncertainty_gmv)
            np.save(dir_out+f'/measurement_uncertainty/binned_measurement_uncertainty_lmaxT{lmaxT}_{fg_model}_{append}_gmv.npy',binned_uncertainty_gmv)

    # Average
    auto_cinv_avg = auto_cinv_all / num
    cross_cinv_avg = cross_cinv_all / num
    if compare and sqe:
        auto_sqe_avg = auto_sqe_all / num
    elif compare:
        auto_gmv_avg = auto_gmv_all / num
    if n0:
        auto_cinv_debiased_avg = auto_cinv_debiased_all / num

        # If debiasing, get the ratio points, error bars for the ratio points, and bin
        mask = ~np.all(ratio_cinv == 0, axis=1)
        ratio_cinv = ratio_cinv[mask]
        errorbars_cinv = np.std(ratio_cinv,axis=0)/np.sqrt(num)
        #TODO
        #errorbars_cinv_temp = np.std(ratio_cinv,axis=0)
        #np.save('errorbars_cinv_standard_lmaxT3000.npy',errorbars_cinv_temp)
        #np.save('errorbars_cinv_standard_lmaxT3000_stderror.npy',errorbars_cinv)
        ratio_cinv = np.mean(ratio_cinv,axis=0)
        # Bin!
        binned_auto_cinv_debiased_avg = [auto_cinv_debiased_avg[digitized == i].mean() for i in range(1, len(lbins))]

        if compare and sqe:
            auto_sqe_debiased_avg = auto_sqe_debiased_all / num

            # If debiasing, get the ratio points, error bars for the ratio points, and bin
            errorbars_sqe = np.std(ratio_sqe,axis=0)/np.sqrt(num)
            ratio_sqe = np.mean(ratio_sqe,axis=0)
            # Bin!
            binned_auto_sqe_debiased_avg = [auto_sqe_debiased_avg[digitized == i].mean() for i in range(1, len(lbins))]

        elif compare:
            auto_gmv_debiased_avg = auto_gmv_debiased_all / num

            # If debiasing, get the ratio points, error bars for the ratio points, and bin
            errorbars_gmv = np.std(ratio_gmv,axis=0)/np.sqrt(num)
            ratio_gmv = np.mean(ratio_gmv,axis=0)
            # Bin!
            binned_auto_gmv_debiased_avg = [auto_gmv_debiased_avg[digitized == i].mean() for i in range(1, len(lbins))]

    # SAVE GAUSSIAN RECONSTRUCTION RESULTS
    results = {}
    results['bin_centers'] = bin_centers
    results['binned_camb_clkk'] = binned_clkk
    results[f'binned_{fg_model}_clkk'] = binned_input_clkk
    results['auto_cinv_debiased_avg'] = auto_cinv_debiased_avg
    results['binned_auto_cinv_debiased_avg'] = binned_auto_cinv_debiased_avg
    results['ratio_cinv'] = ratio_cinv
    results['errorbars_cinv'] = errorbars_cinv
    if not os.path.isfile(dir_out+f'/gaussian_reconstruction/gaussian_reconstruction_results_lmaxT{lmaxT}_{fg_model}_{append}_cinv.npy'):
        with open(dir_out+f'/gaussian_reconstruction/gaussian_reconstruction_results_lmaxT{lmaxT}_{fg_model}_{append}_cinv.npy', 'wb') as f:
            pickle.dump(results, f)
    if compare and sqe:
        results_sqe = {}
        results_sqe['bin_centers'] = bin_centers
        results_sqe['binned_camb_clkk'] = binned_clkk
        results_sqe[f'binned_{fg_model}_clkk'] = binned_input_clkk
        results_sqe['auto_sqe_debiased_avg'] = auto_sqe_debiased_avg
        results_sqe['binned_auto_sqe_debiased_avg'] = binned_auto_sqe_debiased_avg
        results_sqe['ratio_sqe'] = ratio_sqe
        results_sqe['errorbars_sqe'] = errorbars_sqe
        if not os.path.isfile(dir_out+f'/gaussian_reconstruction/gaussian_reconstruction_results_lmaxT{lmaxT}_{fg_model}_{append}_sqe.npy'):
            with open(dir_out+f'/gaussian_reconstruction/gaussian_reconstruction_results_lmaxT{lmaxT}_{fg_model}_{append}_sqe.npy', 'wb') as f:
                pickle.dump(results_sqe, f)
    elif compare:
        results_gmv = {}
        results_gmv['bin_centers'] = bin_centers
        results_gmv['binned_camb_clkk'] = binned_clkk
        results_gmv[f'binned_{fg_model}_clkk'] = binned_input_clkk
        results_gmv['auto_gmv_debiased_avg'] = auto_gmv_debiased_avg
        results_gmv['binned_auto_gmv_debiased_avg'] = binned_auto_gmv_debiased_avg
        results_gmv['ratio_gmv'] = ratio_gmv
        results_gmv['errorbars_gmv'] = errorbars_gmv
        if not os.path.isfile(dir_out+f'/gaussian_reconstruction/gaussian_reconstruction_results_lmaxT{lmaxT}_{fg_model}_{append}_gmv.npy'):
            with open(dir_out+f'/gaussian_reconstruction/gaussian_reconstruction_results_lmaxT{lmaxT}_{fg_model}_{append}_gmv.npy', 'wb') as f:
                pickle.dump(results_gmv, f)

    # Plot
    plt.figure(0)
    plt.clf()
    plt.plot(l, auto_cinv_debiased_avg, color='lightcoral', linestyle='-', alpha=0.5)
    if compare and sqe:
        plt.plot(l, auto_sqe_debiased_avg, color='cornflowerblue', linestyle='-', alpha=0.5)
    elif compare:
        plt.plot(l, auto_gmv_debiased_avg, color='cornflowerblue', linestyle='-', alpha=0.5)
    plt.plot(l, clkk, 'k', label='Fiducial $C_L^{\kappa\kappa}$')
    plt.errorbar(bin_centers,binned_auto_cinv_debiased_avg,yerr=binned_uncertainty_cinv/np.sqrt(250),fmt='o',markerfacecolor='none',markeredgecolor='firebrick',color='firebrick',alpha=0.5,linestyle='None', ms=7, label=f'GMV')
    if compare and sqe:
        plt.errorbar(bin_centers,binned_auto_sqe_debiased_avg,yerr=binned_uncertainty_sqe/np.sqrt(250),fmt='s',markerfacecolor='none',markeredgecolor='darkblue',color='darkblue',alpha=0.5,linestyle='None', ms=7, label="SQE")
    elif compare:
        plt.errorbar(bin_centers,binned_auto_gmv_debiased_avg,yerr=binned_uncertainty_gmv/np.sqrt(250),fmt='s',markerfacecolor='none',markeredgecolor='darkblue',color='darkblue',alpha=0.5,linestyle='None', ms=7, label="GMV, Harmonic Space Approach")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.ylabel("$C_L^{\kappa\kappa}$")
    plt.xlabel('$L$')
    plt.title(f'Reconstructed Kappa Autospectra, Averaged over {num} Sims, No Foreground Cleaning',pad=10)
    plt.legend(loc='upper right', fontsize='small')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,lmax)
    plt.ylim(1e-9,1e-6)
    plt.tight_layout()
    if n1:
        plt.savefig(dir_out+f'/figs/{num}_sims_comparison_{fg_model}_{append}_resp_from_sims_n0n1subtracted_lmaxT{lmaxT}.png',bbox_inches='tight')
    else:
        plt.savefig(dir_out+f'/figs/{num}_sims_comparison_{fg_model}_{append}_resp_from_sims_n0subtracted_lmaxT{lmaxT}.png',bbox_inches='tight')

    plt.figure(1)
    plt.clf()
    # Ratios with error bars
    plt.axhline(y=1, color='k', linestyle='--')
    #===TODO===#
    # This is the same as lensing bias + 1
    #plt.plot(bin_centers, np.array(binned_auto_cinv_debiased_avg)/np.array(binned_clkk), color='darkblue',alpha=0.5,linestyle='--',label='Reconstructed Kappa Autospectrum / CAMB Theory')
    #plt.plot(bin_centers, (np.array(binned_auto_cinv_debiased_avg))/np.array(binned_input_clkk), color='forestgreen',alpha=0.5,linestyle='--',label='Reconstructed Kappa Autospectrum / Agora Input')
    #==========#
    plt.errorbar(bin_centers,ratio_cinv,yerr=errorbars_cinv,color='firebrick',fmt='o',markerfacecolor='none',markeredgecolor='firebrick',alpha=0.5,linestyle='None', ms=7, label="GMV")
    if compare and sqe:
        plt.errorbar(bin_centers,ratio_sqe,yerr=errorbars_sqe,color='darkblue',fmt='s',markerfacecolor='none',markeredgecolor='darkblue',alpha=0.5,linestyle='None',ms=7,label="SQE")
    elif compare:
        plt.errorbar(bin_centers,ratio_gmv,yerr=errorbars_gmv,color='darkblue',fmt='s',markerfacecolor='none',markeredgecolor='darkblue',alpha=0.5,linestyle='None',ms=7,label="GMV, Harmonic Space Approach")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlabel('$L$')
    plt.title(f'Reconstructed Phi / Input Phi, Averaged over {num} Sims, No Foreground Cleaning',pad=10)
    plt.legend(loc='lower left', fontsize='small')
    plt.xscale('log')
    plt.ylim(0.98,1.02)
    plt.xlim(10,lmax)
    plt.tight_layout()
    if n1:
        plt.savefig(dir_out+f'/figs/{num}_sims_comparison_{fg_model}_{append}_resp_from_sims_n0n1subtracted_binnedratio_lmaxT{lmaxT}.png',bbox_inches='tight')
    else:
        plt.savefig(dir_out+f'/figs/{num}_sims_comparison_{fg_model}_{append}_resp_from_sims_n0subtracted_binnedratio_lmaxT{lmaxT}.png',bbox_inches='tight')

def get_n0(sims,qetype,config,append,cmbonly=False,fg_model='agora'):
    '''
    Get N0 bias. qetype should be 'sqe', 'gmv' or 'gmv_cinv'.
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
    filename = dir_out+f'/n0/n0_{num}simpairs_healqest_{qetype}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_resp_from_sims.pkl'

    if os.path.isfile(filename):
        n0 = pickle.load(open(filename,'rb'))

    elif qetype == 'gmv':
        # GMV response
        resp_gmv = get_sim_response('all',config,cinv=False,append=append_original,sims=np.append(sims,num+1),fg_model=fg_model)
        #resp_gmv_TTEETE = get_sim_response('TTEETE',config,cinv=False,append=append_original,sims=np.append(sims,num+1))
        #resp_gmv_TBEB = get_sim_response('TBEB',config,cinv=False,append=append_original,sims=np.append(sims,num+1))
        inv_resp_gmv = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv[1:] = 1./(resp_gmv)[1:]
        #inv_resp_gmv_TTEETE = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TTEETE[1:] = 1./(resp_gmv_TTEETE)[1:]
        #inv_resp_gmv_TBEB = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TBEB[1:] = 1./(resp_gmv_TBEB)[1:]

        n0 = {'total':0, 'TTEETE':0, 'TBEB':0}
        for i, sim1 in enumerate(sims):
            sim2 = sim1 + 1

            # Get the lensed ij sims
            plm_gmv_ij = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy')
            #plm_gmv_TTEETE_ij = np.load(dir_out+f'/plm_TTEETE_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            #plm_gmv_TBEB_ij = np.load(dir_out+f'/plm_TBEB_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')

            # Now get the ji sims
            plm_gmv_ji = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy')
            #plm_gmv_TTEETE_ji = np.load(dir_out+f'/plm_TTEETE_healqest_gmv_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            #plm_gmv_TBEB_ji = np.load(dir_out+f'/plm_TBEB_healqest_gmv_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')

            if np.any(np.isnan(plm_gmv_ij)):
                print(f'Sim {sim1} is bad!')
                num -= 1
                continue

            # Response correct
            plm_gmv_resp_corr_ij = hp.almxfl(plm_gmv_ij,inv_resp_gmv)
            #plm_gmv_resp_corr_TTEETE_ij = hp.almxfl(plm_gmv_TTEETE_ij,inv_resp_gmv_TTEETE)
            #plm_gmv_resp_corr_TBEB_ij = hp.almxfl(plm_gmv_TBEB_ij,inv_resp_gmv_TBEB)

            plm_gmv_resp_corr_ji = hp.almxfl(plm_gmv_ji,inv_resp_gmv)
            #plm_gmv_resp_corr_TTEETE_ji = hp.almxfl(plm_gmv_TTEETE_ji,inv_resp_gmv_TTEETE)
            #plm_gmv_resp_corr_TBEB_ji = hp.almxfl(plm_gmv_TBEB_ji,inv_resp_gmv_TBEB)

            # Get ij auto spectra <ijij>
            auto = hp.alm2cl(plm_gmv_resp_corr_ij, plm_gmv_resp_corr_ij, lmax=lmax)
            #auto_A = hp.alm2cl(plm_gmv_resp_corr_TTEETE_ij, plm_gmv_resp_corr_TTEETE_ij, lmax=lmax)
            #auto_B = hp.alm2cl(plm_gmv_resp_corr_TBEB_ij, plm_gmv_resp_corr_TBEB_ij, lmax=lmax)

            # Get cross spectra <ijji>
            cross = hp.alm2cl(plm_gmv_resp_corr_ij, plm_gmv_resp_corr_ji, lmax=lmax)
            #cross_A = hp.alm2cl(plm_gmv_resp_corr_TTEETE_ij, plm_gmv_resp_corr_TTEETE_ji, lmax=lmax)
            #cross_B = hp.alm2cl(plm_gmv_resp_corr_TBEB_ij, plm_gmv_resp_corr_TBEB_ji, lmax=lmax)

            n0['total'] += auto + cross
            #n0['TTEETE'] += auto_A + cross_A
            #n0['TBEB'] += auto_B + cross_B

        n0['total'] *= 1/num
        #n0['TTEETE'] *= 1/num
        #n0['TBEB'] *= 1/num

        with open(filename, 'wb') as f:
            pickle.dump(n0, f)

    elif qetype == 'gmv_cinv':
        ests = ['TT', 'EE', 'TE', 'ET', 'TB', 'BT', 'EB', 'BE']
        resps = np.zeros((len(l),len(ests)), dtype=np.complex_)
        inv_resps = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
        for i, est in enumerate(ests):
            #resps[:,i] = get_sim_response(est,config,cinv=True,append=append_original,sims=np.append(sims,num+2),fg_model=fg_model)
            resps[:,i] = get_sim_response(est,config,cinv=True,append=append_original,sims=np.append(sims,num+1),fg_model=fg_model)
            inv_resps[1:,i] = 1/(resps)[1:,i]
        resp = np.sum(resps, axis=1)
        resp_TTEETE = resps[:,0]+resps[:,1]+resps[:,2]+resps[:,3]
        resp_TBEB = resps[:,4]+resps[:,5]+resps[:,6]+resps[:,7]
        inv_resp = np.zeros_like(l,dtype=np.complex_); inv_resp[1:] = 1/(resp)[1:]
        inv_resp_TTEETE = np.zeros_like(l,dtype=np.complex_); inv_resp_TTEETE[1:] = 1/(resp_TTEETE)[1:]
        inv_resp_TBEB = np.zeros_like(l,dtype=np.complex_); inv_resp_TBEB[1:] = 1/(resp_TBEB)[1:]

        n0 = {'total':0, 'TT':0, 'EE':0, 'TE':0, 'ET':0, 'TB':0, 'BT':0, 'EB':0, 'BE':0, 'TTEETE':0, 'TBEB':0}
        for i, sim1 in enumerate(sims):
            sim2 = sim1 + 1

            # Get the lensed ij sims
            if os.path.isfile(dir_out+f'/plm_summed_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv.npy'):
                plm_total_ij = np.load(dir_out+f'/plm_summed_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv.npy')
            else:
                plms_ij = np.zeros((len(np.load(dir_out+f'/plm_TT_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv.npy')),len(ests)), dtype=np.complex_)
                for i, est in enumerate(ests):
                    plms_ij[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv.npy')
                plm_total_ij = np.sum(plms_ij, axis=1)
                np.save(dir_out+f'/plm_summed_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv.npy',plm_total_ij)
                for i, est in enumerate(ests):
                    os.remove(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv.npy')

            # Now get the ji sims
            if os.path.isfile(dir_out+f'/plm_summed_healqest_gmv_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv.npy'):
                plm_total_ji = np.load(dir_out+f'/plm_summed_healqest_gmv_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv.npy')
            else:
                plms_ji = np.zeros((len(np.load(dir_out+f'/plm_TT_healqest_gmv_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv.npy')),len(ests)), dtype=np.complex_)
                for i, est in enumerate(ests):
                    plms_ji[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv.npy')
                plm_total_ji = np.sum(plms_ji, axis=1)
                np.save(dir_out+f'/plm_summed_healqest_gmv_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv.npy',plm_total_ji)
                for i, est in enumerate(ests):
                    os.remove(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv.npy')

            # EIGHT estimators!!!
            #plm_TTEETE_ij = plms_ij[:,0]+plms_ij[:,1]+plms_ij[:,2]+plms_ij[:,3]
            #plm_TTEETE_ji = plms_ji[:,0]+plms_ji[:,1]+plms_ji[:,2]+plms_ji[:,3]
            #plm_TBEB_ij = plms_ij[:,4]+plms_ij[:,5]+plms_ij[:,6]+plms_ij[:,7]
            #plm_TBEB_ji = plms_ji[:,4]+plms_ji[:,5]+plms_ji[:,6]+plms_ji[:,7]

            # Response correct
            plm_total_ij = hp.almxfl(plm_total_ij,inv_resp)
            #plm_TTEETE_ij = hp.almxfl(plm_TTEETE_ij,inv_resp_TTEETE)
            #plm_TBEB_ij = hp.almxfl(plm_TBEB_ij,inv_resp_TBEB)
            #for i, est in enumerate(ests):
            #    plms_ij[:,i] = hp.almxfl(plms_ij[:,i],inv_resps[:,i])

            plm_total_ji = hp.almxfl(plm_total_ji,inv_resp)
            #plm_TTEETE_ji = hp.almxfl(plm_TTEETE_ji,inv_resp_TTEETE)
            #plm_TBEB_ji = hp.almxfl(plm_TBEB_ji,inv_resp_TBEB)
            #for i, est in enumerate(ests):
            #    plms_ji[:,i] = hp.almxfl(plms_ji[:,i],inv_resps[:,i])

            # Get ij auto spectra <ijij>
            auto = hp.alm2cl(plm_total_ij, plm_total_ij, lmax=lmax)
            #auto_TTEETE = hp.alm2cl(plm_TTEETE_ij, plm_TTEETE_ij, lmax=lmax)
            #auto_TBEB = hp.alm2cl(plm_TBEB_ij, plm_TBEB_ij, lmax=lmax)
            #auto_TT = hp.alm2cl(plms_ij[:,0], plms_ij[:,0], lmax=lmax)
            #auto_EE = hp.alm2cl(plms_ij[:,1], plms_ij[:,1], lmax=lmax)
            #auto_TE = hp.alm2cl(plms_ij[:,2], plms_ij[:,2], lmax=lmax)
            #auto_ET = hp.alm2cl(plms_ij[:,3], plms_ij[:,3], lmax=lmax)
            #auto_TB = hp.alm2cl(plms_ij[:,4], plms_ij[:,4], lmax=lmax)
            #auto_BT = hp.alm2cl(plms_ij[:,5], plms_ij[:,5], lmax=lmax)
            #auto_EB = hp.alm2cl(plms_ij[:,6], plms_ij[:,6], lmax=lmax)
            #auto_BE = hp.alm2cl(plms_ij[:,7], plms_ij[:,7], lmax=lmax)

            # Get cross spectra <ijji>
            cross = hp.alm2cl(plm_total_ij, plm_total_ji, lmax=lmax)
            #cross_TTEETE = hp.alm2cl(plm_TTEETE_ij, plm_TTEETE_ji, lmax=lmax)
            #cross_TBEB = hp.alm2cl(plm_TBEB_ij, plm_TBEB_ji, lmax=lmax)
            #cross_TT = hp.alm2cl(plms_ij[:,0], plms_ji[:,0], lmax=lmax)
            #cross_EE = hp.alm2cl(plms_ij[:,1], plms_ji[:,1], lmax=lmax)
            #cross_TE = hp.alm2cl(plms_ij[:,2], plms_ji[:,2], lmax=lmax)
            #cross_ET = hp.alm2cl(plms_ij[:,3], plms_ji[:,3], lmax=lmax)
            #cross_TB = hp.alm2cl(plms_ij[:,4], plms_ji[:,4], lmax=lmax)
            #cross_BT = hp.alm2cl(plms_ij[:,5], plms_ji[:,5], lmax=lmax)
            #cross_EB = hp.alm2cl(plms_ij[:,6], plms_ji[:,6], lmax=lmax)
            #cross_BE = hp.alm2cl(plms_ij[:,7], plms_ji[:,7], lmax=lmax)

            n0['total'] += auto + cross
            #n0['TTEETE'] += auto_TTEETE + cross_TTEETE
            #n0['TBEB'] += auto_TBEB + cross_TBEB
            #n0['TT'] += auto_TT + cross_TT
            #n0['EE'] += auto_EE + cross_EE
            #n0['TE'] += auto_TE + cross_TE
            #n0['ET'] += auto_ET + cross_ET
            #n0['TB'] += auto_TB + cross_TB
            #n0['BT'] += auto_BT + cross_BT
            #n0['EB'] += auto_EB + cross_EB
            #n0['BE'] += auto_BE + cross_BE

        n0['total'] *= 1/num
        #n0['TTEETE'] *= 1/num
        #n0['TBEB'] *= 1/num
        #n0['TT'] *= 1/num
        #n0['EE'] *= 1/num
        #n0['TE'] *= 1/num
        #n0['ET'] *= 1/num
        #n0['TB'] *= 1/num
        #n0['BT'] *= 1/num
        #n0['EB'] *= 1/num
        #n0['BE'] *= 1/num

        with open(filename, 'wb') as f:
            pickle.dump(n0, f)

    elif qetype == 'sqe':
        # SQE response
        ests = ['TT', 'EE', 'TE', 'ET', 'TB', 'BT', 'EB', 'BE']
        resps = np.zeros((len(l),len(ests)), dtype=np.complex_)
        inv_resps = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
        for i, est in enumerate(ests):
            resps[:,i] = get_sim_response(est,config,gmv=False,cinv=False,append=append_original,sims=np.append(sims,num+1),fg_model=fg_model)
            inv_resps[1:,i] = 1/(resps)[1:,i]
        resp = np.sum(resps, axis=1)
        resp_TTEETE = resps[:,0]+resps[:,1]+resps[:,2]+resps[:,3]
        resp_TBEB = resps[:,4]+resps[:,5]+resps[:,6]+resps[:,7]
        inv_resp = np.zeros_like(l,dtype=np.complex_); inv_resp[1:] = 1/(resp)[1:]
        inv_resp_TTEETE = np.zeros_like(l,dtype=np.complex_); inv_resp_TTEETE[1:] = 1/(resp_TTEETE)[1:]
        inv_resp_TBEB = np.zeros_like(l,dtype=np.complex_); inv_resp_TBEB[1:] = 1/(resp_TBEB)[1:]

        n0 = {'total':0, 'TT':0, 'EE':0, 'TE':0, 'ET':0, 'TB':0, 'BT':0, 'EB':0, 'BE':0, 'TTEETE':0, 'TBEB':0}
        for i, sim1 in enumerate(sims):
            sim2 = sim1 + 1

            # Get the lensed ij sims
            if os.path.isfile(dir_out+f'/plm_summed_healqest_sqe_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy'):
                plm_total_ij = np.load(dir_out+f'/plm_summed_healqest_sqe_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy')
            else:
                plms_ij = np.zeros((len(np.load(dir_out+f'/plm_TT_healqest_sqe_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy')),len(ests)), dtype=np.complex_)
                for i, est in enumerate(ests):
                    plms_ij[:,i] = np.load(dir_out+f'/plm_{est}_healqest_sqe_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy')
                plm_total_ij = np.sum(plms_ij, axis=1)
                np.save(dir_out+f'/plm_summed_healqest_sqe_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy',plm_total_ij)
                for i, est in enumerate(ests):
                    os.remove(dir_out+f'/plm_{est}_healqest_sqe_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy')

            # Now get the ji sims
            if os.path.isfile(dir_out+f'/plm_summed_healqest_sqe_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy'):
                plm_total_ji = np.load(dir_out+f'/plm_summed_healqest_sqe_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy')
            else:
                plms_ji = np.zeros((len(np.load(dir_out+f'/plm_TT_healqest_sqe_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy')),len(ests)), dtype=np.complex_)
                for i, est in enumerate(ests):
                    plms_ji[:,i] = np.load(dir_out+f'/plm_{est}_healqest_sqe_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy')
                plm_total_ji = np.sum(plms_ji, axis=1)
                np.save(dir_out+f'/plm_summed_healqest_sqe_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy',plm_total_ji)
                for i, est in enumerate(ests):
                    os.remove(dir_out+f'/plm_{est}_healqest_sqe_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy')

            # EIGHT estimators!!!
            #plm_TTEETE_ij = plms_ij[:,0]+plms_ij[:,1]+plms_ij[:,2]+plms_ij[:,3]
            #plm_TTEETE_ji = plms_ji[:,0]+plms_ji[:,1]+plms_ji[:,2]+plms_ji[:,3]
            #plm_TBEB_ij = plms_ij[:,4]+plms_ij[:,5]+plms_ij[:,6]+plms_ij[:,7]
            #plm_TBEB_ji = plms_ji[:,4]+plms_ji[:,5]+plms_ji[:,6]+plms_ji[:,7]

            # Response correct
            plm_total_ij = hp.almxfl(plm_total_ij,inv_resp)
            #plm_TTEETE_ij = hp.almxfl(plm_TTEETE_ij,inv_resp_TTEETE)
            #plm_TBEB_ij = hp.almxfl(plm_TBEB_ij,inv_resp_TBEB)
            #for i, est in enumerate(ests):
            #    plms_ij[:,i] = hp.almxfl(plms_ij[:,i],inv_resps[:,i])

            plm_total_ji = hp.almxfl(plm_total_ji,inv_resp)
            #plm_TTEETE_ji = hp.almxfl(plm_TTEETE_ji,inv_resp_TTEETE)
            #plm_TBEB_ji = hp.almxfl(plm_TBEB_ji,inv_resp_TBEB)
            #for i, est in enumerate(ests):
            #    plms_ji[:,i] = hp.almxfl(plms_ji[:,i],inv_resps[:,i])

            # Get ij auto spectra <ijij>
            auto = hp.alm2cl(plm_total_ij, plm_total_ij, lmax=lmax)
            #auto_TTEETE = hp.alm2cl(plm_TTEETE_ij, plm_TTEETE_ij, lmax=lmax)
            #auto_TBEB = hp.alm2cl(plm_TBEB_ij, plm_TBEB_ij, lmax=lmax)
            #auto_TT = hp.alm2cl(plms_ij[:,0], plms_ij[:,0], lmax=lmax)
            #auto_EE = hp.alm2cl(plms_ij[:,1], plms_ij[:,1], lmax=lmax)
            #auto_TE = hp.alm2cl(plms_ij[:,2], plms_ij[:,2], lmax=lmax)
            #auto_ET = hp.alm2cl(plms_ij[:,3], plms_ij[:,3], lmax=lmax)
            #auto_TB = hp.alm2cl(plms_ij[:,4], plms_ij[:,4], lmax=lmax)
            #auto_BT = hp.alm2cl(plms_ij[:,5], plms_ij[:,5], lmax=lmax)
            #auto_EB = hp.alm2cl(plms_ij[:,6], plms_ij[:,6], lmax=lmax)
            #auto_BE = hp.alm2cl(plms_ij[:,7], plms_ij[:,7], lmax=lmax)

            # Get cross spectra <ijji>
            cross = hp.alm2cl(plm_total_ij, plm_total_ji, lmax=lmax)
            #cross_TTEETE = hp.alm2cl(plm_TTEETE_ij, plm_TTEETE_ji, lmax=lmax)
            #cross_TBEB = hp.alm2cl(plm_TBEB_ij, plm_TBEB_ji, lmax=lmax)
            #cross_TT = hp.alm2cl(plms_ij[:,0], plms_ji[:,0], lmax=lmax)
            #cross_EE = hp.alm2cl(plms_ij[:,1], plms_ji[:,1], lmax=lmax)
            #cross_TE = hp.alm2cl(plms_ij[:,2], plms_ji[:,2], lmax=lmax)
            #cross_ET = hp.alm2cl(plms_ij[:,3], plms_ji[:,3], lmax=lmax)
            #cross_TB = hp.alm2cl(plms_ij[:,4], plms_ji[:,4], lmax=lmax)
            #cross_BT = hp.alm2cl(plms_ij[:,5], plms_ji[:,5], lmax=lmax)
            #cross_EB = hp.alm2cl(plms_ij[:,6], plms_ji[:,6], lmax=lmax)
            #cross_BE = hp.alm2cl(plms_ij[:,7], plms_ji[:,7], lmax=lmax)

            n0['total'] += auto + cross
            #n0['TTEETE'] += auto_TTEETE + cross_TTEETE
            #n0['TBEB'] += auto_TBEB + cross_TBEB
            #n0['TT'] += auto_TT + cross_TT
            #n0['EE'] += auto_EE + cross_EE
            #n0['TE'] += auto_TE + cross_TE
            #n0['ET'] += auto_ET + cross_ET
            #n0['TB'] += auto_TB + cross_TB
            #n0['BT'] += auto_BT + cross_BT
            #n0['EB'] += auto_EB + cross_EB
            #n0['BE'] += auto_BE + cross_BE

        n0['total'] *= 1/num
        #n0['TTEETE'] *= 1/num
        #n0['TBEB'] *= 1/num
        #n0['TT'] *= 1/num
        #n0['EE'] *= 1/num
        #n0['TE'] *= 1/num
        #n0['ET'] *= 1/num
        #n0['TB'] *= 1/num
        #n0['BT'] *= 1/num
        #n0['EB'] *= 1/num
        #n0['BE'] *= 1/num

        with open(filename, 'wb') as f:
            pickle.dump(n0, f)

    else:
        print('Invalid argument qetype.')

    return n0

def get_n1(sims,qetype,config,append,fg_model='agora'):
    '''
    Get N1 bias. qetype should be 'sqe', 'gmv' or 'gmv_cinv'.
    Returns dictionary containing keys 'total', 'TTEETE', and 'TBEB' for 'gmv'.
    Similarly for 'gmv_cinv' and 'sqe' cases.
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
    filename = dir_out+f'/n1/n1_{num}simpairs_healqest_{qetype}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_resp_from_sims.pkl'

    if os.path.isfile(filename):
        n1 = pickle.load(open(filename,'rb'))

    elif qetype == 'gmv':
        # GMV response
        resp_gmv = get_sim_response('all',config,cinv=False,append=append,sims=np.append(sims,num+1),fg_model=fg_model)
        #resp_gmv_TTEETE = get_sim_response('TTEETE',config,cinv=False,append=append,sims=np.append(sims,num+1))
        #resp_gmv_TBEB = get_sim_response('TBEB',config,cinv=False,append=append,sims=np.append(sims,num+1))
        inv_resp_gmv = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv[1:] = 1./(resp_gmv)[1:]
        #inv_resp_gmv_TTEETE = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TTEETE[1:] = 1./(resp_gmv_TTEETE)[1:]
        #inv_resp_gmv_TBEB = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TBEB[1:] = 1./(resp_gmv_TBEB)[1:]

        n0 = get_n0(sims=sims,qetype=qetype,config=config,
                    append=append,cmbonly=True,fg_model=fg_model)

        n1 = {'total':0, 'TTEETE':0, 'TBEB':0}
        for i, sim in enumerate(sims):
            # These are reconstructions using sims that were lensed with the same phi but different CMB realizations, no foregrounds
            # Get the lensed ij sims
            plm_gmv_ij = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cmbonly_phi1_tqu1tqu2.npy')
            #plm_gmv_TTEETE_ij = np.load(dir_out+f'/plm_TTEETE_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu1tqu2.npy')
            #plm_gmv_TBEB_ij = np.load(dir_out+f'/plm_TBEB_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu1tqu2.npy')

            # Now get the ji sims
            plm_gmv_ji = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cmbonly_phi1_tqu2tqu1.npy')
            #plm_gmv_TTEETE_ji = np.load(dir_out+f'/plm_TTEETE_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu2tqu1.npy')
            #plm_gmv_TBEB_ji = np.load(dir_out+f'/plm_TBEB_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu2tqu1.npy')

            if np.any(np.isnan(plm_gmv_ij)):
                print(f'Sim {sim} is bad!')
                num -= 1
                continue

            # Response correct
            plm_gmv_resp_corr_ij = hp.almxfl(plm_gmv_ij,inv_resp_gmv)
            #plm_gmv_resp_corr_TTEETE_ij = hp.almxfl(plm_gmv_TTEETE_ij,inv_resp_gmv_TTEETE)
            #plm_gmv_resp_corr_TBEB_ij = hp.almxfl(plm_gmv_TBEB_ij,inv_resp_gmv_TBEB)

            plm_gmv_resp_corr_ji = hp.almxfl(plm_gmv_ji,inv_resp_gmv)
            #plm_gmv_resp_corr_TTEETE_ji = hp.almxfl(plm_gmv_TTEETE_ji,inv_resp_gmv_TTEETE)
            #plm_gmv_resp_corr_TBEB_ji = hp.almxfl(plm_gmv_TBEB_ji,inv_resp_gmv_TBEB)

            # Get ij auto spectra <ijij>
            auto = hp.alm2cl(plm_gmv_resp_corr_ij, plm_gmv_resp_corr_ij, lmax=lmax)
            #auto_A = hp.alm2cl(plm_gmv_resp_corr_TTEETE_ij, plm_gmv_resp_corr_TTEETE_ij, lmax=lmax)
            #auto_B = hp.alm2cl(plm_gmv_resp_corr_TBEB_ij, plm_gmv_resp_corr_TBEB_ij, lmax=lmax)

            # Get cross spectra <ijji>
            cross = hp.alm2cl(plm_gmv_resp_corr_ij, plm_gmv_resp_corr_ji, lmax=lmax)
            #cross_A = hp.alm2cl(plm_gmv_resp_corr_TTEETE_ij, plm_gmv_resp_corr_TTEETE_ji, lmax=lmax)
            #cross_B = hp.alm2cl(plm_gmv_resp_corr_TBEB_ij, plm_gmv_resp_corr_TBEB_ji, lmax=lmax)

            n1['total'] += auto + cross
            #n1['TTEETE'] += auto_A + cross_A
            #n1['TBEB'] += auto_B + cross_B

        n1['total'] *= 1/num
        #n1['TTEETE'] *= 1/num
        #n1['TBEB'] *= 1/num

        n1['total'] -= n0['total']
        #n1['TTEETE'] -= n0['TTEETE']
        #n1['TBEB'] -= n0['TBEB']

        with open(filename, 'wb') as f:
            pickle.dump(n1, f)

    elif qetype == 'gmv_cinv':
        ests = ['TT','EE', 'TE', 'ET', 'TB', 'BT', 'EB', 'BE']
        resps = np.zeros((len(l),len(ests)), dtype=np.complex_)
        inv_resps = np.zeros((len(l),len(ests)),dtype=np.complex_)
        for i, est in enumerate(ests):
            resps[:,i] = get_sim_response(est,config,cinv=True,append=append,sims=np.append(sims,num+1),fg_model=fg_model)
            inv_resps[1:,i] = 1/(resps)[1:,i]
        resp = np.sum(resps, axis=1)
        inv_resp = np.zeros_like(l,dtype=np.complex_); inv_resp[1:] = 1/(resp)[1:]

        n0 = get_n0(sims=sims,qetype=qetype,config=config,
                    append=append,cmbonly=True,fg_model=fg_model)

        n1 = {'total':0, 'TT':0, 'EE':0, 'TE':0, 'ET':0, 'TB':0, 'BT':0, 'EB':0, 'BE':0}
        for i, sim in enumerate(sims):
            # Get the lensed ij sims
            if os.path.isfile(dir_out+f'/plm_summed_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cmbonly_phi1_tqu1tqu2_cinv.npy'):
                plm_total_ij = np.load(dir_out+f'/plm_summed_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cmbonly_phi1_tqu1tqu2_cinv.npy')
            else:
                plms_ij = np.zeros((len(np.load(dir_out+f'/plm_TT_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cmbonly_phi1_tqu1tqu2_cinv.npy')),len(ests)), dtype=np.complex_)
                for i, est in enumerate(ests):
                    plms_ij[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cmbonly_phi1_tqu1tqu2_cinv.npy')
                plm_total_ij = np.sum(plms_ij, axis=1)
                np.save(dir_out+f'/plm_summed_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cmbonly_phi1_tqu1tqu2_cinv.npy',plm_total_ij)
                for i, est in enumerate(ests):
                    os.remove(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cmbonly_phi1_tqu1tqu2_cinv.npy')

            # Now get the ji sims
            if os.path.isfile(dir_out+f'/plm_summed_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cmbonly_phi1_tqu2tqu1_cinv.npy'):
                plm_total_ji = np.load(dir_out+f'/plm_summed_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cmbonly_phi1_tqu2tqu1_cinv.npy')
            else:
                plms_ji = np.zeros((len(np.load(dir_out+f'/plm_TT_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cmbonly_phi1_tqu2tqu1_cinv.npy')),len(ests)), dtype=np.complex_)
                for i, est in enumerate(ests):
                    plms_ji[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cmbonly_phi1_tqu2tqu1_cinv.npy')
                plm_total_ji = np.sum(plms_ji, axis=1)
                np.save(dir_out+f'/plm_summed_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cmbonly_phi1_tqu2tqu1_cinv.npy',plm_total_ji)
                for i, est in enumerate(ests):
                    os.remove(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cmbonly_phi1_tqu2tqu1_cinv.npy')

            # Response correct
            plm_total_ij = hp.almxfl(plm_total_ij,inv_resp)
            #plm_TT_ij = hp.almxfl(plm_TT_ij,inv_resps[:,0])
            #plm_EE_ij = hp.almxfl(plm_EE_ij,inv_resps[:,1])
            #plm_TE_ij = hp.almxfl(plm_TE_ij,inv_resps[:,2])
            #plm_ET_ij = hp.almxfl(plm_ET_ij,inv_resps[:,3])
            #plm_TB_ij = hp.almxfl(plm_TB_ij,inv_resps[:,4])
            #plm_BT_ij = hp.almxfl(plm_BT_ij,inv_resps[:,5])
            #plm_EB_ij = hp.almxfl(plm_EB_ij,inv_resps[:,6])
            #plm_BE_ij = hp.almxfl(plm_BE_ij,inv_resps[:,7])

            plm_total_ji = hp.almxfl(plm_total_ji,inv_resp)
            #plm_TT_ji = hp.almxfl(plm_TT_ji,inv_resps[:,0])
            #plm_EE_ji = hp.almxfl(plm_EE_ji,inv_resps[:,1])
            #plm_TE_ji = hp.almxfl(plm_TE_ji,inv_resps[:,2])
            #plm_ET_ji = hp.almxfl(plm_ET_ji,inv_resps[:,3])
            #plm_TB_ji = hp.almxfl(plm_TB_ji,inv_resps[:,4])
            #plm_BT_ji = hp.almxfl(plm_BT_ji,inv_resps[:,5])
            #plm_EB_ji = hp.almxfl(plm_EB_ji,inv_resps[:,6])
            #plm_BE_ji = hp.almxfl(plm_BE_ji,inv_resps[:,7])

            # Get ij auto spectra <ijij>
            auto = hp.alm2cl(plm_total_ij, plm_total_ij, lmax=lmax)
            #auto_TT = hp.alm2cl(plm_TT_ij, plm_TT_ij, lmax=lmax)
            #auto_EE = hp.alm2cl(plm_EE_ij, plm_EE_ij, lmax=lmax)
            #auto_TE = hp.alm2cl(plm_TE_ij, plm_TE_ij, lmax=lmax)
            #auto_ET = hp.alm2cl(plm_ET_ij, plm_ET_ij, lmax=lmax)
            #auto_TB = hp.alm2cl(plm_TB_ij, plm_TB_ij, lmax=lmax)
            #auto_BT = hp.alm2cl(plm_BT_ij, plm_BT_ij, lmax=lmax)
            #auto_EB = hp.alm2cl(plm_EB_ij, plm_EB_ij, lmax=lmax)
            #auto_BE = hp.alm2cl(plm_BE_ij, plm_BE_ij, lmax=lmax)

            # Get cross spectra <ijji>
            cross = hp.alm2cl(plm_total_ij, plm_total_ji, lmax=lmax)
            #cross_TT = hp.alm2cl(plm_TT_ij, plm_TT_ji, lmax=lmax)
            #cross_EE = hp.alm2cl(plm_EE_ij, plm_EE_ji, lmax=lmax)
            #cross_TE = hp.alm2cl(plm_TE_ij, plm_TE_ji, lmax=lmax)
            #cross_ET = hp.alm2cl(plm_ET_ij, plm_ET_ji, lmax=lmax)
            #cross_TB = hp.alm2cl(plm_TB_ij, plm_TB_ji, lmax=lmax)
            #cross_BT = hp.alm2cl(plm_BT_ij, plm_BT_ji, lmax=lmax)
            #cross_EB = hp.alm2cl(plm_EB_ij, plm_EB_ji, lmax=lmax)
            #cross_BE = hp.alm2cl(plm_BE_ij, plm_BE_ji, lmax=lmax)

            n1['total'] += auto + cross
            #n1['TT'] += auto_TT + cross_TT
            #n1['EE'] += auto_EE + cross_EE
            #n1['TE'] += auto_TE + cross_TE
            #n1['ET'] += auto_ET + cross_ET
            #n1['TB'] += auto_TB + cross_TB
            #n1['BT'] += auto_BT + cross_BT
            #n1['EB'] += auto_EB + cross_EB
            #n1['BE'] += auto_BE + cross_BE

        n1['total'] *= 1/num
        #n1['TT'] *= 1/num
        #n1['EE'] *= 1/num
        #n1['TE'] *= 1/num
        #n1['ET'] *= 1/num
        #n1['TB'] *= 1/num
        #n1['BT'] *= 1/num
        #n1['EB'] *= 1/num
        #n1['BE'] *= 1/num

        n1['total'] -= n0['total']
        #n1['TT'] -= n0['TT']
        #n1['EE'] -= n0['EE']
        #n1['TE'] -= n0['TE']
        #n1['ET'] -= n0['ET']
        #n1['TB'] -= n0['TB']
        #n1['BT'] -= n0['BT']
        #n1['EB'] -= n0['EB']
        #n1['BE'] -= n0['BE']

        with open(filename, 'wb') as f:
            pickle.dump(n1, f)

    elif qetype == 'sqe':
        # Get SQE response
        ests = ['TT','EE', 'TE', 'ET', 'TB', 'BT', 'EB', 'BE']
        resps = np.zeros((len(l),len(ests)), dtype=np.complex_)
        inv_resps = np.zeros((len(l),len(ests)),dtype=np.complex_)
        for i, est in enumerate(ests):
            resps[:,i] = get_sim_response(est,config,gmv=False,cinv=False,append=append,sims=np.append(sims,num+1),fg_model=fg_model)
            inv_resps[1:,i] = 1/(resps)[1:,i]
        resp = np.sum(resps, axis=1)
        inv_resp = np.zeros_like(l,dtype=np.complex_); inv_resp[1:] = 1/(resp)[1:]

        n0 = get_n0(sims=sims,qetype=qetype,config=config,
                    append=append,cmbonly=True,fg_model=fg_model)

        n1 = {'total':0, 'TT':0, 'EE':0, 'TE':0, 'ET':0, 'TB':0, 'BT':0, 'EB':0, 'BE':0}
        for i, sim in enumerate(sims):
            # Get the lensed ij sims
            if os.path.isfile(dir_out+f'/plm_summed_healqest_sqe_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cmbonly_phi1_tqu1tqu2.npy'):
                plm_total_ij = np.load(dir_out+f'/plm_summed_healqest_sqe_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cmbonly_phi1_tqu1tqu2.npy')
            else:
                plms_ij = np.zeros((len(np.load(dir_out+f'/plm_TT_healqest_sqe_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cmbonly_phi1_tqu1tqu2.npy')),len(ests)), dtype=np.complex_)
                for i, est in enumerate(ests):
                    plms_ij[:,i] = np.load(dir_out+f'/plm_{est}_healqest_sqe_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cmbonly_phi1_tqu1tqu2.npy')
                plm_total_ij = np.sum(plms_ij, axis=1)
                np.save(dir_out+f'/plm_summed_healqest_sqe_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cmbonly_phi1_tqu1tqu2.npy',plm_total_ij)
                for i, est in enumerate(ests):
                    os.remove(dir_out+f'/plm_{est}_healqest_sqe_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cmbonly_phi1_tqu1tqu2.npy')

            # Now get the ji sims
            if os.path.isfile(dir_out+f'/plm_summed_healqest_sqe_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cmbonly_phi1_tqu2tqu1.npy'):
                plm_total_ji = np.load(dir_out+f'/plm_summed_healqest_sqe_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cmbonly_phi1_tqu2tqu1.npy')
            else:
                plms_ji = np.zeros((len(np.load(dir_out+f'/plm_TT_healqest_sqe_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cmbonly_phi1_tqu2tqu1.npy')),len(ests)), dtype=np.complex_)
                for i, est in enumerate(ests):
                    plms_ji[:,i] = np.load(dir_out+f'/plm_{est}_healqest_sqe_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cmbonly_phi1_tqu2tqu1.npy')
                plm_total_ji = np.sum(plms_ji, axis=1)
                np.save(dir_out+f'/plm_summed_healqest_sqe_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cmbonly_phi1_tqu2tqu1.npy',plm_total_ji)
                for i, est in enumerate(ests):
                    os.remove(dir_out+f'/plm_{est}_healqest_sqe_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cmbonly_phi1_tqu2tqu1.npy')

            # Response correct healqest
            plm_total_ij = hp.almxfl(plm_total_ij,inv_resp)
            #plm_TT_ij = hp.almxfl(plm_TT_ij,inv_resps[:,0])
            #plm_EE_ij = hp.almxfl(plm_EE_ij,inv_resps[:,1])
            #plm_TE_ij = hp.almxfl(plm_TE_ij,inv_resps[:,2])
            #plm_ET_ij = hp.almxfl(plm_ET_ij,inv_resps[:,3])
            #plm_TB_ij = hp.almxfl(plm_TB_ij,inv_resps[:,4])
            #plm_BT_ij = hp.almxfl(plm_BT_ij,inv_resps[:,5])
            #plm_EB_ij = hp.almxfl(plm_EB_ij,inv_resps[:,6])
            #plm_BE_ij = hp.almxfl(plm_BE_ij,inv_resps[:,7])

            plm_total_ji = hp.almxfl(plm_total_ji,inv_resp)
            #plm_TT_ji = hp.almxfl(plm_TT_ji,inv_resps[:,0])
            #plm_EE_ji = hp.almxfl(plm_EE_ji,inv_resps[:,1])
            #plm_TE_ji = hp.almxfl(plm_TE_ji,inv_resps[:,2])
            #plm_ET_ji = hp.almxfl(plm_ET_ji,inv_resps[:,3])
            #plm_TB_ji = hp.almxfl(plm_TB_ji,inv_resps[:,4])
            #plm_BT_ji = hp.almxfl(plm_BT_ji,inv_resps[:,5])
            #plm_EB_ji = hp.almxfl(plm_EB_ji,inv_resps[:,6])
            #plm_BE_ji = hp.almxfl(plm_BE_ji,inv_resps[:,7])

            # Get ij auto spectra <ijij>
            auto = hp.alm2cl(plm_total_ij, plm_total_ij, lmax=lmax)
            #auto_TT = hp.alm2cl(plm_TT_ij, plm_TT_ij, lmax=lmax)
            #auto_EE = hp.alm2cl(plm_EE_ij, plm_EE_ij, lmax=lmax)
            #auto_TE = hp.alm2cl(plm_TE_ij, plm_TE_ij, lmax=lmax)
            #auto_ET = hp.alm2cl(plm_ET_ij, plm_ET_ij, lmax=lmax)
            #auto_TB = hp.alm2cl(plm_TB_ij, plm_TB_ij, lmax=lmax)
            #auto_BT = hp.alm2cl(plm_BT_ij, plm_BT_ij, lmax=lmax)
            #auto_EB = hp.alm2cl(plm_EB_ij, plm_EB_ij, lmax=lmax)
            #auto_BE = hp.alm2cl(plm_BE_ij, plm_BE_ij, lmax=lmax)

            # Get cross spectra <ijji>
            cross = hp.alm2cl(plm_total_ij, plm_total_ji, lmax=lmax)
            #cross_TT = hp.alm2cl(plm_TT_ij, plm_TT_ji, lmax=lmax)
            #cross_EE = hp.alm2cl(plm_EE_ij, plm_EE_ji, lmax=lmax)
            #cross_TE = hp.alm2cl(plm_TE_ij, plm_TE_ji, lmax=lmax)
            #cross_ET = hp.alm2cl(plm_ET_ij, plm_ET_ji, lmax=lmax)
            #cross_TB = hp.alm2cl(plm_TB_ij, plm_TB_ji, lmax=lmax)
            #cross_BT = hp.alm2cl(plm_BT_ij, plm_BT_ji, lmax=lmax)
            #cross_EB = hp.alm2cl(plm_EB_ij, plm_EB_ji, lmax=lmax)
            #cross_BE = hp.alm2cl(plm_BE_ij, plm_BE_ji, lmax=lmax)

            n1['total'] += auto + cross
            #n1['TT'] += auto_TT + cross_TT
            #n1['EE'] += auto_EE + cross_EE
            #n1['TE'] += auto_TE + cross_TE
            #n1['ET'] += auto_ET + cross_ET
            #n1['TB'] += auto_TB + cross_TB
            #n1['BT'] += auto_BT + cross_BT
            #n1['EB'] += auto_EB + cross_EB
            #n1['BE'] += auto_BE + cross_BE

        n1['total'] *= 1/num
        #n1['TT'] *= 1/num
        #n1['EE'] *= 1/num
        #n1['TE'] *= 1/num
        #n1['ET'] *= 1/num
        #n1['TB'] *= 1/num
        #n1['BT'] *= 1/num
        #n1['EB'] *= 1/num
        #n1['BE'] *= 1/num

        n1['total'] -= n0['total']
        #n1['TT'] -= n0['TT']
        #n1['EE'] -= n0['EE']
        #n1['TE'] -= n0['TE']
        #n1['ET'] -= n0['ET']
        #n1['TB'] -= n0['TB']
        #n1['BT'] -= n0['BT']
        #n1['EB'] -= n0['EB']
        #n1['BE'] -= n0['BE']

        with open(filename, 'wb') as f:
            pickle.dump(n1, f)

    else:
        print('Invalid argument qetype.')

    return n1

def get_sim_response(est,config,cinv,append,sims,filename=None,gmv=True,fg_model='agora'):
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
        if cinv:
            fn += f'_gmv_cinv_est{est}'
        elif gmv:
            fn += f'_gmv_est{est}'
        else:
            fn += f'_sqe_est{est}'
        fn += f'_lmaxT{lmaxT}_lmaxP{lmaxP}_lmin{lmin}_cltype{cltype}_{fg_model}_{append}'
        filename = dir_out+f'/resp/sim_resp_{num}sims{fn}.npy'

    if os.path.isfile(filename):
        print('Loading from existing file!')
        sim_resp = np.load(filename)
    else:
        # File doesn't exist!
        cross_uncorrected_all = 0
        auto_input_all = 0
        for ii, sim in enumerate(sims):
            # Load plm
            if cinv:
                plm = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv.npy')
            elif gmv:
                plm = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy')
            else:
                plm = np.load(dir_out+f'/plm_{est}_healqest_sqe_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy')
            if np.any(np.isnan(plm)):
                print(f'Sim {sim} is bad!')
                num -= 1
                continue
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

analyze(config_file='test_yuka.yaml',compare=True,sqe=True,fg_model='agora')
analyze(config_file='test_yuka_lmaxT3500.yaml',compare=True,sqe=True,fg_model='agora')
analyze(config_file='test_yuka_lmaxT4000.yaml',compare=True,sqe=True,fg_model='agora')

#analyze(config_file='test_yuka.yaml',compare=False,fg_model='websky')
#analyze(config_file='test_yuka_lmaxT3500.yaml',compare=False,fg_model='websky')
#analyze(config_file='test_yuka_lmaxT4000.yaml',compare=False,fg_model='websky')
#analyze(config_file='test_yuka_lmaxT3500.yaml',compare=False,fg_model='agora',lbins = np.array((50,1000,2000,3000)))
#analyze(config_file='test_yuka_lmaxT3500.yaml',compare=True,sqe=False,fg_model='agora')

'''
est='TTEETE';cinv=True;append='standard';sims=np.arange(250)+1;gmv=True;fg_model='agora'
config = utils.parse_yaml('test_yuka_lmaxT3500.yaml')
lmax = config['lensrec']['Lmax']
lmaxT = config['lensrec']['lmaxT']
lmaxP = config['lensrec']['lmaxP']
lmin = config['lensrec']['lminT']
nside = config['lensrec']['nside']
cltype = config['lensrec']['cltype']
dir_out = config['dir_out']
l = np.arange(0,lmax+1)
num = len(sims)
fn = ''
fn += f'_gmv_cinv_est{est}'
fn += f'_lmaxT{lmaxT}_lmaxP{lmaxP}_lmin{lmin}_cltype{cltype}_{fg_model}_{append}'
filename = dir_out+f'/resp/sim_resp_{num}sims{fn}.npy'
# File doesn't exist!
cross_uncorrected_all = 0
auto_input_all = 0
for ii, sim in enumerate(sims):
    # Load plm
    plm = 0
    ests = ['TT','EE', 'TE', 'ET']
    for e in ests:
        plm += np.load(dir_out+f'/plm_{e}_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_cinv.npy')
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
'''
