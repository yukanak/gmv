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
            config_file='test_yuka_lmaxT4000.yaml',
            append='mh',withT3=False,
            n0=True,n1=True,
            lbins=np.logspace(np.log10(50),np.log10(3000),20),
            compare=False,sqe=False):
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

    # Get response
    ests = ['T1T2', 'T2T1', 'EE', 'TE', 'ET', 'TB', 'BT', 'EB', 'BE']
    resps = np.zeros((len(l),len(ests)), dtype=np.complex_)
    inv_resps = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
    for i, est in enumerate(ests):
        resps[:,i] = get_sim_response(est,config,cinv=True,append=append,sims=sims,gmv=True,withT3=withT3)
        inv_resps[1:,i] = 1/(resps)[1:,i]
    resp = 0.5*resps[:,0]+0.5*resps[:,1]+np.sum(resps[:,2:], axis=1)
    inv_resp = np.zeros_like(l,dtype=np.complex_); inv_resp[1:] = 1/(resp)[1:]
    if compare and sqe:
        resps_sqe = np.zeros((len(l),len(ests)), dtype=np.complex_)
        inv_resps_sqe = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
        for i, est in enumerate(ests):
            resps_sqe[:,i] = get_sim_response(est,config,cinv=False,gmv=False,append=append,sims=sims,withT3=withT3)
            inv_resps_sqe[1:,i] = 1/(resps_sqe)[1:,i]
        resp_sqe = 0.5*resps_sqe[:,0]+0.5*resps_sqe[:,1]+np.sum(resps_sqe[:,2:], axis=1)
        inv_resp_sqe = np.zeros_like(l,dtype=np.complex_); inv_resp_sqe[1:] = 1/(resp_sqe)[1:]
    elif compare:
        # Original (not cinv-style) GMV response
        resp_gmv = get_sim_response('all',config,cinv=False,append=append,sims=sims,withT3=withT3)
        #resp_gmv_TTEETE = get_sim_response('TTEETE',config,cinv=False,append=append,sims=sims,withT3=withT3)
        #resp_gmv_TBEB = get_sim_response('TBEB',config,cinv=False,append=append,sims=sims,withT3=withT3)
        inv_resp_gmv = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv[1:] = 1./(resp_gmv)[1:]
        #inv_resp_gmv_TTEETE = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TTEETE[1:] = 1./(resp_gmv_TTEETE)[1:]
        #inv_resp_gmv_TBEB = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TBEB[1:] = 1./(resp_gmv_TBEB)[1:]

    if n0:
        # Get N0 correction (remember these still need (l*(l+1))**2/4 factor to convert to kappa)
        n0_cinv = get_n0(sims=n0_n1_sims,qetype='gmv_cinv',config=config,
                         append=append,withT3=withT3)
        n0_cinv_total = n0_cinv['total'] * (l*(l+1))**2/4
        if compare and sqe:
            n0_sqe = get_n0(sims=n0_n1_sims,qetype='sqe',config=config,append=append,withT3=withT3)
            n0_sqe_total = n0_sqe['total'] * (l*(l+1))**2/4
        elif compare:
            n0_gmv = get_n0(sims=n0_n1_sims,qetype='gmv',config=config,append=append,withT3=withT3)
            n0_gmv_total = n0_gmv['total'] * (l*(l+1))**2/4

    if n1:
        n1_cinv = get_n1(sims=n0_n1_sims,qetype='gmv_cinv',config=config,
                         append=append,withT3=withT3)
        n1_cinv_total = n1_cinv['total'] * (l*(l+1))**2/4
        if compare and sqe:
            n1_sqe = get_n1(sims=n0_n1_sims,qetype='sqe',config=config,append=append,withT3=withT3)
            n1_sqe_total = n1_sqe['total'] * (l*(l+1))**2/4
        elif compare:
            n1_gmv = get_n1(sims=n0_n1_sims,qetype='gmv',config=config,append=append,withT3=withT3)
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
    binned_temp_gmv = np.zeros((len(sims),len(l)),dtype=np.complex_)
    binned_temp_cinv = np.zeros((len(sims),len(l)),dtype=np.complex_)
    binned_temp_sqe = np.zeros((len(sims),len(l)),dtype=np.complex_)

    for ii, sim in enumerate(sims):
        # Load cinv-style GMV plms
        plms = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_noT3.npy')),len(ests)), dtype=np.complex_)
        for i, est in enumerate(ests):
            if withT3:
                plms[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_withT3.npy')
            else:
                plms[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_noT3.npy')
        plm = 0.5*np.sum(plms[:,:2], axis=1)+np.sum(plms[:,2:], axis=1)

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
            binned_temp_cinv = [auto_debiased[digitized == i].mean() for i in range(1, len(lbins))]

            # If debiasing, get the binned ratio against input
            input_plm = hp.read_alm(f'/oak/stanford/orgs/kipac//users/yukanaka/lensing19-20/inputcmb/phi/phi_lmax_{lmax}/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_phi1_seed{sim}_lmax{lmax}.alm')
            auto_input = hp.alm2cl(input_plm, input_plm, lmax=lmax) * (l*(l+1))**2/4
            # Bin!
            binned_auto_cinv_debiased = [auto_debiased[digitized == i].mean() for i in range(1, len(lbins))]
            binned_auto_input = [auto_input[digitized == i].mean() for i in range(1, len(lbins))]
            # Get ratio
            ratio_cinv[ii,:] = np.array(binned_auto_cinv_debiased) / np.array(binned_auto_input)

        if compare and sqe:
            # Load SQE plms
            plms_sqe = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_noT3.npy')),len(ests)), dtype=np.complex_)
            for i, est in enumerate(ests):
                if withT3:
                    plms_sqe[:,i] = np.load(dir_out+f'/plm_{est}_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_withT3.npy')
                else:
                    plms_sqe[:,i] = np.load(dir_out+f'/plm_{est}_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_noT3.npy')
            plm_sqe = 0.5*np.sum(plms_sqe[:,:2], axis=1)+np.sum(plms_sqe[:,2:], axis=1)

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
                binned_temp_sqe = [auto_sqe_debiased[digitized == i].mean() for i in range(1, len(lbins))]

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
            if withT3:
                plm_gmv = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_withT3.npy')
            else:
                plm_gmv = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_noT3.npy')

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
                binned_temp_gmv = [auto_gmv_debiased[digitized == i].mean() for i in range(1, len(lbins))]

                # If debiasing, get the binned ratio against input
                input_plm = hp.read_alm(f'/oak/stanford/orgs/kipac//users/yukanaka/lensing19-20/inputcmb/phi/phi_lmax_{lmax}/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_phi1_seed{sim}_lmax{lmax}.alm')
                auto_input = hp.alm2cl(input_plm, input_plm, lmax=lmax) * (l*(l+1))**2/4
                # Bin!
                binned_auto_gmv_debiased = [auto_gmv_debiased[digitized == i].mean() for i in range(1, len(lbins))]
                binned_auto_input = [auto_input[digitized == i].mean() for i in range(1, len(lbins))]
                # Get ratio
                ratio_gmv[ii,:] = np.array(binned_auto_gmv_debiased) / np.array(binned_auto_input)

    # GET THE UNCERTAINTIES (error bar from spread of sims - measurement uncertainty of bandpowers)
    uncertainty = np.zeros((len(l),2), dtype=np.complex_)
    binned_uncertainty = np.zeros((len(bin_centers),2), dtype=np.complex_)
    uncertainty[:,0] = np.std(temp_cinv,axis=0)
    #uncertainty[:,1] = np.std(temp_cinv_TT,axis=0)
    binned_uncertainty[:,0] = np.std(binned_temp_cinv,axis=0)
    #binned_uncertainty[:,1] = np.std(binned_temp_cinv_TT,axis=0)
    if withT3:
        pass
        #np.save(dir_out+f'/agora_reconstruction/measurement_uncertainty_lmaxT{lmaxT}_{append}_cinv_withT3.npy',uncertainty)
        #np.save(dir_out+f'/agora_reconstruction/binned_measurement_uncertainty_lmaxT{lmaxT}_{append}_cinv_withT3.npy',binned_uncertainty)
    else:
        pass
        #np.save(dir_out+f'/agora_reconstruction/measurement_uncertainty_lmaxT{lmaxT}_{append}_cinv_noT3.npy',uncertainty)
        #np.save(dir_out+f'/agora_reconstruction/binned_measurement_uncertainty_lmaxT{lmaxT}_{append}_cinv_noT3.npy',binned_uncertainty)
    if compare and sqe:
        uncertainty = np.zeros((len(l),2), dtype=np.complex_)
        binned_uncertainty = np.zeros((len(bin_centers),2), dtype=np.complex_)
        uncertainty[:,0] = np.std(temp_sqe,axis=0)
        binned_uncertainty[:,0] = np.std(binned_temp_sqe,axis=0)
        if withT3:
            pass
            #np.save(dir_out+f'/agora_reconstruction/measurement_uncertainty_lmaxT{lmaxT}_{append}_sqe_withT3.npy',uncertainty)
            #np.save(dir_out+f'/agora_reconstruction/binned_measurement_uncertainty_lmaxT{lmaxT}_{append}_sqe_withT3.npy',binned_uncertainty)
        else:
            pass
            #np.save(dir_out+f'/agora_reconstruction/measurement_uncertainty_lmaxT{lmaxT}_{append}_sqe_noT3.npy',uncertainty)
            #np.save(dir_out+f'/agora_reconstruction/binned_measurement_uncertainty_lmaxT{lmaxT}_{append}_sqe_noT3.npy',binned_uncertainty)
    elif compare:
        uncertainty = np.zeros((len(l),2), dtype=np.complex_)
        binned_uncertainty = np.zeros((len(bin_centers),2), dtype=np.complex_)
        uncertainty[:,0] = np.std(temp_gmv,axis=0)
        binned_uncertainty[:,0] = np.std(binned_temp_gmv,axis=0)
        if withT3:
            pass
            #np.save(dir_out+f'/agora_reconstruction/measurement_uncertainty_lmaxT{lmaxT}_{append}_gmv_withT3.npy',uncertainty)
            #np.save(dir_out+f'/agora_reconstruction/binned_measurement_uncertainty_lmaxT{lmaxT}_{append}_gmv_withT3.npy',binned_uncertainty)
        else:
            pass
            #np.save(dir_out+f'/agora_reconstruction/measurement_uncertainty_lmaxT{lmaxT}_{append}_gmv_noT3.npy',uncertainty)
            #np.save(dir_out+f'/agora_reconstruction/binned_measurement_uncertainty_lmaxT{lmaxT}_{append}_gmv_noT3.npy',binned_uncertainty)

    # Average
    auto_cinv_avg = auto_cinv_all / num
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

    # Theory spectrum
    clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
    ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
    clkk = slpp * (l*(l+1))**2/4

    # Plot
    plt.figure(0)
    plt.clf()
    plt.plot(l, auto_cinv_debiased_avg, color='lightcoral', linestyle='-', alpha=0.5, label=f'Auto Spectrum (cinv-style GMV)')
    if compare and sqe:
        plt.plot(l, auto_sqe_debiased_avg, color='cornflowerblue', linestyle='-', label="Auto Spectrum (SQE)")
    elif compare:
        plt.plot(l, auto_gmv_debiased_avg, color='cornflowerblue', linestyle='-', label="Auto Spectrum (GMV)")

    plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')

    plt.plot(bin_centers, binned_auto_cinv_debiased_avg, color='firebrick', marker='o', linestyle='None', ms=3, label="Auto Spectrum (Cinv-style GMV)")
    if compare and sqe:
        plt.plot(bin_centers, binned_auto_sqe_debiased_avg, color='darkblue', marker='o', linestyle='None', ms=3, label="Auto Spectrum (SQE)")
    elif compare:
        plt.plot(bin_centers, binned_auto_gmv_debiased_avg, color='darkblue', marker='o', linestyle='None', ms=3, label="Auto Spectrum (GMV)")

    plt.ylabel("$C_\ell^{\kappa\kappa}$")
    plt.xlabel('$\ell$')
    plt.title(f'Spectra Averaged over {num} Sims')
    plt.legend(loc='upper left', fontsize='x-small', bbox_to_anchor=(1,0.5))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,lmax)
    plt.ylim(1e-9,1e-6)
    if n1:
        plt.savefig(dir_out+f'/figs/{num}_sims_comparison_{append}_resp_from_sims_n0n1subtracted_lmaxT{lmaxT}.png',bbox_inches='tight')
    else:
        plt.savefig(dir_out+f'/figs/{num}_sims_comparison_{append}_resp_from_sims_n0subtracted_lmaxT{lmaxT}.png',bbox_inches='tight')

    plt.figure(1)
    plt.clf()
    # Ratios with error bars
    plt.axhline(y=1, color='k', linestyle='--')
    plt.errorbar(bin_centers,ratio_cinv,yerr=errorbars_cinv,color='firebrick', alpha=0.5, marker='o', linestyle='None', ms=3, label="Ratio Cinv-style GMV/Input")
    if compare and sqe:
        plt.errorbar(bin_centers,ratio_sqe,yerr=errorbars_sqe,color='darkblue', marker='o', linestyle='None', ms=3, label="Ratio SQE/Input")
    elif compare:
        plt.errorbar(bin_centers,ratio_gmv,yerr=errorbars_gmv,color='darkblue', marker='o', linestyle='None', ms=3, label="Ratio GMV/Input")
    #TODO
    #results_mh_12ests_lmaxT3500 = pickle.load(open('reconstruction_lmaxT3500_12ests_mh.pkl','rb'))
    #plt.errorbar(bin_centers,results_mh_12ests_lmaxT3500['ratio_cinv'],yerr=results_mh_12ests_lmaxT3500['errorbars_cinv'],color='forestgreen', alpha=0.5, marker='o', linestyle='None', ms=3, label="Ratio Cinv-style GMV/Input, 12 estimators")
    plt.xlabel('$\ell$')
    plt.title(f'Spectra Averaged over {num} Sims')
    plt.legend(loc='lower left', fontsize='x-small')
    plt.xscale('log')
    plt.ylim(0.98,1.02)
    plt.xlim(10,lmax)
    if n1:
        plt.savefig(dir_out+f'/figs/{num}_sims_comparison_{append}_resp_from_sims_n0n1subtracted_binnedratio_lmaxT{lmaxT}.png',bbox_inches='tight')
    else:
        plt.savefig(dir_out+f'/figs/{num}_sims_comparison_{append}_resp_from_sims_n0subtracted_binnedratio_lmaxT{lmaxT}.png',bbox_inches='tight')

    #TODO
    filename = 'reconstruction_lmaxT3500_9ests_mh_withT3.pkl'
    results = {}
    results['ratio_cinv'] = ratio_cinv
    results['errorbars_cinv'] = errorbars_cinv
    with open(filename, 'wb') as f:
        pickle.dump(results, f)


def get_n0(sims,qetype,config,append,cmbonly=False,withT3=False):
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
    if withT3:
        filename = dir_out+f'/n0/n0_{num}simpairs_healqest_{qetype}_withT3_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_resp_from_sims_9ests.pkl'
    else:
        filename = dir_out+f'/n0/n0_{num}simpairs_healqest_{qetype}_noT3_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_resp_from_sims_9ests.pkl'

    if os.path.isfile(filename):
        n0 = pickle.load(open(filename,'rb'))

    elif qetype == 'gmv':
        # GMV response
        resp_gmv = get_sim_response('all',config,cinv=False,append=append_original,sims=np.append(sims,num+1),withT3=withT3)
        #resp_gmv_TTEETE = get_sim_response('TTEETE',config,cinv=False,append=append_original,sims=np.append(sims,num+1),withT3=withT3)
        #resp_gmv_TBEB = get_sim_response('TBEB',config,cinv=False,append=append_original,sims=np.append(sims,num+1),withT3=withT3)
        inv_resp_gmv = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv[1:] = 1./(resp_gmv)[1:]
        #inv_resp_gmv_TTEETE = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TTEETE[1:] = 1./(resp_gmv_TTEETE)[1:]
        #inv_resp_gmv_TBEB = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TBEB[1:] = 1./(resp_gmv_TBEB)[1:]

        n0 = {'total':0, 'TTEETE':0, 'TBEB':0}
        for i, sim1 in enumerate(sims):
            sim2 = sim1 + 1

            if withT3:
                # Get the lensed ij sims
                plm_gmv_ij = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_withT3.npy')
                #plm_gmv_TTEETE_ij = np.load(dir_out+f'/plm_TTEETE_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_withT3.npy')
                #plm_gmv_TBEB_ij = np.load(dir_out+f'/plm_TBEB_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_withT3.npy')

                # Now get the ji sims
                plm_gmv_ji = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_withT3.npy')
                #plm_gmv_TTEETE_ji = np.load(dir_out+f'/plm_TTEETE_healqest_gmv_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_withT3.npy')
                #plm_gmv_TBEB_ji = np.load(dir_out+f'/plm_TBEB_healqest_gmv_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_withT3.npy')
            else:
                # Get the lensed ij sims
                plm_gmv_ij = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_noT3.npy')
                #plm_gmv_TTEETE_ij = np.load(dir_out+f'/plm_TTEETE_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_noT3.npy')
                #plm_gmv_TBEB_ij = np.load(dir_out+f'/plm_TBEB_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_noT3.npy')

                # Now get the ji sims
                plm_gmv_ji = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_noT3.npy')
                #plm_gmv_TTEETE_ji = np.load(dir_out+f'/plm_TTEETE_healqest_gmv_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_noT3.npy')
                #plm_gmv_TBEB_ji = np.load(dir_out+f'/plm_TBEB_healqest_gmv_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_noT3.npy')

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
        ests = ['T1T2', 'T2T1', 'EE', 'TE', 'ET', 'TB', 'BT', 'EB', 'BE']
        resps = np.zeros((len(l),len(ests)), dtype=np.complex_)
        inv_resps = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
        for i, est in enumerate(ests):
            resps[:,i] = get_sim_response(est,config,cinv=True,append=append_original,sims=np.append(sims,num+1),gmv=True,withT3=withT3)
            inv_resps[1:,i] = 1/(resps)[1:,i]
        resp = 0.5*resps[:,0]+0.5*resps[:,1]+np.sum(resps[:,2:], axis=1)
        resp_TTEETE = 0.5*np.sum(resps[:,:2], axis=1)+np.sum(resps[:,2:5], axis=1)
        resp_TBEB = np.sum(resps[:,5:], axis=1)
        inv_resp = np.zeros_like(l,dtype=np.complex_); inv_resp[1:] = 1/(resp)[1:]
        inv_resp_TTEETE = np.zeros_like(l,dtype=np.complex_); inv_resp_TTEETE[1:] = 1/(resp_TTEETE)[1:]
        inv_resp_TBEB = np.zeros_like(l,dtype=np.complex_); inv_resp_TBEB[1:] = 1/(resp_TBEB)[1:]

        n0 = {'total':0, 'T1T2':0, 'T2T1':0, 'EE':0, 'TE':0, 'ET':0, 'TB':0, 'BT':0, 'EB':0, 'BE':0, 'TTEETE':0, 'TBEB':0}
        for i, sim1 in enumerate(sims):
            sim2 = sim1 + 1

            if withT3:
                # Get the lensed ij sims
                plms_ij = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_withT3.npy')),len(ests)), dtype=np.complex_)
                for i, est in enumerate(ests):
                    plms_ij[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_withT3.npy')
                plm_total_ij = 0.5*np.sum(plms_ij[:,:2], axis=1)+np.sum(plms_ij[:,2:], axis=1)

                # Now get the ji sims
                plms_ji = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_gmv_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_withT3.npy')),len(ests)), dtype=np.complex_)
                for i, est in enumerate(ests):
                    plms_ji[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_withT3.npy')
                plm_total_ji = 0.5*np.sum(plms_ji[:,:2], axis=1)+np.sum(plms_ji[:,2:], axis=1)
            else:
                # Get the lensed ij sims
                plms_ij = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_noT3.npy')),len(ests)), dtype=np.complex_)
                for i, est in enumerate(ests):
                    plms_ij[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_noT3.npy')
                plm_total_ij = 0.5*np.sum(plms_ij[:,:2], axis=1)+np.sum(plms_ij[:,2:], axis=1)

                # Now get the ji sims
                plms_ji = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_gmv_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_noT3.npy')),len(ests)), dtype=np.complex_)
                for i, est in enumerate(ests):
                    plms_ji[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_noT3.npy')
                plm_total_ji = 0.5*np.sum(plms_ji[:,:2], axis=1)+np.sum(plms_ji[:,2:], axis=1)

            if np.any(np.isnan(plms_ij[:,0])):
                print(f'Sim {sim1} is bad!')
                num -= 1
                continue

            # NINE estimators!!!
            plm_TTEETE_ij = 0.5*np.sum(plms_ij[:,:2], axis=1)+np.sum(plms_ij[:,2:5], axis=1)
            plm_TTEETE_ji = 0.5*np.sum(plms_ji[:,:2], axis=1)+np.sum(plms_ji[:,2:5], axis=1)
            plm_TBEB_ij = np.sum(plms_ij[:,5:], axis=1)
            plm_TBEB_ji = np.sum(plms_ji[:,5:], axis=1)

            # Response correct
            plm_total_ij = hp.almxfl(plm_total_ij,inv_resp)
            plm_TTEETE_ij = hp.almxfl(plm_TTEETE_ij,inv_resp_TTEETE)
            plm_TBEB_ij = hp.almxfl(plm_TBEB_ij,inv_resp_TBEB)
            for i, est in enumerate(ests):
                plms_ij[:,i] = hp.almxfl(plms_ij[:,i],inv_resps[:,i])

            plm_total_ji = hp.almxfl(plm_total_ji,inv_resp)
            plm_TTEETE_ji = hp.almxfl(plm_TTEETE_ji,inv_resp_TTEETE)
            plm_TBEB_ji = hp.almxfl(plm_TBEB_ji,inv_resp_TBEB)
            for i, est in enumerate(ests):
                plms_ji[:,i] = hp.almxfl(plms_ji[:,i],inv_resps[:,i])

            # Get ij auto spectra <ijij>
            auto = hp.alm2cl(plm_total_ij, plm_total_ij, lmax=lmax)
            auto_TTEETE = hp.alm2cl(plm_TTEETE_ij, plm_TTEETE_ij, lmax=lmax)
            auto_TBEB = hp.alm2cl(plm_TBEB_ij, plm_TBEB_ij, lmax=lmax)
            auto_T1T2 = hp.alm2cl(plms_ij[:,0], plms_ij[:,0], lmax=lmax)
            auto_T2T1 = hp.alm2cl(plms_ij[:,1], plms_ij[:,1], lmax=lmax)
            auto_EE = hp.alm2cl(plms_ij[:,2], plms_ij[:,2], lmax=lmax)
            auto_TE = hp.alm2cl(plms_ij[:,3], plms_ij[:,3], lmax=lmax)
            auto_ET = hp.alm2cl(plms_ij[:,4], plms_ij[:,4], lmax=lmax)
            auto_TB = hp.alm2cl(plms_ij[:,5], plms_ij[:,5], lmax=lmax)
            auto_BT = hp.alm2cl(plms_ij[:,6], plms_ij[:,6], lmax=lmax)
            auto_EB = hp.alm2cl(plms_ij[:,7], plms_ij[:,7], lmax=lmax)
            auto_BE = hp.alm2cl(plms_ij[:,8], plms_ij[:,8], lmax=lmax)

            # Get cross spectra <ijji>
            cross = hp.alm2cl(plm_total_ij, plm_total_ji, lmax=lmax)
            cross_TTEETE = hp.alm2cl(plm_TTEETE_ij, plm_TTEETE_ji, lmax=lmax)
            cross_TBEB = hp.alm2cl(plm_TBEB_ij, plm_TBEB_ji, lmax=lmax)
            cross_T1T2 = hp.alm2cl(plms_ij[:,0], plms_ji[:,0], lmax=lmax)
            cross_T2T1 = hp.alm2cl(plms_ij[:,1], plms_ji[:,1], lmax=lmax)
            cross_EE = hp.alm2cl(plms_ij[:,2], plms_ji[:,2], lmax=lmax)
            cross_TE = hp.alm2cl(plms_ij[:,3], plms_ji[:,3], lmax=lmax)
            cross_ET = hp.alm2cl(plms_ij[:,4], plms_ji[:,4], lmax=lmax)
            cross_TB = hp.alm2cl(plms_ij[:,5], plms_ji[:,5], lmax=lmax)
            cross_BT = hp.alm2cl(plms_ij[:,6], plms_ji[:,6], lmax=lmax)
            cross_EB = hp.alm2cl(plms_ij[:,7], plms_ji[:,7], lmax=lmax)
            cross_BE = hp.alm2cl(plms_ij[:,8], plms_ji[:,8], lmax=lmax)

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

        with open(filename, 'wb') as f:
            pickle.dump(n0, f)

    elif qetype == 'sqe':
        # SQE response
        ests = ['T1T2', 'T2T1', 'EE', 'TE', 'ET', 'TB', 'BT', 'EB', 'BE']
        resps = np.zeros((len(l),len(ests)), dtype=np.complex_)
        inv_resps = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
        for i, est in enumerate(ests):
            resps[:,i] = get_sim_response(est,config,gmv=False,cinv=False,append=append_original,sims=np.append(sims,num+1),withT3=withT3)
            inv_resps[1:,i] = 1/(resps)[1:,i]
        resp = 0.5*resps[:,0]+0.5*resps[:,1]+np.sum(resps[:,2:], axis=1)
        resp_TTEETE = 0.5*np.sum(resps[:,:2], axis=1)+np.sum(resps[:,2:5], axis=1)
        resp_TBEB = np.sum(resps[:,5:], axis=1)
        inv_resp = np.zeros_like(l,dtype=np.complex_); inv_resp[1:] = 1/(resp)[1:]
        inv_resp_TTEETE = np.zeros_like(l,dtype=np.complex_); inv_resp_TTEETE[1:] = 1/(resp_TTEETE)[1:]
        inv_resp_TBEB = np.zeros_like(l,dtype=np.complex_); inv_resp_TBEB[1:] = 1/(resp_TBEB)[1:]

        n0 = {'total':0, 'T1T2':0, 'T2T1':0, 'EE':0, 'TE':0, 'ET':0, 'TB':0, 'BT':0, 'EB':0, 'BE':0, 'TTEETE':0, 'TBEB':0}
        for i, sim1 in enumerate(sims):
            sim2 = sim1 + 1

            if withT3:
                # Get the lensed ij sims
                plms_ij = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_withT3.npy')),len(ests)), dtype=np.complex_)
                for i, est in enumerate(ests):
                    plms_ij[:,i] = np.load(dir_out+f'/plm_{est}_healqest_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_withT3.npy')
                plm_total_ij = 0.5*np.sum(plms_ij[:,:2], axis=1)+np.sum(plms_ij[:,2:], axis=1)

                # Now get the ji sims
                plms_ji = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_withT3.npy')),len(ests)), dtype=np.complex_)
                for i, est in enumerate(ests):
                    plms_ji[:,i] = np.load(dir_out+f'/plm_{est}_healqest_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_withT3.npy')
                plm_total_ji = 0.5*np.sum(plms_ji[:,:2], axis=1)+np.sum(plms_ji[:,2:], axis=1)
            else:
                # Get the lensed ij sims
                plms_ij = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_noT3.npy')),len(ests)), dtype=np.complex_)
                for i, est in enumerate(ests):
                    plms_ij[:,i] = np.load(dir_out+f'/plm_{est}_healqest_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_noT3.npy')
                plm_total_ij = 0.5*np.sum(plms_ij[:,:2], axis=1)+np.sum(plms_ij[:,2:], axis=1)

                # Now get the ji sims
                plms_ji = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_noT3.npy')),len(ests)), dtype=np.complex_)
                for i, est in enumerate(ests):
                    plms_ji[:,i] = np.load(dir_out+f'/plm_{est}_healqest_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_noT3.npy')
                plm_total_ji = 0.5*np.sum(plms_ji[:,:2], axis=1)+np.sum(plms_ji[:,2:], axis=1)

            if np.any(np.isnan(plms_ij[:,0])):
                print(f'Sim {sim1} is bad!')
                num -= 1
                continue

            # NINE estimators!!!
            plm_TTEETE_ij = 0.5*np.sum(plms_ij[:,:2], axis=1)+np.sum(plms_ij[:,2:5], axis=1)
            plm_TTEETE_ji = 0.5*np.sum(plms_ji[:,:2], axis=1)+np.sum(plms_ji[:,2:5], axis=1)
            plm_TBEB_ij = np.sum(plms_ij[:,5:], axis=1)
            plm_TBEB_ji = np.sum(plms_ji[:,5:], axis=1)

            # Response correct
            plm_total_ij = hp.almxfl(plm_total_ij,inv_resp)
            plm_TTEETE_ij = hp.almxfl(plm_TTEETE_ij,inv_resp_TTEETE)
            plm_TBEB_ij = hp.almxfl(plm_TBEB_ij,inv_resp_TBEB)
            for i, est in enumerate(ests):
                plms_ij[:,i] = hp.almxfl(plms_ij[:,i],inv_resps[:,i])

            plm_total_ji = hp.almxfl(plm_total_ji,inv_resp)
            plm_TTEETE_ji = hp.almxfl(plm_TTEETE_ji,inv_resp_TTEETE)
            plm_TBEB_ji = hp.almxfl(plm_TBEB_ji,inv_resp_TBEB)
            for i, est in enumerate(ests):
                plms_ji[:,i] = hp.almxfl(plms_ji[:,i],inv_resps[:,i])

            # Get ij auto spectra <ijij>
            auto = hp.alm2cl(plm_total_ij, plm_total_ij, lmax=lmax)
            auto_TTEETE = hp.alm2cl(plm_TTEETE_ij, plm_TTEETE_ij, lmax=lmax)
            auto_TBEB = hp.alm2cl(plm_TBEB_ij, plm_TBEB_ij, lmax=lmax)
            auto_T1T2 = hp.alm2cl(plms_ij[:,0], plms_ij[:,0], lmax=lmax)
            auto_T2T1 = hp.alm2cl(plms_ij[:,1], plms_ij[:,1], lmax=lmax)
            auto_EE = hp.alm2cl(plms_ij[:,2], plms_ij[:,2], lmax=lmax)
            auto_TE = hp.alm2cl(plms_ij[:,3], plms_ij[:,3], lmax=lmax)
            auto_ET = hp.alm2cl(plms_ij[:,4], plms_ij[:,4], lmax=lmax)
            auto_TB = hp.alm2cl(plms_ij[:,5], plms_ij[:,5], lmax=lmax)
            auto_BT = hp.alm2cl(plms_ij[:,6], plms_ij[:,6], lmax=lmax)
            auto_EB = hp.alm2cl(plms_ij[:,7], plms_ij[:,7], lmax=lmax)
            auto_BE = hp.alm2cl(plms_ij[:,8], plms_ij[:,8], lmax=lmax)

            # Get cross spectra <ijji>
            cross = hp.alm2cl(plm_total_ij, plm_total_ji, lmax=lmax)
            cross_TTEETE = hp.alm2cl(plm_TTEETE_ij, plm_TTEETE_ji, lmax=lmax)
            cross_TBEB = hp.alm2cl(plm_TBEB_ij, plm_TBEB_ji, lmax=lmax)
            cross_T1T2 = hp.alm2cl(plms_ij[:,0], plms_ji[:,0], lmax=lmax)
            cross_T2T1 = hp.alm2cl(plms_ij[:,1], plms_ji[:,1], lmax=lmax)
            cross_EE = hp.alm2cl(plms_ij[:,2], plms_ji[:,2], lmax=lmax)
            cross_TE = hp.alm2cl(plms_ij[:,3], plms_ji[:,3], lmax=lmax)
            cross_ET = hp.alm2cl(plms_ij[:,4], plms_ji[:,4], lmax=lmax)
            cross_TB = hp.alm2cl(plms_ij[:,5], plms_ji[:,5], lmax=lmax)
            cross_BT = hp.alm2cl(plms_ij[:,6], plms_ji[:,6], lmax=lmax)
            cross_EB = hp.alm2cl(plms_ij[:,7], plms_ji[:,7], lmax=lmax)
            cross_BE = hp.alm2cl(plms_ij[:,8], plms_ji[:,8], lmax=lmax)

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

        with open(filename, 'wb') as f:
            pickle.dump(n0, f)

    else:
        print('Invalid argument qetype.')

    return n0

def get_n1(sims,qetype,config,append,withT3=False):
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
    append_original = append
    if withT3:
        filename = dir_out+f'/n1/n1_{num}simpairs_healqest_{qetype}_withT3_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_resp_from_sims_9ests.pkl'
    else:
        filename = dir_out+f'/n1/n1_{num}simpairs_healqest_{qetype}_noT3_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_resp_from_sims_9ests.pkl'

    if os.path.isfile(filename):
        n1 = pickle.load(open(filename,'rb'))

    elif qetype == 'gmv':
        # GMV response
        resp_gmv = get_sim_response('all',config,cinv=False,append=append_original,sims=np.append(sims,num+1),withT3=withT3)
        #resp_gmv_TTEETE = get_sim_response('TTEETE',config,cinv=False,append=append_original,sims=np.append(sims,num+1),withT3=withT3)
        #resp_gmv_TBEB = get_sim_response('TBEB',config,cinv=False,append=append_original,sims=np.append(sims,num+1),withT3=withT3)
        inv_resp_gmv = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv[1:] = 1./(resp_gmv)[1:]
        #inv_resp_gmv_TTEETE = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TTEETE[1:] = 1./(resp_gmv_TTEETE)[1:]
        #inv_resp_gmv_TBEB = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TBEB[1:] = 1./(resp_gmv_TBEB)[1:]

        n1 = {'total':0, 'TTEETE':0, 'TBEB':0}
        for i, sim in enumerate(sims):
            if withT3:
                # These are reconstructions using sims that were lensed with the same phi but different CMB realizations, no foregrounds
                # Get the lensed ij sims
                plm_gmv_ij = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu1tqu2_withT3.npy')
                #plm_gmv_TTEETE_ij = np.load(dir_out+f'/plm_TTEETE_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu1tqu2_withT3.npy')
                #plm_gmv_TBEB_ij = np.load(dir_out+f'/plm_TBEB_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu1tqu2_withT3.npy')

                # Now get the ji sims
                plm_gmv_ji = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu2tqu1_withT3.npy')
                #plm_gmv_TTEETE_ji = np.load(dir_out+f'/plm_TTEETE_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu2tqu1_withT3.npy')
                #plm_gmv_TBEB_ji = np.load(dir_out+f'/plm_TBEB_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu2tqu1_withT3.npy')
            else:
                # These are reconstructions using sims that were lensed with the same phi but different CMB realizations, no foregrounds
                # Get the lensed ij sims
                plm_gmv_ij = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu1tqu2_noT3.npy')
                #plm_gmv_TTEETE_ij = np.load(dir_out+f'/plm_TTEETE_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu1tqu2_noT3.npy')
                #plm_gmv_TBEB_ij = np.load(dir_out+f'/plm_TBEB_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu1tqu2_noT3.npy')

                # Now get the ji sims
                plm_gmv_ji = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu2tqu1_noT3.npy')
                #plm_gmv_TTEETE_ji = np.load(dir_out+f'/plm_TTEETE_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu2tqu1_noT3.npy')
                #plm_gmv_TBEB_ji = np.load(dir_out+f'/plm_TBEB_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu2tqu1_noT3.npy')

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

        n0 = get_n0(sims=sims,qetype=qetype,config=config,
                    append=append,cmbonly=True,withT3=withT3)

        n1['total'] -= n0['total']
        #n1['TTEETE'] -= n0['TTEETE']
        #n1['TBEB'] -= n0['TBEB']

        with open(filename, 'wb') as f:
            pickle.dump(n1, f)

    elif qetype == 'gmv_cinv':
        ests = ['T1T2', 'T2T1', 'EE', 'TE', 'ET', 'TB', 'BT', 'EB', 'BE']
        resps = np.zeros((len(l),len(ests)), dtype=np.complex_)
        inv_resps = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
        for i, est in enumerate(ests):
            resps[:,i] = get_sim_response(est,config,cinv=True,append=append_original,sims=np.append(sims,num+1),gmv=True,withT3=withT3)
            inv_resps[1:,i] = 1/(resps)[1:,i]
        resp = 0.5*resps[:,0]+0.5*resps[:,1]+np.sum(resps[:,2:], axis=1)
        resp_TTEETE = 0.5*np.sum(resps[:,:2], axis=1)+np.sum(resps[:,2:5], axis=1)
        resp_TBEB = np.sum(resps[:,5:], axis=1)
        inv_resp = np.zeros_like(l,dtype=np.complex_); inv_resp[1:] = 1/(resp)[1:]
        inv_resp_TTEETE = np.zeros_like(l,dtype=np.complex_); inv_resp_TTEETE[1:] = 1/(resp_TTEETE)[1:]
        inv_resp_TBEB = np.zeros_like(l,dtype=np.complex_); inv_resp_TBEB[1:] = 1/(resp_TBEB)[1:]

        n1 = {'total':0, 'T1T2':0, 'T2T1':0, 'EE':0, 'TE':0, 'ET':0, 'TB':0, 'BT':0, 'EB':0, 'BE':0, 'TTEETE':0, 'TBEB':0}
        for i, sim in enumerate(sims):
            if not withT3:
                # Get the lensed ij sims
                plms_ij = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu1tqu2_cinv_noT3.npy')),len(ests)),dtype=np.complex_)
                for i, est in enumerate(ests):
                    plms_ij[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu1tqu2_cinv_noT3.npy')
                plm_total_ij = 0.5*np.sum(plms_ij[:,:2], axis=1)+np.sum(plms_ij[:,2:], axis=1)

                # Now get the ji sims
                plms_ji = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu2tqu1_cinv_noT3.npy')),len(ests)),dtype=np.complex_)
                for i, est in enumerate(ests):
                    plms_ji[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu2tqu1_cinv_noT3.npy')
                plm_total_ji = 0.5*np.sum(plms_ji[:,:2], axis=1)+np.sum(plms_ji[:,2:], axis=1)
            else:
                # Get the lensed ij sims
                plms_ij = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu1tqu2_cinv_withT3.npy')),len(ests)),dtype=np.complex_)
                for i, est in enumerate(ests):
                    plms_ij[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu1tqu2_cinv_withT3.npy')
                plm_total_ij = 0.5*np.sum(plms_ij[:,:2], axis=1)+np.sum(plms_ij[:,2:], axis=1)

                # Now get the ji sims
                plms_ji = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu2tqu1_cinv_withT3.npy')),len(ests)),dtype=np.complex_)
                for i, est in enumerate(ests):
                    plms_ji[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu2tqu1_cinv_withT3.npy')
                plm_total_ji = 0.5*np.sum(plms_ji[:,:2], axis=1)+np.sum(plms_ji[:,2:], axis=1)

            # NINE estimators!!!
            plm_TTEETE_ij = 0.5*np.sum(plms_ij[:,:2], axis=1)+np.sum(plms_ij[:,2:5], axis=1)
            plm_TTEETE_ji = 0.5*np.sum(plms_ji[:,:2], axis=1)+np.sum(plms_ji[:,2:5], axis=1)
            plm_TBEB_ij = np.sum(plms_ij[:,5:], axis=1)
            plm_TBEB_ji = np.sum(plms_ji[:,5:], axis=1)

            # Response correct
            plm_total_ij = hp.almxfl(plm_total_ij,inv_resp)
            plm_TTEETE_ij = hp.almxfl(plm_TTEETE_ij,inv_resp_TTEETE)
            plm_TBEB_ij = hp.almxfl(plm_TBEB_ij,inv_resp_TBEB)
            for i, est in enumerate(ests):
                plms_ij[:,i] = hp.almxfl(plms_ij[:,i],inv_resps[:,i])

            plm_total_ji = hp.almxfl(plm_total_ji,inv_resp)
            plm_TTEETE_ji = hp.almxfl(plm_TTEETE_ji,inv_resp_TTEETE)
            plm_TBEB_ji = hp.almxfl(plm_TBEB_ji,inv_resp_TBEB)
            for i, est in enumerate(ests):
                plms_ji[:,i] = hp.almxfl(plms_ji[:,i],inv_resps[:,i])

            # Get ij auto spectra <ijij>
            auto = hp.alm2cl(plm_total_ij, plm_total_ij, lmax=lmax)
            auto_TTEETE = hp.alm2cl(plm_TTEETE_ij, plm_TTEETE_ij, lmax=lmax)
            auto_TBEB = hp.alm2cl(plm_TBEB_ij, plm_TBEB_ij, lmax=lmax)
            auto_T1T2 = hp.alm2cl(plms_ij[:,0], plms_ij[:,0], lmax=lmax)
            auto_T2T1 = hp.alm2cl(plms_ij[:,1], plms_ij[:,1], lmax=lmax)
            auto_EE = hp.alm2cl(plms_ij[:,2], plms_ij[:,2], lmax=lmax)
            auto_TE = hp.alm2cl(plms_ij[:,3], plms_ij[:,3], lmax=lmax)
            auto_ET = hp.alm2cl(plms_ij[:,4], plms_ij[:,4], lmax=lmax)
            auto_TB = hp.alm2cl(plms_ij[:,5], plms_ij[:,5], lmax=lmax)
            auto_BT = hp.alm2cl(plms_ij[:,6], plms_ij[:,6], lmax=lmax)
            auto_EB = hp.alm2cl(plms_ij[:,7], plms_ij[:,7], lmax=lmax)
            auto_BE = hp.alm2cl(plms_ij[:,8], plms_ij[:,8], lmax=lmax)

            # Get cross spectra <ijji>
            cross = hp.alm2cl(plm_total_ij, plm_total_ji, lmax=lmax)
            cross_TTEETE = hp.alm2cl(plm_TTEETE_ij, plm_TTEETE_ji, lmax=lmax)
            cross_TBEB = hp.alm2cl(plm_TBEB_ij, plm_TBEB_ji, lmax=lmax)
            cross_T1T2 = hp.alm2cl(plms_ij[:,0], plms_ji[:,0], lmax=lmax)
            cross_T2T1 = hp.alm2cl(plms_ij[:,1], plms_ji[:,1], lmax=lmax)
            cross_EE = hp.alm2cl(plms_ij[:,2], plms_ji[:,2], lmax=lmax)
            cross_TE = hp.alm2cl(plms_ij[:,3], plms_ji[:,3], lmax=lmax)
            cross_ET = hp.alm2cl(plms_ij[:,4], plms_ji[:,4], lmax=lmax)
            cross_TB = hp.alm2cl(plms_ij[:,5], plms_ji[:,5], lmax=lmax)
            cross_BT = hp.alm2cl(plms_ij[:,6], plms_ji[:,6], lmax=lmax)
            cross_EB = hp.alm2cl(plms_ij[:,7], plms_ji[:,7], lmax=lmax)
            cross_BE = hp.alm2cl(plms_ij[:,8], plms_ji[:,8], lmax=lmax)

            n1['total'] += auto + cross
            n1['TTEETE'] += auto_TTEETE + cross_TTEETE
            n1['TBEB'] += auto_TBEB + cross_TBEB
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
        n1['TTEETE'] *= 1/num
        n1['TBEB'] *= 1/num
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
                    append=append,cmbonly=True,withT3=withT3)

        n1['total'] -= n0['total']
        n1['TTEETE'] -= n0['TTEETE']
        n1['TBEB'] -= n0['TBEB']
        n1['T1T2'] -= n0['T1T2']
        n1['T2T1'] -= n0['T2T1']
        n1['EE'] -= n0['EE']
        n1['TE'] -= n0['TE']
        n1['ET'] -= n0['ET']
        n1['TB'] -= n0['TB']
        n1['BT'] -= n0['BT']
        n1['EB'] -= n0['EB']
        n1['BE'] -= n0['BE']

        with open(filename, 'wb') as f:
            pickle.dump(n1, f)

    elif qetype == 'sqe':
        # SQE response
        ests = ['T1T2', 'T2T1', 'EE', 'TE', 'ET', 'TB', 'BT', 'EB', 'BE']
        resps = np.zeros((len(l),len(ests)), dtype=np.complex_)
        inv_resps = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
        for i, est in enumerate(ests):
            resps[:,i] = get_sim_response(est,config,gmv=False,cinv=False,append=append_original,sims=np.append(sims,num+1),withT3=withT3)
            inv_resps[1:,i] = 1/(resps)[1:,i]
        resp = 0.5*resps[:,0]+0.5*resps[:,1]+np.sum(resps[:,2:], axis=1)
        resp_TTEETE = 0.5*np.sum(resps[:,:2], axis=1)+np.sum(resps[:,2:5], axis=1)
        resp_TBEB = np.sum(resps[:,5:], axis=1)
        inv_resp = np.zeros_like(l,dtype=np.complex_); inv_resp[1:] = 1/(resp)[1:]
        inv_resp_TTEETE = np.zeros_like(l,dtype=np.complex_); inv_resp_TTEETE[1:] = 1/(resp_TTEETE)[1:]
        inv_resp_TBEB = np.zeros_like(l,dtype=np.complex_); inv_resp_TBEB[1:] = 1/(resp_TBEB)[1:]

        n1 = {'total':0, 'T1T2':0, 'T2T1':0, 'EE':0, 'TE':0, 'ET':0, 'TB':0, 'BT':0, 'EB':0, 'BE':0, 'TTEETE':0, 'TBEB':0}
        for i, sim in enumerate(sims):
            if not withT3:
                # Get the lensed ij sims
                plms_ij = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu1tqu2_noT3.npy')),len(ests)),dtype=np.complex_)
                for i, est in enumerate(ests):
                    plms_ij[:,i] = np.load(dir_out+f'/plm_{est}_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu1tqu2_noT3.npy')
                plm_total_ij = 0.5*np.sum(plms_ij[:,:2], axis=1)+np.sum(plms_ij[:,2:], axis=1)

                # Now get the ji sims
                plms_ji = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu2tqu1_noT3.npy')),len(ests)),dtype=np.complex_)
                for i, est in enumerate(ests):
                    plms_ji[:,i] = np.load(dir_out+f'/plm_{est}_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu2tqu1_noT3.npy')
                plm_total_ji = 0.5*np.sum(plms_ji[:,:2], axis=1)+np.sum(plms_ji[:,2:], axis=1)
            else:
                # Get the lensed ij sims
                plms_ij = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu1tqu2_withT3.npy')),len(ests)),dtype=np.complex_)
                for i, est in enumerate(ests):
                    plms_ij[:,i] = np.load(dir_out+f'/plm_{est}_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu1tqu2_withT3.npy')
                plm_total_ij = 0.5*np.sum(plms_ij[:,:2], axis=1)+np.sum(plms_ij[:,2:], axis=1)

                # Now get the ji sims
                plms_ji = np.zeros((len(np.load(dir_out+f'/plm_EE_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu2tqu1_withT3.npy')),len(ests)),dtype=np.complex_)
                for i, est in enumerate(ests):
                    plms_ji[:,i] = np.load(dir_out+f'/plm_{est}_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cmbonly_phi1_tqu2tqu1_withT3.npy')
                plm_total_ji = 0.5*np.sum(plms_ji[:,:2], axis=1)+np.sum(plms_ji[:,2:], axis=1)

            # NINE estimators!!!
            plm_TTEETE_ij = 0.5*np.sum(plms_ij[:,:2], axis=1)+np.sum(plms_ij[:,2:5], axis=1)
            plm_TTEETE_ji = 0.5*np.sum(plms_ji[:,:2], axis=1)+np.sum(plms_ji[:,2:5], axis=1)
            plm_TBEB_ij = np.sum(plms_ij[:,5:], axis=1)
            plm_TBEB_ji = np.sum(plms_ji[:,5:], axis=1)

            # Response correct
            plm_total_ij = hp.almxfl(plm_total_ij,inv_resp)
            plm_TTEETE_ij = hp.almxfl(plm_TTEETE_ij,inv_resp_TTEETE)
            plm_TBEB_ij = hp.almxfl(plm_TBEB_ij,inv_resp_TBEB)
            for i, est in enumerate(ests):
                plms_ij[:,i] = hp.almxfl(plms_ij[:,i],inv_resps[:,i])

            plm_total_ji = hp.almxfl(plm_total_ji,inv_resp)
            plm_TTEETE_ji = hp.almxfl(plm_TTEETE_ji,inv_resp_TTEETE)
            plm_TBEB_ji = hp.almxfl(plm_TBEB_ji,inv_resp_TBEB)
            for i, est in enumerate(ests):
                plms_ji[:,i] = hp.almxfl(plms_ji[:,i],inv_resps[:,i])

            # Get ij auto spectra <ijij>
            auto = hp.alm2cl(plm_total_ij, plm_total_ij, lmax=lmax)
            auto_TTEETE = hp.alm2cl(plm_TTEETE_ij, plm_TTEETE_ij, lmax=lmax)
            auto_TBEB = hp.alm2cl(plm_TBEB_ij, plm_TBEB_ij, lmax=lmax)
            auto_T1T2 = hp.alm2cl(plms_ij[:,0], plms_ij[:,0], lmax=lmax)
            auto_T2T1 = hp.alm2cl(plms_ij[:,1], plms_ij[:,1], lmax=lmax)
            auto_EE = hp.alm2cl(plms_ij[:,2], plms_ij[:,2], lmax=lmax)
            auto_TE = hp.alm2cl(plms_ij[:,3], plms_ij[:,3], lmax=lmax)
            auto_ET = hp.alm2cl(plms_ij[:,4], plms_ij[:,4], lmax=lmax)
            auto_TB = hp.alm2cl(plms_ij[:,5], plms_ij[:,5], lmax=lmax)
            auto_BT = hp.alm2cl(plms_ij[:,6], plms_ij[:,6], lmax=lmax)
            auto_EB = hp.alm2cl(plms_ij[:,7], plms_ij[:,7], lmax=lmax)
            auto_BE = hp.alm2cl(plms_ij[:,8], plms_ij[:,8], lmax=lmax)

            # Get cross spectra <ijji>
            cross = hp.alm2cl(plm_total_ij, plm_total_ji, lmax=lmax)
            cross_TTEETE = hp.alm2cl(plm_TTEETE_ij, plm_TTEETE_ji, lmax=lmax)
            cross_TBEB = hp.alm2cl(plm_TBEB_ij, plm_TBEB_ji, lmax=lmax)
            cross_T1T2 = hp.alm2cl(plms_ij[:,0], plms_ji[:,0], lmax=lmax)
            cross_T2T1 = hp.alm2cl(plms_ij[:,1], plms_ji[:,1], lmax=lmax)
            cross_EE = hp.alm2cl(plms_ij[:,2], plms_ji[:,2], lmax=lmax)
            cross_TE = hp.alm2cl(plms_ij[:,3], plms_ji[:,3], lmax=lmax)
            cross_ET = hp.alm2cl(plms_ij[:,4], plms_ji[:,4], lmax=lmax)
            cross_TB = hp.alm2cl(plms_ij[:,5], plms_ji[:,5], lmax=lmax)
            cross_BT = hp.alm2cl(plms_ij[:,6], plms_ji[:,6], lmax=lmax)
            cross_EB = hp.alm2cl(plms_ij[:,7], plms_ji[:,7], lmax=lmax)
            cross_BE = hp.alm2cl(plms_ij[:,8], plms_ji[:,8], lmax=lmax)

            n1['total'] += auto + cross
            n1['TTEETE'] += auto_TTEETE + cross_TTEETE
            n1['TBEB'] += auto_TBEB + cross_TBEB
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
        n1['TTEETE'] *= 1/num
        n1['TBEB'] *= 1/num
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
                    append=append,cmbonly=True,withT3=withT3)

        n1['total'] -= n0['total']
        n1['TTEETE'] -= n0['TTEETE']
        n1['TBEB'] -= n0['TBEB']
        n1['T1T2'] -= n0['T1T2']
        n1['T2T1'] -= n0['T2T1']
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

def get_sim_response(est,config,cinv,append,sims,filename=None,gmv=True,withT3=False):
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
        fn += f'_lmaxT{lmaxT}_lmaxP{lmaxP}_lmin{lmin}_cltype{cltype}_{append}'
        if withT3:
            fn += '_withT3'
        else:
            fn += '_noT3'
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
                if withT3:
                    plm = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_withT3.npy')
                else:
                    plm = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_noT3.npy')
            elif gmv:
                if withT3:
                    plm = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_withT3.npy')
                else:
                    plm = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_noT3.npy')
            else:
                if withT3:
                    plm = np.load(dir_out+f'/plm_{est}_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_withT3.npy')
                else:
                    plm = np.load(dir_out+f'/plm_{est}_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_noT3.npy')
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

analyze(append='mh',config_file='test_yuka_lmaxT3500.yaml',withT3=True)
#analyze(append='mh',config_file='test_yuka_lmaxT4000.yaml')
#analyze(append='crossilc_twoseds',config_file='test_yuka_lmaxT3500.yaml',withT3=True)
#analyze(append='crossilc_twoseds',config_file='test_yuka_lmaxT4000.yaml')
