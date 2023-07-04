#!/usr/bin/env python3
import numpy as np
import healpy as hp
import camb
import os, sys
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
from astropy.io import fits
import utils
import matplotlib.pyplot as plt
import weights_combined_qestobj
import qest_combined_qestobj
import wignerd
import resp

def compare_gmv_specs(sim=100,lmax=4096,nside=8192,fluxlim=0.200,dir_out='/scratch/users/yukanaka/gmv/',
                      u=np.ones(4097,dtype=np.complex_),save_fig=True,config_file='profhrd_yuka.yaml'):
    '''
    Temporary, to test weight change in GMV TE.
    No hardening.
    '''
    config = utils.parse_yaml(config_file)
    clfile = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat'
    l = np.arange(0,lmax+1)

    # Load GMV plms
    append = f'tsrc_fluxlim{fluxlim:.3f}'
    plm_gmv = np.load(dir_out+f'/plm_all_healqest_gmv_seed{sim}_lmax{lmax}_nside{nside}_{append}.npy')
    plm_gmv_A = np.load(dir_out+f'/plm_TTEETE_healqest_gmv_seed{sim}_lmax{lmax}_nside{nside}_{append}.npy')
    plm_gmv_TEl1l2flip = np.load(dir_out+f'/plm_all_healqest_gmv_seed{sim}_lmax{lmax}_nside{nside}_{append}_TEl1l2flip.npy')
    plm_gmv_A_TEl1l2flip = np.load(dir_out+f'/plm_TTEETE_healqest_gmv_seed{sim}_lmax{lmax}_nside{nside}_{append}_TEl1l2flip.npy')
    gmv_resp_data = np.genfromtxt('gmv_resp/True_variance_individual_custom_lmin2.0_lmaxT4096_lmaxP4096_beam0_noise0_50.txt')
    inv_resp_gmv = gmv_resp_data[:,3] / l**2
    inv_resp_gmv_A = gmv_resp_data[:,1] / l**2

    print(plm_gmv)
    print(plm_gmv_TEl1l2flip)
    print('...')
    print(plm_gmv_A)
    print(plm_gmv_A_TEl1l2flip)

    # Response correct GMV
    # N is 1/R
    plm_gmv_resp_corr = hp.almxfl(plm_gmv,inv_resp_gmv)
    plm_gmv_A_resp_corr = hp.almxfl(plm_gmv_A,inv_resp_gmv_A)
    plm_gmv_resp_corr_TEl1l2flip = hp.almxfl(plm_gmv_TEl1l2flip,inv_resp_gmv)
    plm_gmv_A_resp_corr_TEl1l2flip = hp.almxfl(plm_gmv_A_TEl1l2flip,inv_resp_gmv_A)

    # Get GMV spectra
    auto_gmv = hp.alm2cl(plm_gmv_resp_corr, plm_gmv_resp_corr, lmax=lmax) * (l*(l+1))**2/4
    auto_gmv_A = hp.alm2cl(plm_gmv_A_resp_corr, plm_gmv_A_resp_corr, lmax=lmax) * (l*(l+1))**2/4
    auto_gmv_TEl1l2flip = hp.alm2cl(plm_gmv_resp_corr_TEl1l2flip, plm_gmv_resp_corr_TEl1l2flip, lmax=lmax) * (l*(l+1))**2/4
    auto_gmv_A_TEl1l2flip = hp.alm2cl(plm_gmv_A_resp_corr_TEl1l2flip, plm_gmv_A_resp_corr_TEl1l2flip, lmax=lmax) * (l*(l+1))**2/4

    # Theory spectrum
    clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
    ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
    clkk = slpp * (l*(l+1))**2/4

    # Plot
    plt.figure(0)
    plt.clf()
    plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')
    plt.plot(l, auto_gmv, color='darkblue', linestyle='-', label="Auto Spectrum (GMV)")
    plt.plot(l, auto_gmv_A, color='dodgerblue', linestyle='-', label=f'Auto Spectrum (GMV [TT, EE, TE])')
    plt.plot(l, auto_gmv_TEl1l2flip, color='cornflowerblue', linestyle='--', label="Auto Spectrum (GMV) with l1/l2 flipped in second half of TE weights")
    plt.plot(l, auto_gmv_A_TEl1l2flip, color='deepskyblue', linestyle='--', label=f'Auto Spectrum (GMV [TT, EE, TE]) with l1/l2 flipped in second half of TE weights')
    plt.plot(l, inv_resp_gmv * (l*(l+1))**2/4, color='firebrick', linestyle='-', label='1/R (GMV)')
    plt.plot(l, inv_resp_gmv_A * (l*(l+1))**2/4, color='peru', linestyle='-', label='1/R (GMV [TT, EE, TE])')
    plt.plot(l, inv_resp_gmv_TEl1l2flip * (l*(l+1))**2/4, color='lightcoral', linestyle='--', label='1/R (GMV) with l1/l2 flipped in second half of TE weights')
    plt.plot(l, inv_resp_gmv_A_TEl1l2flip * (l*(l+1))**2/4, color='sandybrown', linestyle='--', label='1/R (GMV [TT, EE, TE]) with l1/l2 flipped in second half of TE weights')
    plt.ylabel("$C_\ell^{\kappa\kappa}$")
    plt.xlabel('$\ell$')
    plt.title(f'Spectra for Sim {sim}, with Point Sources in T (GMV)')
    plt.legend(loc='upper right', fontsize='small')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,lmax)
    plt.ylim(5e-9,1e-6)
    if save_fig:
        plt.savefig(dir_out+f'/figs/temp_weight_comparison_spec_gmv_fluxlim{fluxlim:.3f}.png')

def compare_profile_hardening(sim=100,lmax=4096,nside=8192,fluxlim=0.200,dir_out='/scratch/users/yukanaka/gmv/',
                              u=np.ones(4097,dtype=np.complex_),save_fig=True,config_file='profhrd_yuka.yaml'):
    '''
    Argument fluxlim = 0.010 or 0.200.
    '''
    config = utils.parse_yaml(config_file)
    clfile = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat'
    l = np.arange(0,lmax+1)
    ests = ['TT', 'EE', 'TE', 'TB', 'EB']

    # Load GMV plms
    append = f'tsrc_fluxlim{fluxlim:.3f}'
    plm_gmv = np.load(dir_out+f'/plm_all_healqest_gmv_seed{sim}_lmax{lmax}_nside{nside}_{append}.npy')
    plm_gmv_A = np.load(dir_out+f'/plm_TTEETE_healqest_gmv_seed{sim}_lmax{lmax}_nside{nside}_{append}.npy')
    plm_gmv_B = np.load(dir_out+f'/plm_TBEB_healqest_gmv_seed{sim}_lmax{lmax}_nside{nside}_{append}.npy')
    gmv_resp_data = np.genfromtxt('gmv_resp/True_variance_individual_custom_lmin2.0_lmaxT4096_lmaxP4096_beam0_noise0_50.txt')
    inv_resp_gmv = gmv_resp_data[:,3] / l**2
    inv_resp_gmv_A = gmv_resp_data[:,1] / l**2
    inv_resp_gmv_B = gmv_resp_data[:,2] / l**2

    # Harden!
    glm_prf_A = np.load(dir_out+f'/plm_TTEETEprf_healqest_gmv_seed{sim}_lmax{lmax}_nside{nside}_{append}.npy')
    gmv_resp_data_ss = np.genfromtxt('gmv_resp/True_variance_individual_custom_lmin2.0_lmaxT4096_lmaxP4096_beam0_noise0_50_PRF_SS.txt')
    gmv_resp_data_sk = np.genfromtxt('gmv_resp/True_variance_individual_custom_lmin2.0_lmaxT4096_lmaxP4096_beam0_noise0_50_PRF_SK.txt')
    #TODO: for some reason we need a -1 factor for SK
    gmv_resp_data_sk *= -1
    inv_resp_gmv_A_ss = gmv_resp_data_ss[:,1] / l**2
    inv_resp_gmv_A_sk = gmv_resp_data_sk[:,1] / l**2
    # Abhi's code calculates the reconstruction noise for d field rather than phi field, see GMV_QE.py, line 292 for example
    resp_gmv = 1/inv_resp_gmv
    resp_gmv_A = 1/inv_resp_gmv_A
    resp_gmv_B = 1/inv_resp_gmv_B
    resp_gmv_A_sk = 1/inv_resp_gmv_A_sk
    resp_gmv_A_ss = 1/inv_resp_gmv_A_ss
    weight_gmv = -1 * resp_gmv_A_sk / resp_gmv_A_ss
    plm_gmv_A_hrd = plm_gmv_A + hp.almxfl(glm_prf_A, weight_gmv)
    plm_gmv_hrd = plm_gmv_A_hrd + plm_gmv_B
    #plm_gmv_hrd = plm_gmv + hp.almxfl(glm_prf_A, weight_gmv)
    resp_gmv_A_hrd = resp_gmv_A + weight_gmv*resp_gmv_A_sk
    resp_gmv_hrd = resp_gmv_A_hrd + resp_gmv_B
    #resp_gmv_hrd   = resp_gmv + weight_gmv*resp_gmv_A_sk
    inv_resp_gmv_A_hrd = np.zeros_like(l, dtype=np.complex_); inv_resp_gmv_A_hrd[1:] = 1/(resp_gmv_A_hrd)[1:]
    inv_resp_gmv_hrd = np.zeros_like(l, dtype=np.complex_); inv_resp_gmv_hrd[1:] = 1/(resp_gmv_hrd)[1:]

    # Load healqest plms
    plms_original = np.zeros((len(np.load(dir_out+f'/plm_TT_healqest_seed{sim}_lmax{lmax}_nside{nside}_{append}.npy')),5), dtype=np.complex_)
    resps_original = np.zeros((len(l),5), dtype=np.complex_)
    inv_resps_original = np.zeros((len(l),5) ,dtype=np.complex_)
    for i, est in enumerate(ests):
        plms_original[:,i] = np.load(dir_out+f'/plm_{est}_healqest_seed{sim}_lmax{lmax}_nside{nside}_{append}.npy')
        resps_original[:,i] = get_analytic_response(est,config,lmax,fwhm=0,nlev_t=0,nlev_p=0,u=u,clfile=clfile)
        inv_resps_original[1:,i] = 1/(resps_original)[1:,i]
    plm_original = np.sum(plms_original, axis=1)
    resp_original = np.sum(resps_original, axis=1)
    inv_resp_original = np.zeros_like(l,dtype=np.complex_); inv_resp_original[1:] = 1/(resp_original)[1:]

    # Harden!
    glm_prf_TTprf = np.load(dir_out+f'/plm_TTprf_healqest_seed{sim}_lmax{lmax}_nside{nside}_{append}.npy')
    resp_original_ss = get_analytic_response('TTprf',config,lmax,fwhm=0,nlev_t=0,nlev_p=0,u=u,clfile=clfile,
                                              filename='/scratch/users/yukanaka/gmv/resp/an_resp_SS_TTprf_healqest_lmax{}_fwhm0_nlevt0_nlevp0.npy'.format(lmax))
    resp_original_sk = get_analytic_response('TTprf',config,lmax,fwhm=0,nlev_t=0,nlev_p=0,u=u,clfile=clfile,
                                             filename='/scratch/users/yukanaka/gmv/resp/an_resp_SK_TTprf_TT_healqest_lmax{}_fwhm0_nlevt0_nlevp0.npy'.format(lmax),
                                             qeZA=weights_combined_qestobj.weights('TT',lmax,config,cltype='len'))
    weight_original = -1 * resp_original_sk / resp_original_ss
    plm_original_TT_hrd = plms_original[:,0] + hp.almxfl(glm_prf_TTprf, weight_original)
    plm_original_hrd = plm_original_TT_hrd + np.sum(plms_original[:,1:], axis=1)
    resp_original_TT_hrd = resps_original[:,0] + weight_original*resp_original_sk
    resp_original_hrd = resp_original_TT_hrd + np.sum(resps_original[:,1:], axis=1)
    inv_resp_original_TT_hrd = np.zeros_like(l, dtype=np.complex_); inv_resp_original_TT_hrd[1:] = 1/(resp_original_TT_hrd)[1:]
    inv_resp_original_hrd = np.zeros_like(l, dtype=np.complex_); inv_resp_original_hrd[1:] = 1/(resp_original_hrd)[1:]

    # Response correct GMV
    # N is 1/R
    plm_gmv_resp_corr = hp.almxfl(plm_gmv,inv_resp_gmv)
    plm_gmv_A_resp_corr = hp.almxfl(plm_gmv_A,inv_resp_gmv_A)
    plm_gmv_B_resp_corr = hp.almxfl(plm_gmv_B,inv_resp_gmv_B)
    plm_gmv_resp_corr_hrd = hp.almxfl(plm_gmv_hrd,inv_resp_gmv_hrd)
    plm_gmv_A_resp_corr_hrd = hp.almxfl(plm_gmv_A_hrd,inv_resp_gmv_A_hrd)

    # Get GMV spectra
    auto_gmv = hp.alm2cl(plm_gmv_resp_corr, plm_gmv_resp_corr, lmax=lmax) * (l*(l+1))**2/4
    auto_gmv_A = hp.alm2cl(plm_gmv_A_resp_corr, plm_gmv_A_resp_corr, lmax=lmax) * (l*(l+1))**2/4
    auto_gmv_B = hp.alm2cl(plm_gmv_B_resp_corr, plm_gmv_B_resp_corr, lmax=lmax) * (l*(l+1))**2/4
    auto_gmv_hrd = hp.alm2cl(plm_gmv_resp_corr_hrd, plm_gmv_resp_corr_hrd, lmax=lmax) * (l*(l+1))**2/4
    auto_gmv_A_hrd = hp.alm2cl(plm_gmv_A_resp_corr_hrd, plm_gmv_A_resp_corr_hrd, lmax=lmax) * (l*(l+1))**2/4

    # Response correct healqest
    plm_original_resp_corr = hp.almxfl(plm_original,inv_resp_original)
    plm_TT_resp_corr = hp.almxfl(plms_original[:,0],inv_resps_original[:,0])
    plm_EE_resp_corr = hp.almxfl(plms_original[:,1],inv_resps_original[:,1])
    plm_TE_resp_corr = hp.almxfl(plms_original[:,2],inv_resps_original[:,2])
    plm_TB_resp_corr = hp.almxfl(plms_original[:,3],inv_resps_original[:,3])
    plm_EB_resp_corr = hp.almxfl(plms_original[:,4],inv_resps_original[:,4])
    plm_original_resp_corr_hrd = hp.almxfl(plm_original_hrd,inv_resp_original_hrd)
    plm_TT_resp_corr_hrd = hp.almxfl(plm_original_TT_hrd,inv_resp_original_TT_hrd)

    # Get healqest spectra
    auto_original = hp.alm2cl(plm_original_resp_corr, plm_original_resp_corr, lmax=lmax) * (l*(l+1))**2/4
    auto_TT = hp.alm2cl(plm_TT_resp_corr, plm_TT_resp_corr, lmax=lmax) * (l*(l+1))**2/4
    auto_TE = hp.alm2cl(plm_TE_resp_corr, plm_TE_resp_corr, lmax=lmax) * (l*(l+1))**2/4
    auto_EE = hp.alm2cl(plm_EE_resp_corr, plm_EE_resp_corr, lmax=lmax) * (l*(l+1))**2/4
    auto_TB = hp.alm2cl(plm_TB_resp_corr, plm_TB_resp_corr, lmax=lmax) * (l*(l+1))**2/4
    auto_EB = hp.alm2cl(plm_EB_resp_corr, plm_EB_resp_corr, lmax=lmax) * (l*(l+1))**2/4
    auto_original_hrd = hp.alm2cl(plm_original_resp_corr_hrd, plm_original_resp_corr_hrd, lmax=lmax) * (l*(l+1))**2/4
    auto_TT_hrd = hp.alm2cl(plm_TT_resp_corr_hrd, plm_TT_resp_corr_hrd, lmax=lmax) * (l*(l+1))**2/4

    # Get N0 correction (remember these still need (l*(l+1))**2/4 factor to convert to kappa)
    n0_gmv = get_n0(sims=np.arange(10)+100,qetype='gmv',fluxlim=fluxlim,lmax=lmax,u=u,config_file=config_file,dir_out=dir_out)
    n0_original = get_n0(sims=np.arange(10)+100,qetype='original',fluxlim=fluxlim,lmax=lmax,u=u,config_file=config_file,dir_out=dir_out)

    # Theory spectrum
    clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
    ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
    clkk = slpp * (l*(l+1))**2/4

    # Cross correlate with input plm
    #input_plm = hp.read_alm(f'')
    #input_plm = utils.reduce_lmax(input_plm,lmax=lmax)
    #cross_gmv = hp.alm2cl(input_plm, plm_gmv_resp_corr, lmax=lmax) * (l*(l+1))**2/4
    #cross_original = hp.alm2cl(input_plm, plm_original_resp_corr, lmax=lmax) * (l*(l+1))**2/4
    #auto_input = hp.alm2cl(input_plm, input_plm, lmax=lmax) * (l*(l+1))**2/4

    # Plot
    plt.figure(0)
    plt.clf()
    plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')
    plt.plot(l, auto_gmv_hrd, color='firebrick', label="Auto Spectrum (GMV, hardened)")
    plt.plot(l, auto_gmv_A_hrd, color='peru', label=f'Auto Spectrum (GMV [TT, EE, TE], hardened)')
    plt.plot(l, auto_gmv, color='darkblue', linestyle='-', label="Auto Spectrum (GMV)")
    plt.plot(l, auto_gmv_A, color='dodgerblue', linestyle='-', label=f'Auto Spectrum (GMV [TT, EE, TE])')
    #plt.plot(l, auto_gmv_hrd - n0_gmv['total_hrd'] * (l*(l+1))**2/4, color='firebrick', linestyle='-', label="Auto Spectrum - N0 (GMV, hardened)")
    #plt.plot(l, auto_gmv_A_hrd - n0_gmv['TTEETE_hrd'] * (l*(l+1))**2/4, color='peru', linestyle='-', label=f'Auto Spectrum - N0 (GMV [TT, EE, TE], hardened)')
    #plt.plot(l, auto_gmv - n0_gmv['total'] * (l*(l+1))**2/4, color='darkblue', linestyle='-', label="Auto Spectrum - N0 (GMV)")
    #plt.plot(l, auto_gmv_A - n0_gmv['TTEETE'] * (l*(l+1))**2/4, color='dodgerblue', linestyle='-', label=f'Auto Spectrum - N0 (GMV [TT, EE, TE])')
    plt.plot(l, inv_resp_gmv_hrd * (l*(l+1))**2/4, color='lightcoral', linestyle='--', label='1/R (GMV, hardened)')
    plt.plot(l, inv_resp_gmv_A_hrd * (l*(l+1))**2/4, color='sandybrown', linestyle='--', label='1/R (GMV [TT, EE, TE], hardened)')
    plt.plot(l, inv_resp_gmv * (l*(l+1))**2/4, color='cornflowerblue', linestyle='--', label='1/R (GMV)')
    plt.plot(l, inv_resp_gmv_A * (l*(l+1))**2/4, color='deepskyblue', linestyle='--', label='1/R (GMV [TT, EE, TE])')
    #plt.plot(l, n0_gmv['total_hrd'] * (l*(l+1))**2/4, color='lightcoral', linestyle='--', label='N0 (GMV, hardened)')
    #plt.plot(l, n0_gmv['TTEETE_hrd'] * (l*(l+1))**2/4, color='sandybrown', linestyle='--', label= 'N0 (GMV [TT, EE, TE], hardened)')
    #plt.plot(l, n0_gmv['total'] * (l*(l+1))**2/4, color='cornflowerblue', linestyle='--', label='N0 (GMV)')
    #plt.plot(l, n0_gmv['TTEETE'] * (l*(l+1))**2/4, color='deepskyblue', linestyle='--', label='N0 (GMV [TT, EE, TE])')
    plt.ylabel("$C_\ell^{\kappa\kappa}$")
    plt.xlabel('$\ell$')
    plt.title(f'Spectra for Sim {sim}, with Point Sources in T (GMV)')
    plt.legend(loc='upper right', fontsize='small')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,lmax)
    #plt.ylim(5e-9,1e-6)
    plt.ylim(1e-10,1e-4)
    if save_fig:
        plt.savefig(dir_out+f'/figs/prfhrd_comparison_spec_gmv_fluxlim{fluxlim:.3f}.png')
        #plt.savefig(dir_out+f'/figs/prfhrd_comparison_spec_gmv_fluxlim{fluxlim:.3f}_n0_subtracted.png')

    plt.figure(1)
    plt.clf()
    plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')
    plt.plot(l, auto_original_hrd, color='firebrick', label="Auto Spectrum (Original, hardened)")
    plt.plot(l, auto_TT_hrd, color='peru', label="Auto Spectrum (Original TT, hardened)")
    plt.plot(l, auto_original, color='darkblue', label="Auto Spectrum (Original)")
    plt.plot(l, auto_TT, color='dodgerblue', label="Auto Spectrum (Original TT)")
    #plt.plot(l, auto_original_hrd - n0_original['total_hrd'] * (l*(l+1))**2/4, color='firebrick', label="Auto Spectrum - N0 (Original, hardened)")
    #plt.plot(l, auto_TT_hrd - n0_original['TT_hrd'] * (l*(l+1))**2/4, color='peru', label="Auto Spectrum - N0 (Original TT, hardened)")
    #plt.plot(l, auto_original - n0_original['total'] * (l*(l+1))**2/4, color='darkblue', label="Auto Spectrum - N0 (Original)")
    #plt.plot(l, auto_TT - n0_original['TT'] * (l*(l+1))**2/4, color='dodgerblue', label="Auto Spectrum - N0 (Original TT)")
    plt.plot(l, inv_resp_original_hrd * (l*(l+1))**2/4, color='lightcoral', linestyle='--', label='1/R (Original, hardened)')
    plt.plot(l, inv_resp_original_TT_hrd * (l*(l+1))**2/4, color='sandybrown', linestyle='--', label='1/R (Original TT, hardened)')
    plt.plot(l, inv_resp_original * (l*(l+1))**2/4, color='cornflowerblue', linestyle='--', label='1/R (Original)')
    plt.plot(l, inv_resps_original[:,0] * (l*(l+1))**2/4, color='deepskyblue', linestyle='--', label='1/R (Original TT)')
    #plt.plot(l, n0_gmv['total_hrd'] * (l*(l+1))**2/4, color='lightcoral', linestyle='--', label='N0 (Original, hardened)')
    #plt.plot(l, n0_gmv['TT_hrd'] * (l*(l+1))**2/4, color='sandybrown', linestyle='--', label='N0 (Original TT, hardened)')
    #plt.plot(l, n0_gmv['total'] * (l*(l+1))**2/4, color='cornflowerblue', linestyle='--', label='N0 (Original)')
    #plt.plot(l, n0_gmv['TT'] * (l*(l+1))**2/4, color='deepskyblue', linestyle='--', label='N0 (Original TT)')
    plt.ylabel("$C_\ell^{\kappa\kappa}$")
    plt.xlabel('$\ell$')
    plt.title(f'Spectra for Sim {sim}, with Point Sources in T (healqest)')
    plt.legend(loc='upper right', fontsize='small')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,lmax)
    #plt.ylim(5e-9,1e-6)
    plt.ylim(1e-10,1e-4)
    if save_fig:
        plt.savefig(dir_out+f'/figs/prfhrd_comparison_spec_original_fluxlim{fluxlim:.3f}.png')
        #plt.savefig(dir_out+f'/figs/prfhrd_comparison_spec_original_fluxlim{fluxlim:.3f}_n0_subtracted.png')

def get_n0(sims=np.arange(10)+100,qetype='gmv',fluxlim=0.200,
           lmax=4096,u=np.ones(4097,dtype=np.complex_),config_file='profhrd_yuka.yaml',dir_out='/scratch/users/yukanaka/gmv/'):
    '''
    Get N0 bias. qetype should be 'gmv' or 'original'.
    '''
    num = len(sims) - 1
    l = np.arange(0,lmax+1)
    append = f'tsrc_fluxlim{fluxlim:.3f}'
    config = utils.parse_yaml(config_file)
    clfile = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat'
    filename = f'/scratch/users/yukanaka/gmv/n0/n0_{num}simpairs_healqest_{qetype}_lmax{lmax}_nside8192_{append}.npy'
    if os.path.isfile(filename):
        n0 = np.load(filename)
    elif qetype == 'gmv':
        gmv_resp_data = np.genfromtxt('gmv_resp/True_variance_individual_custom_lmin2.0_lmaxT4096_lmaxP4096_beam0_noise0_50.txt')
        inv_resp_gmv = gmv_resp_data[:,3] / l**2
        inv_resp_gmv_A = gmv_resp_data[:,1] / l**2
        inv_resp_gmv_B = gmv_resp_data[:,2] / l**2
        gmv_resp_data_ss = np.genfromtxt('gmv_resp/True_variance_individual_custom_lmin2.0_lmaxT4096_lmaxP4096_beam0_noise0_50_PRF_SS.txt')
        gmv_resp_data_sk = np.genfromtxt('gmv_resp/True_variance_individual_custom_lmin2.0_lmaxT4096_lmaxP4096_beam0_noise0_50_PRF_SK.txt')
        #TODO: for some reason we need a -1 factor for SK
        gmv_resp_data_sk *= -1
        inv_resp_gmv_A_ss = gmv_resp_data_ss[:,1] / l**2
        inv_resp_gmv_A_sk = gmv_resp_data_sk[:,1] / l**2
        # Abhi's code calculates the reconstruction noise for d field rather than phi field, see GMV_QE.py, line 292 for example
        resp_gmv = 1/inv_resp_gmv
        resp_gmv_A = 1/inv_resp_gmv_A
        resp_gmv_B = 1/inv_resp_gmv_B
        resp_gmv_A_sk = 1/inv_resp_gmv_A_sk
        resp_gmv_A_ss = 1/inv_resp_gmv_A_ss
        weight_gmv = -1 * resp_gmv_A_sk / resp_gmv_A_ss
        resp_gmv_A_hrd = resp_gmv_A + weight_gmv*resp_gmv_A_sk
        resp_gmv_hrd = resp_gmv_A_hrd + resp_gmv_B
        inv_resp_gmv_A_hrd = np.zeros_like(l, dtype=np.complex_); inv_resp_gmv_A_hrd[1:] = 1/(resp_gmv_A_hrd)[1:]
        inv_resp_gmv_hrd = np.zeros_like(l, dtype=np.complex_); inv_resp_gmv_hrd[1:] = 1/(resp_gmv_hrd)[1:]
        n0 = {'total':0, 'TTEETE':0, 'TBEB':0, 'total_hrd':0, 'TTEETE_hrd':0}
        for i, sim1 in enumerate(sims[:-1]):
            sim2 = sim1 + 1
            # These are lensed
            plm_gmv_ij = np.load(dir_out+f'/plm_all_healqest_gmv_seed1{sim1}_seed2{sim2}_lmax{lmax}_nside{nside}_{append}.npy')
            plm_gmv_A_ij = np.load(dir_out+f'/plm_TTEETE_healqest_gmv_seed1{sim1}_seed2{sim2}_lmax{lmax}_nside{nside}_{append}.npy')
            plm_gmv_B_ij = np.load(dir_out+f'/plm_TBEB_healqest_gmv_seed1{sim1}_seed2{sim2}_lmax{lmax}_nside{nside}_{append}.npy')
            # Harden!
            glm_prf_A_ij = np.load(dir_out+f'/plm_TTEETEprf_healqest_gmv_seed1{sim1}_seed2{sim2}_lmax{lmax}_nside{nside}_{append}.npy')
            plm_gmv_A_hrd_ij = plm_gmv_A_ij + hp.almxfl(glm_prf_A_ij, weight_gmv)
            plm_gmv_hrd_ij = plm_gmv_A_hrd_ij + plm_gmv_B_ij
            # Response correct GMV
            # N is 1/R
            plm_all_ij = hp.almxfl(plm_gmv_ij,inv_resp_gmv)
            plm_A_ij = hp.almxfl(plm_gmv_A_ij,inv_resp_gmv_A)
            plm_B_ij = hp.almxfl(plm_gmv_B_ij,inv_resp_gmv_B)
            plm_all_hrd_ij = hp.almxfl(plm_gmv_hrd_ij,inv_resp_gmv_hrd)
            plm_A_hrd_ij = hp.almxfl(plm_gmv_A_hrd_ij,inv_resp_gmv_A_hrd)
            # Get ij auto spectra <ijij>
            auto = hp.alm2cl(plm_all_ij, plm_all_ij, lmax=lmax)
            auto_A = hp.alm2cl(plm_A_ij, plm_A_ij, lmax=lmax)
            auto_B = hp.alm2cl(plm_B_ij, plm_B_ij, lmax=lmax)
            auto_hrd = hp.alm2cl(plm_all_hrd_ij, plm_all_hrd_ij, lmax=lmax)
            auto_A_hrd = hp.alm2cl(plm_A_hrd_ij, plm_A_hrd_ij, lmax=lmax)
            # Now get the ji sims
            plm_gmv_ji = np.load(dir_out+f'/plm_all_healqest_gmv_seed1{sim2}_seed2{sim1}_lmax{lmax}_nside{nside}_{append}.npy')
            plm_gmv_A_ji = np.load(dir_out+f'/plm_TTEETE_healqest_gmv_seed1{sim2}_seed2{sim1}_lmax{lmax}_nside{nside}_{append}.npy')
            plm_gmv_B_ji = np.load(dir_out+f'/plm_TBEB_healqest_gmv_seed1{sim2}_seed2{sim1}_lmax{lmax}_nside{nside}_{append}.npy')
            # Harden!
            glm_prf_A_ji = np.load(dir_out+f'/plm_TTEETEprf_healqest_gmv_seed1{sim2}_seed2{sim1}_lmax{lmax}_nside{nside}_{append}.npy')
            plm_gmv_A_hrd_ji = plm_gmv_A_ji + hp.almxfl(glm_prf_A_ji, weight_gmv)
            plm_gmv_hrd_ji = plm_gmv_A_hrd_ji + plm_gmv_B_ji
            # Response correct GMV
            # N is 1/R
            plms_all_ji = hp.almxfl(plm_gmv_ji,inv_resp_gmv)
            plms_A_ji = hp.almxfl(plm_gmv_A_ji,inv_resp_gmv_A)
            plms_B_ji = hp.almxfl(plm_gmv_B_ji,inv_resp_gmv_B)
            plms_all_hrd_ji = hp.almxfl(plm_gmv_hrd_ji,inv_resp_gmv_hrd)
            plms_A_hrd_ji = hp.almxfl(plm_gmv_A_hrd_ji,inv_resp_gmv_A_hrd)
            # Get cross spectra <ijji>
            cross = hp.alm2cl(plm_all_ij, plm_all_ji, lmax=lmax)
            cross_A = hp.alm2cl(plm_A_ij, plm_A_ji, lmax=lmax)
            cross_B = hp.alm2cl(plm_B_ij, plm_B_ji, lmax=lmax)
            cross_hrd = hp.alm2cl(plm_all_hrd_ij, plm_all_hrd_ji, lmax=lmax)
            cross_A_hrd = hp.alm2cl(plm_A_hrd_ij, plm_A_hrd_ji, lmax=lmax)
            n0['total'] += auto + cross
            n0['TTEETE'] += auto_A + cross_A
            n0['TBEB'] += auto_B + cross_B
            n0['total_hrd'] += auto_hrd + cross_hrd
            n0['TTEETE_hrd'] += auto_A_hrd + cross_A_hrd
        n0['total'] *= 1/num
        n0['TTEETE'] *= 1/num
        n0['TBEB'] *= 1/num
        n0['total_hrd'] *= 1/num
        n0['TTEETE_hrd'] *= 1/num
        np.save(filename, n0)
    elif qetype == 'original':
        ests = ['TT', 'EE', 'TE', 'TB', 'EB']
        resps_original = np.zeros((len(l),5), dtype=np.complex_)
        inv_resps_original = np.zeros((len(l),5) ,dtype=np.complex_)
        for i, est in enumerate(ests):
            resps_original[:,i] = get_analytic_response(est,config,lmax,fwhm=0,nlev_t=0,nlev_p=0,u=u,clfile=clfile)
            inv_resps_original[1:,i] = 1/(resps_original)[1:,i]
        resp_original = np.sum(resps_original, axis=1)
        inv_resp_original = np.zeros_like(l,dtype=np.complex_); inv_resp_original[1:] = 1/(resp_original)[1:]
        resp_original_ss = get_analytic_response('TTprf',config,lmax,fwhm=0,nlev_t=0,nlev_p=0,u=u,clfile=clfile,
                                                 filename='/scratch/users/yukanaka/gmv/resp/an_resp_SS_TTprf_healqest_lmax{}_fwhm0_nlevt0_nlevp0.npy'.format(lmax))
        resp_original_sk = get_analytic_response('TTprf',config,lmax,fwhm=0,nlev_t=0,nlev_p=0,u=u,clfile=clfile,
                                                 filename='/scratch/users/yukanaka/gmv/resp/an_resp_SK_TTprf_TT_healqest_lmax{}_fwhm0_nlevt0_nlevp0.npy'.format(lmax),
                                                 qeZA=weights_combined_qestobj.weights('TT',lmax,config,cltype='len'))
        weight_original = -1 * resp_original_sk / resp_original_ss
        resp_original_TT_hrd = resps_original[:,0] + weight_original*resp_original_sk
        resp_original_hrd = resp_original_TT_hrd + np.sum(resps_original[:,1:], axis=1)
        inv_resp_original_TT_hrd = np.zeros_like(l, dtype=np.complex_); inv_resp_original_TT_hrd[1:] = 1/(resp_original_TT_hrd)[1:]
        inv_resp_original_hrd = np.zeros_like(l, dtype=np.complex_); inv_resp_original_hrd[1:] = 1/(resp_original_hrd)[1:]
        n0 = {'total':0, 'TT':0, 'EE':0, 'TE':0, 'TB':0, 'EB':0, 'total_hrd':0, 'TT_hrd':0}
        for i, sim1 in enumerate(sims[:-1]):
            sim2 = sim1 + 1
            # These are lensed
            plm_TT_ij = np.load(dir_out+f'/plm_TT_healqest_seed1{sim1}_seed2{sim2}_lmax{lmax}_nside{nside}_{append}.npy')
            plm_EE_ij = np.load(dir_out+f'/plm_EE_healqest_seed1{sim1}_seed2{sim2}_lmax{lmax}_nside{nside}_{append}.npy')
            plm_TE_ij = np.load(dir_out+f'/plm_TE_healqest_seed1{sim1}_seed2{sim2}_lmax{lmax}_nside{nside}_{append}.npy')
            plm_TB_ij = np.load(dir_out+f'/plm_TB_healqest_seed1{sim1}_seed2{sim2}_lmax{lmax}_nside{nside}_{append}.npy')
            plm_EB_ij = np.load(dir_out+f'/plm_EB_healqest_seed1{sim1}_seed2{sim2}_lmax{lmax}_nside{nside}_{append}.npy')
            plm_total_ij = plm_TT_ij + plm_EE_ij + plm_TE_ij + plm_TB_ij + plm_EB_ij
            # Harden!
            glm_prf_TTprf_ij = np.load(dir_out+f'/plm_TTprf_healqest_seed1{sim1}_seed2{sim2}_lmax{lmax}_nside{nside}_{append}.npy')
            plm_original_TT_hrd_ij = plm_TT_ij + hp.almxfl(glm_prf_TTprf_ij, weight_original)
            plm_original_hrd_ij = plm_original_TT_hrd_ij + plm_EE_ij + plm_TE_ij + plm_TB_ij + plm_EB_ij
            # Response correct healqest
            plm_total_ij = hp.almxfl(plm_total_ij,inv_resp_original)
            plm_TT_ij = hp.almxfl(plm_TT_ij,inv_resps_original[:,0])
            plm_EE_ij = hp.almxfl(plm_EE_ij,inv_resps_original[:,1])
            plm_TE_ij = hp.almxfl(plm_TE_ij,inv_resps_original[:,2])
            plm_TB_ij = hp.almxfl(plm_TB_ij,inv_resps_original[:,3])
            plm_EB_ij = hp.almxfl(plm_EB_ij,inv_resps_original[:,4])
            plm_total_hrd_ij = hp.almxfl(plm_original_hrd_ij,inv_resp_original_hrd)
            plm_TT_hrd_ij = hp.almxfl(plm_original_TT_hrd_ij,inv_resp_original_TT_hrd)
            # Get ij auto spectra <ijij>
            auto = hp.alm2cl(plm_total_ij, plm_total_ij, lmax=lmax)
            auto_TT = hp.alm2cl(plm_TT_ij, plm_TT_ij, lmax=lmax)
            auto_TE = hp.alm2cl(plm_TE_ij, plm_TE_ij, lmax=lmax)
            auto_EE = hp.alm2cl(plm_EE_ij, plm_EE_ij, lmax=lmax)
            auto_TB = hp.alm2cl(plm_TB_ij, plm_TB_ij, lmax=lmax)
            auto_EB = hp.alm2cl(plm_EB_ij, plm_EB_ij, lmax=lmax)
            auto_hrd = hp.alm2cl(plm_total_hrd_ij, plm_total_hrd_ij, lmax=lmax)
            auto_TT_hrd = hp.alm2cl(plm_TT_hrd_ij, plm_TT_hrd_ij, lmax=lmax)
            # Now get the ji sims
            plm_TT_ji = np.load(dir_out+f'/plm_TT_healqest_seed1{sim2}_seed2{sim1}_lmax{lmax}_nside{nside}_{append}.npy')
            plm_EE_ji = np.load(dir_out+f'/plm_EE_healqest_seed1{sim2}_seed2{sim1}_lmax{lmax}_nside{nside}_{append}.npy')
            plm_TE_ji = np.load(dir_out+f'/plm_TE_healqest_seed1{sim2}_seed2{sim1}_lmax{lmax}_nside{nside}_{append}.npy')
            plm_TB_ji = np.load(dir_out+f'/plm_TB_healqest_seed1{sim2}_seed2{sim1}_lmax{lmax}_nside{nside}_{append}.npy')
            plm_EB_ji = np.load(dir_out+f'/plm_EB_healqest_seed1{sim2}_seed2{sim1}_lmax{lmax}_nside{nside}_{append}.npy')
            plm_total_ji = plm_TT_ji + plm_EE_ji + plm_TE_ji + plm_TB_ji + plm_EB_ji
            # Harden!
            glm_prf_TTprf_ji = np.load(dir_out+f'/plm_TTprf_healqest_seed1{sim2}_seed2{sim1}_lmax{lmax}_nside{nside}_{append}.npy')
            plm_original_TT_hrd_ji = plm_TT_ji + hp.almxfl(glm_prf_TTprf_ji, weight_original)
            plm_original_hrd_ji = plm_original_TT_hrd_ji + plm_EE_ji + plm_TE_ji + plm_TB_ji + plm_EB_ji
            # Response correct healqest
            plm_total_ji = hp.almxfl(plm_total_ji,inv_resp_original)
            plm_TT_ji = hp.almxfl(plm_TT_ji,inv_resps_original[:,0])
            plm_EE_ji = hp.almxfl(plm_EE_ji,inv_resps_original[:,1])
            plm_TE_ji = hp.almxfl(plm_TE_ji,inv_resps_original[:,2])
            plm_TB_ji = hp.almxfl(plm_TB_ji,inv_resps_original[:,3])
            plm_EB_ji = hp.almxfl(plm_EB_ji,inv_resps_original[:,4])
            plm_total_hrd_ji = hp.almxfl(plm_original_hrd_ji,inv_resp_original_hrd)
            plm_TT_hrd_ji = hp.almxfl(plm_original_TT_hrd_ji,inv_resp_original_TT_hrd)
            # Get cross spectra <ijji>
            cross = hp.alm2cl(plm_total_ij, plm_total_ji, lmax=lmax)
            cross_TT = hp.alm2cl(plm_TT_ij, plm_TT_ji, lmax=lmax)
            cross_TE = hp.alm2cl(plm_TE_ij, plm_TE_ji, lmax=lmax)
            cross_EE = hp.alm2cl(plm_EE_ij, plm_EE_ji, lmax=lmax)
            cross_TB = hp.alm2cl(plm_TB_ij, plm_TB_ji, lmax=lmax)
            cross_EB = hp.alm2cl(plm_EB_ij, plm_EB_ji, lmax=lmax)
            cross_hrd = hp.alm2cl(plm_total_hrd_ij, plm_total_hrd_ji, lmax=lmax)
            cross_TT_hrd = hp.alm2cl(plm_TT_hrd_ij, plm_TT_hrd_ji, lmax=lmax)
            n0['total'] += auto + cross
            n0['TT'] += auto_TT + cross_TT
            n0['EE'] += auto_TE + cross_TE
            n0['TE'] += auto_EE + cross_EE
            n0['TB'] += auto_TB + cross_TB
            n0['EB'] += auto_EB + cross_EB
            n0['total_hrd'] += auto_hrd + cross_hrd
            n0['TT_hrd'] += auto_TT_hrd + cross_TT_hrd
        n0['total'] *= 1/num
        n0['TT'] *= 1/num
        n0['EE'] *= 1/num
        n0['TE'] *= 1/num
        n0['TB'] *= 1/num
        n0['EB'] *= 1/num
        n0['total_hrd'] *= 1/num
        n0['TT_hrd'] *= 1/num
        np.save(filename, n0)
    else:
        print('Invalid argument qetype.')
    return n0

def compare_lensing_resp(lmax=1000,dir_out='/scratch/users/yukanaka/gmv/',config_file='profhrd_yuka.yaml',save_fig=True):
    # Flat sky healqest response
    clfile = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat'
    config = utils.parse_yaml(config_file)
    l = np.arange(0,lmax+1)
    resp_healqest_TT_kk = get_analytic_response('TT',config,lmax,fwhm=1,nlev_t=3,nlev_p=3,clfile=clfile)
    inv_resp_healqest_TT_kk = np.zeros_like(l,dtype=np.complex_); inv_resp_healqest_TT_kk[1:] = 1/(resp_healqest_TT_kk)[1:]

    # GMV response
    gmv_resp_data_kk = np.genfromtxt('gmv_resp/True_variance_individual_custom_lmin2.0_lmaxT1000_lmaxP1000_beam1_noise3_50.txt')
    inv_resp_gmv_kk = gmv_resp_data_kk[:,3] / l**2
    inv_resp_gmv_A_kk = gmv_resp_data_kk[:,1] / l**2
    inv_resp_gmv_B_kk = gmv_resp_data_kk[:,2] / l**2

    # Theory spectrum
    clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
    ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
    clkk = slpp * (l*(l+1))**2/4

    plt.figure(0)
    plt.clf()
    plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')
    plt.plot(l, inv_resp_healqest_TT_kk * (l*(l+1))**2/4, color='maroon', linestyle=':', label='$1/R^{KK}$ (Healqest TT)')
    plt.plot(l, inv_resp_gmv_A_kk * (l*(l+1))**2/4, color='darkgreen', linestyle=':', label='$1/R^{KK}$ (GMV [TT, EE, TE])')
    plt.ylabel("$C_\ell^{\kappa\kappa}$")
    plt.xlabel('$\ell$')
    plt.title('$1/R$')
    plt.legend(loc='upper left', fontsize='small')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,lmax)
    #plt.ylim(8e-9,1e-5)
    if save_fig:
        plt.savefig(dir_out+f'/figs/lensing_response_comparison_noiseless.png')

def compare_profile_hardening_resp(u=None,lmax=4096,nside=8192,dir_out='/scratch/users/yukanaka/gmv/',
                                   config_file='profhrd_yuka.yaml',save_fig=True):
    clfile = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat'
    config = utils.parse_yaml(config_file)
    l = np.arange(0,lmax+1)
    if u is None:
        #u=np.ones(4097,dtype=np.complex_)
        u = hp.sphtfunc.gauss_beam(1*(np.pi/180.)/60., lmax=lmax)

    # Flat sky healqest response
    resp_healqest_TT = get_analytic_response('TTprf',config,lmax,fwhm=1,nlev_t=5,nlev_p=5,u=u,clfile=clfile)
    #resp_healqest_TT = get_analytic_response('TTprf',config,lmax,fwhm=0,nlev_t=0,nlev_p=0,u=u,clfile=clfile,
    #                                          filename='/scratch/users/yukanaka/gmv/resp/an_resp_SS_TTprf_healqest_lmax{}_fwhm0_nlevt0_nlevp0.npy'.format(lmax))
    inv_resp_healqest_TT = np.zeros_like(l,dtype=np.complex_); inv_resp_healqest_TT[1:] = 1/(resp_healqest_TT)[1:]
    resp_healqest_TT_sk = get_analytic_response('TTprf',config,lmax,fwhm=1,nlev_t=5,nlev_p=5,u=u,clfile=clfile,
                                                filename='/scratch/users/yukanaka/gmv/resp/an_resp_TTprf_healqest_lmax{}_fwhm1_nlevt5_nlevp5_SK_TT.npy'.format(lmax),
                                                qeZA=weights_combined_qestobj.weights('TT',lmax,config,cltype='len'))
    #resp_healqest_TT_sk = get_analytic_response('TTprf',config,lmax,fwhm=0,nlev_t=0,nlev_p=0,u=u,clfile=clfile,
    #                                         filename='/scratch/users/yukanaka/gmv/resp/an_resp_SK_TTprf_TT_healqest_lmax{}_fwhm0_nlevt0_nlevp0.npy'.format(lmax),
    #                                         qeZA=weights_combined_qestobj.weights('TT',lmax,config,cltype='len'))
    inv_resp_healqest_TT_sk = np.zeros_like(l,dtype=np.complex_); inv_resp_healqest_TT_sk[1:] = 1/(resp_healqest_TT_sk)[1:]
    resp_healqest_TT_kk = get_analytic_response('TT',config,lmax,fwhm=1,nlev_t=5,nlev_p=5,clfile=clfile)
    #resp_healqest_TT_kk = get_analytic_response('TT',config,lmax,fwhm=0,nlev_t=0,nlev_p=0,clfile=clfile)
    inv_resp_healqest_TT_kk = np.zeros_like(l,dtype=np.complex_); inv_resp_healqest_TT_kk[1:] = 1/(resp_healqest_TT_kk)[1:]

    # GMV response
    gmv_resp_data = np.genfromtxt('gmv_resp/True_variance_individual_custom_lmin0.0_lmaxT4096_lmaxP4096_beam1_noise5_50_PRF.txt')
    #gmv_resp_data = np.genfromtxt('gmv_resp/True_variance_individual_custom_lmin2.0_lmaxT4096_lmaxP4096_beam0_noise0_50_PRF_SS.txt')
    inv_resp_gmv = gmv_resp_data[:,3] / l**2
    inv_resp_gmv_A = gmv_resp_data[:,1] / l**2
    inv_resp_gmv_B = gmv_resp_data[:,2] / l**2
    gmv_resp_data_sk = np.genfromtxt('gmv_resp/True_variance_individual_custom_lmin0.0_lmaxT4096_lmaxP4096_beam1_noise5_50_PRF_SK.txt')
    #gmv_resp_data_sk = np.genfromtxt('gmv_resp/True_variance_individual_custom_lmin2.0_lmaxT4096_lmaxP4096_beam0_noise0_50_PRF_SK.txt')
    inv_resp_gmv_sk = gmv_resp_data_sk[:,3] / l**2
    inv_resp_gmv_A_sk = gmv_resp_data_sk[:,1] / l**2
    inv_resp_gmv_B_sk = gmv_resp_data_sk[:,2] / l**2
    gmv_resp_data_kk = np.genfromtxt('gmv_resp/True_variance_individual_custom_lmin0.0_lmaxT4096_lmaxP4096_beam1_noise5_50.txt')
    #gmv_resp_data_kk = np.genfromtxt('gmv_resp/True_variance_individual_custom_lmin2.0_lmaxT4096_lmaxP4096_beam0_noise0_50.txt')
    inv_resp_gmv_kk = gmv_resp_data_kk[:,3] / l**2
    inv_resp_gmv_A_kk = gmv_resp_data_kk[:,1] / l**2
    inv_resp_gmv_B_kk = gmv_resp_data_kk[:,2] / l**2
    gmv_resp_data_ftt_only = np.genfromtxt('gmv_resp/True_variance_individual_custom_lmin0.0_lmaxT4096_lmaxP4096_beam1_noise5_50_PRF_SS_FTT_only_no_clte.txt')
    #gmv_resp_data_ftt_only = np.genfromtxt('gmv_resp/True_variance_individual_custom_lmin2.0_lmaxT4096_lmaxP4096_beam0_noise0_50_PRF_SS_FTT_only_no_clte.txt')
    inv_resp_gmv_ftt_only = gmv_resp_data_ftt_only[:,3] / l**2
    inv_resp_gmv_A_ftt_only = gmv_resp_data_ftt_only[:,1] / l**2
    inv_resp_gmv_B_ftt_only = gmv_resp_data_ftt_only[:,2] / l**2
    gmv_resp_data_sk_ftt_only = np.genfromtxt('gmv_resp/True_variance_individual_custom_lmin0.0_lmaxT4096_lmaxP4096_beam1_noise5_50_PRF_SK_FTT_only_no_clte.txt')
    #gmv_resp_data_sk_ftt_only = np.genfromtxt('gmv_resp/True_variance_individual_custom_lmin2.0_lmaxT4096_lmaxP4096_beam0_noise0_50_PRF_SK_FTT_only_no_clte.txt')
    inv_resp_gmv_sk_ftt_only = gmv_resp_data_sk_ftt_only[:,3] / l**2
    inv_resp_gmv_A_sk_ftt_only = gmv_resp_data_sk_ftt_only[:,1] / l**2
    inv_resp_gmv_B_sk_ftt_only = gmv_resp_data_sk_ftt_only[:,2] / l**2
    gmv_resp_data_kk_ftt_only = np.genfromtxt('gmv_resp/True_variance_individual_custom_lmin0.0_lmaxT4096_lmaxP4096_beam1_noise5_50_FTT_only_no_clte.txt')
    inv_resp_gmv_kk_ftt_only = gmv_resp_data_kk_ftt_only[:,3] / l**2
    inv_resp_gmv_A_kk_ftt_only = gmv_resp_data_kk_ftt_only[:,1] / l**2
    inv_resp_gmv_B_kk_ftt_only = gmv_resp_data_kk_ftt_only[:,2] / l**2

    # Theory spectrum
    clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
    ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
    clkk = slpp * (l*(l+1))**2/4

    plt.figure(0)
    plt.clf()
    plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')
    plt.plot(l, inv_resp_healqest_TT * (l*(l+1))**2/4, color='firebrick', linestyle='-', label='$1/R^{SS}$ (Healqest TT)')
    plt.plot(l, inv_resp_healqest_TT_sk * (l*(l+1))**2/4, color='lightcoral', linestyle='--', label='$1/R^{SK}$ (Healqest TT)')
    plt.plot(l, inv_resp_healqest_TT_kk * (l*(l+1))**2/4, color='maroon', linestyle=':', label='$1/R^{KK}$ (Healqest TT)')
    ##plt.plot(l, inv_resp_gmv * (l*(l+1))**2/4, color='darkkhaki', linestyle='-', label='$1/R^{SS}$ (GMV)')
    plt.plot(l, inv_resp_gmv_A * (l*(l+1))**2/4, color='seagreen', linestyle='-', label='$1/R^{SS}$ (GMV [TT, EE, TE])')
    ##plt.plot(l, inv_resp_gmv_B * (l*(l+1))**2/4, color='orchid', linestyle='-', label='$1/R^{SS}$ (GMV [TB, EB])')
    ##plt.plot(l, inv_resp_gmv_sk * (l*(l+1))**2/4, color='palegoldenrod', linestyle='--', label='$1/R^{SK}$ (GMV)')
    plt.plot(l, -1*inv_resp_gmv_A_sk * (l*(l+1))**2/4, color='lightgreen', linestyle='--', label='-1 x $1/R^{SK}$ (GMV [TT, EE, TE])')
    plt.plot(l, inv_resp_gmv_A_kk * (l*(l+1))**2/4, color='darkgreen', linestyle=':', label='$1/R^{KK}$ (GMV [TT, EE, TE])')
    ##plt.plot(l, inv_resp_gmv_B_sk * (l*(l+1))**2/4, color='plum', linestyle='--', label='$1/R^{SK}$ (GMV [TB, EB])')
    plt.plot(l, inv_resp_gmv_A_ftt_only * (l*(l+1))**2/4, color='tan', linestyle='-', label='$1/R^{SS}$ (GMV [TT])')
    plt.plot(l, -1*inv_resp_gmv_A_sk_ftt_only * (l*(l+1))**2/4, color='palegoldenrod', linestyle='--', label='-1 x $1/R^{SK}$ (GMV [TT])')
    plt.plot(l, inv_resp_gmv_A_kk_ftt_only * (l*(l+1))**2/4, color='darkgoldenrod', linestyle=':', label='$1/R^{KK}$ (GMV [TT])')
    plt.ylabel("$C_\ell^{\kappa\kappa}$")
    plt.xlabel('$\ell$')
    plt.title('$1/R$')
    plt.legend(loc='upper left', fontsize='small')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,lmax)
    #plt.ylim(8e-9,1e-5)
    if save_fig:
        plt.savefig(dir_out+f'/figs/profile_response_comparison.png')
        #plt.savefig(dir_out+f'/figs/profile_response_comparison_noiseless.png')

    plt.figure(1)
    plt.clf()
    #plt.plot(l, np.abs(inv_resp_gmv_A/inv_resp_healqest_TT - 1)*100, color='seagreen', linestyle='-', label='$R^{SS}$ Comparison')
    #plt.plot(l, np.abs(-1*inv_resp_gmv_A_sk/inv_resp_healqest_TT_sk - 1)*100, color='lightgreen', linestyle='--', label='$R^{SK}$ Comparison (with x-1)')
    #plt.plot(l, np.abs(inv_resp_gmv_A_kk/inv_resp_healqest_TT_kk - 1)*100, color='darkgreen', linestyle=':', label='$R^{KK}$ Comparison')
    #plt.plot(l, np.abs(inv_resp_gmv_A_ftt_only/inv_resp_healqest_TT - 1)*100, color='tan', linestyle='-', label='$R^{SS}$ Comparison (GMV TT only)')
    #plt.plot(l, np.abs(-1*inv_resp_gmv_A_sk_ftt_only/inv_resp_healqest_TT_sk - 1)*100, color='palegoldenrod', linestyle='--', label='$R^{SK}$ Comparison (with x-1, GMV TT only)')
    #plt.plot(l, np.abs(inv_resp_gmv_A_kk_ftt_only/inv_resp_healqest_TT_kk - 1)*100, color='darkgoldenrod', linestyle=':', label='$R^{KK}$ Comparison (GMV TT only)')
    plt.plot(l, np.abs(inv_resp_gmv_A/inv_resp_gmv_A_ftt_only - 1)*100, color='seagreen', linestyle='-', label='$R^{SS}$ Comparison')
    plt.plot(l, np.abs(inv_resp_gmv_A_sk/inv_resp_gmv_A_sk_ftt_only - 1)*100, color='lightgreen', linestyle='--', label='$R^{SK}$ Comparison')
    plt.plot(l, np.abs(inv_resp_gmv_A_kk/inv_resp_gmv_A_kk_ftt_only - 1)*100, color='darkgreen', linestyle=':', label='$R^{KK}$ Comparison')
    plt.xlabel('$\ell$')
    #plt.title("$|{R_{healqest}}/{R_{GMV,A}} - 1|$ x 100%")
    plt.title("$|{R_{GMV [TT]}}/{R_{GMV [TT, EE, TE]}} - 1|$ x 100%")
    plt.legend(loc='upper right', fontsize='small')
    plt.xlim(10,lmax)
    plt.ylim(0,30)
    #plt.ylim(0,50)
    if save_fig:
        #plt.savefig(dir_out+f'/figs/profile_response_comparison_frac_diff.png')
        #plt.savefig(dir_out+f'/figs/profile_response_comparison_frac_diff_noiseless.png')
        plt.savefig(dir_out+f'/figs/profile_response_comparison_frac_diff_gmv_with_without_zeroing.png')

def get_analytic_response(est, config, lmax=4096, fwhm=0, nlev_t=0, nlev_p=0, u=None,
                          clfile='/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat',
                          filename=None, qeZA=None):
    if filename is None:
        filename = '/scratch/users/yukanaka/gmv/resp/an_resp_{}_healqest_lmax{}_fwhm{}_nlevt{}_nlevp{}.npy'.format(est,lmax,fwhm,nlev_t,nlev_p)
    if os.path.isfile(filename):
        R = np.load(filename)
    else:
        ell,sltt,slee,slbb,slte = utils.get_lensedcls(clfile,lmax=lmax)
        bl = hp.gauss_beam(fwhm=fwhm*0.00029088,lmax=lmax)
        nltt = (np.pi/180./60.*nlev_t)**2 / bl**2
        nlee=nlbb = (np.pi/180./60.*nlev_p)**2 / bl**2
        # Signal + noise spectra
        cltt = sltt + nltt
        clee = slee + nlee
        clbb = slbb + nlbb
        # Create 1/Nl filters
        flt = np.zeros(lmax+1); flt[100:] = 1./cltt[100:]
        fle = np.zeros(lmax+1); fle[100:] = 1./clee[100:]
        flb = np.zeros(lmax+1); flb[100:] = 1./clbb[100:]
        # Define qest from quicklens (commented out for Python3)
        if est == 'TT' or est == 'TTprf':
            flX = flt
            flY = flt
        elif est == 'EE':
            flX = fle
            flY = fle
        elif est == 'TE':
            flX = flt
            flY = fle
        elif est == 'TB':
            flX = flt
            flY = flb
        elif est == 'BT':
            flX = flb
            flY = flt
        elif est == 'EB':
            flX = fle
            flY = flb
        elif est == 'ET':
            flX = fle
            flY = flt
        elif est == 'BE':
            flX = flb
            flY = fle
        R = resp.fill_resp(weights_combined_qestobj.weights(est,lmax,config,cltype='len',u=u), np.zeros(lmax+1, dtype=np.complex_), flX, flY, qeZA=qeZA)
        np.save(filename, R)
    return R

def alm_cutlmax(almin,new_lmax):
    '''
    Get a new alm with a smaller lmax.
    Note that in an alm array, values where m > l are left out, because they are zero.
    '''
    # getidx takes args (old) lmax, l, m and returns an array of indices for new alm
    lmmap = hp.Alm.getidx(hp.Alm.getlmax(np.shape(almin)[-1]),
                          *hp.Alm.getlm(new_lmax,np.arange(hp.Alm.getsize(new_lmax))))
    nreal = np.shape(almin)[0]

    if nreal <= 3:
        # Case if almin is a list of T, E and B alms and not just a single alm
        almout = np.zeros((nreal,hp.Alm.getsize(new_lmax)),dtype=np.complex_)
        for i in range(nreal):
            almout[i] = almin[i][lmmap]
    else:
        almout = np.zeros(hp.Alm.getsize(new_lmax),dtype=np.complex_)
        almout = almin[lmmap]

    return almout

####################

#compare_profile_hardening_resp()
#compare_lensing_resp()
#compare_profile_hardening()
compare_gmv_specs()
