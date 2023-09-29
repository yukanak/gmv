#!/usr/bin/env python3
import numpy as np
import healpy as hp
import camb
import os, sys
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
from astropy.io import fits
import utils
import matplotlib.pyplot as plt
import weights_combined
import qest
import wignerd
import resp

def compare_profile_hardening_resp(u=None,lmax=4096,nside=8192,dir_out='/scratch/users/yukanaka/gmv/',save_fig=True):
    clfile = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat'
    l = np.arange(0,lmax+1)
    if u is None:
        u = hp.sphtfunc.gauss_beam(1*(np.pi/180.)/60., lmax=lmax)

    # Flat sky healqest response
    resp_healqest_TT = get_analytic_response('TTprf',lmax,fwhm=1,nlev_t=5,nlev_p=5,u=u,clfile=clfile,unl=False)
    inv_resp_healqest_TT = np.zeros_like(l,dtype=np.complex_); inv_resp_healqest_TT[1:] = 1/(resp_healqest_TT)[1:]
    resp_healqest_TT_sk = get_analytic_response('TTprf',lmax,fwhm=1,nlev_t=5,nlev_p=5,u=u,clfile=clfile,unl=False,
                                                filename='/scratch/users/yukanaka/gmv/resp/an_resp_TTprf_healqest_lmax{}_fwhm1_nlevt5_nlevp5_SK_TT.npy'.format(lmax),
                                                qeZA=weights_combined.weights('TT',lmax,clfile))
    inv_resp_healqest_TT_sk = np.zeros_like(l,dtype=np.complex_); inv_resp_healqest_TT_sk[1:] = 1/(resp_healqest_TT_sk)[1:]

    # GMV response
    gmv_resp_data = np.genfromtxt('True_variance_individual_custom_lmin0.0_lmaxT4096_lmaxP4096_beam1_noise5_50_PRF.txt')
    inv_resp_gmv = gmv_resp_data[:,3] / l**2
    inv_resp_gmv_A = gmv_resp_data[:,1] / l**2
    inv_resp_gmv_B = gmv_resp_data[:,2] / l**2
    gmv_resp_data_sk = np.genfromtxt('True_variance_individual_custom_lmin0.0_lmaxT4096_lmaxP4096_beam1_noise5_50_PRF_SK.txt')
    inv_resp_gmv_sk = gmv_resp_data_sk[:,3] / l**2
    inv_resp_gmv_A_sk = gmv_resp_data_sk[:,1] / l**2
    inv_resp_gmv_B_sk = gmv_resp_data_sk[:,2] / l**2
    gmv_resp_data_ftt_only = np.genfromtxt('True_variance_individual_custom_lmin0.0_lmaxT4096_lmaxP4096_beam1_noise5_50_PRF_SS_FTT_only.txt')
    inv_resp_gmv_ftt_only = gmv_resp_data[:,3] / l**2
    inv_resp_gmv_A_ftt_only = gmv_resp_data[:,1] / l**2
    inv_resp_gmv_B_ftt_only = gmv_resp_data[:,2] / l**2
    gmv_resp_data_sk_ftt_only = np.genfromtxt('True_variance_individual_custom_lmin0.0_lmaxT4096_lmaxP4096_beam1_noise5_50_PRF_SK_FTT_only.txt')
    inv_resp_gmv_sk_ftt_only = gmv_resp_data_sk[:,3] / l**2
    inv_resp_gmv_A_sk_ftt_only = gmv_resp_data_sk[:,1] / l**2
    inv_resp_gmv_B_sk_ftt_only = gmv_resp_data_sk[:,2] / l**2

    # Theory spectrum
    clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
    ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
    clkk = slpp * (l*(l+1))**2/4

    plt.figure(0)
    plt.clf()
    plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')
    plt.plot(l, inv_resp_healqest_TT * (l*(l+1))**2/4, color='firebrick', linestyle='-', label='$1/R^{SS}$ (Healqest TT)')
    plt.plot(l, inv_resp_healqest_TT_sk * (l*(l+1))**2/4, color='lightcoral', linestyle='--', label='$1/R^{SK}$ (Healqest TT)')
    #plt.plot(l, inv_resp_gmv * (l*(l+1))**2/4, color='darkkhaki', linestyle='-', label='$1/R^{SS}$ (GMV)')
    plt.plot(l, inv_resp_gmv_A * (l*(l+1))**2/4, color='seagreen', linestyle='-', label='$1/R^{SS}$ (GMV [TT, EE, TE])')
    #plt.plot(l, inv_resp_gmv_B * (l*(l+1))**2/4, color='orchid', linestyle='-', label='$1/R^{SS}$ (GMV [TB, EB])')
    #plt.plot(l, inv_resp_gmv_sk * (l*(l+1))**2/4, color='palegoldenrod', linestyle='--', label='$1/R^{SK}$ (GMV)')
    plt.plot(l, -1*inv_resp_gmv_A_sk * (l*(l+1))**2/4, color='lightgreen', linestyle='--', label='-1 x $1/R^{SK}$ (GMV [TT, EE, TE])')
    #plt.plot(l, inv_resp_gmv_B_sk * (l*(l+1))**2/4, color='plum', linestyle='--', label='$1/R^{SK}$ (GMV [TB, EB])')
    plt.plot(l, inv_resp_gmv_A_ftt_only * (l*(l+1))**2/4, color='tan', linestyle='-', label='$1/R^{SS}$ (GMV [TT])')
    plt.plot(l, -1*inv_resp_gmv_A_sk_ftt_only * (l*(l+1))**2/4, color='palegoldenrod', linestyle='--', label='-1 x $1/R^{SK}$ (GMV [TT])')
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

    plt.figure(1)
    plt.clf()
    plt.plot(l, np.abs(inv_resp_gmv_A/inv_resp_healqest_TT - 1)*100, color='seagreen', linestyle='-', label='$R^{SS}$ Comparison')
    plt.plot(l, np.abs(-1*inv_resp_gmv_A_sk/inv_resp_healqest_TT_sk - 1)*100, color='lightgreen', linestyle='--', label='$R^{SK}$ Comparison (with x-1)')
    plt.plot(l, np.abs(inv_resp_gmv_A_ftt_only/inv_resp_healqest_TT - 1)*100, color='tan', linestyle='-', label='$R^{SS}$ Comparison (GMV TT only)')
    plt.plot(l, np.abs(-1*inv_resp_gmv_A_sk_ftt_only/inv_resp_healqest_TT_sk - 1)*100, color='palegoldenrod', linestyle='--', label='$R^{SK}$ Comparison (with x-1, GMV TT only)')
    plt.xlabel('$\ell$')
    plt.title("$|{R_{healqest}}/{R_{GMV,A}} - 1|$ x 100%")
    plt.legend(loc='upper right', fontsize='small')
    plt.xlim(10,lmax)
    plt.ylim(0,30)
    if save_fig:
        plt.savefig(dir_out+f'/figs/profile_response_comparison_frac_diff.png')

def compare_gmv_len(sim=2,lmax=4096,nside=8192,dir_out='/scratch/users/yukanaka/gmv/',save_fig=True):
    unl = False
    l = np.arange(0,lmax+1)
    ests = ['TT', 'EE', 'TE', 'TB', 'EB']

    # Load plms
    clfile = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat'
    plm_gmv = np.load(dir_out+f'/output/plm_healqest_seed{sim}_lmax{lmax}_nside{nside}_qest_gmv_withBTBE.npy')
    plm_gmv_A = np.load(dir_out+f'/output/plm_healqest_seed{sim}_lmax{lmax}_nside{nside}_qest_gmv_withBTBE_A.npy')
    plm_gmv_B = np.load(dir_out+f'/output/plm_healqest_seed{sim}_lmax{lmax}_nside{nside}_qest_gmv_withBTBE_B.npy')
    #plm_gmv_noBTBE = np.load(dir_out+f'/output/plm_healqest_seed{sim}_lmax{lmax}_nside{nside}_qest_gmv.npy')
    #plm_gmv = np.load(dir_out+f'/output/plm_healqest_seed{sim}_lmax{lmax}_nside{nside}_qest_gmv_yuuki.npy')
    #plm_gmv_A = np.load(dir_out+f'/output/plm_healqest_seed{sim}_lmax{lmax}_nside{nside}_qest_gmv_A_yuuki.npy')
    #plm_gmv_B = np.load(dir_out+f'/output/plm_healqest_seed{sim}_lmax{lmax}_nside{nside}_qest_gmv_B_yuuki.npy')
    #plm_gmv = np.load(dir_out+f'/output/plm_healqest_seed{sim}_lmax{lmax}_nside{nside}_qest_gmv_yuuki_no_BTBE.npy')
    #plm_gmv_A = np.load(dir_out+f'/output/plm_healqest_seed{sim}_lmax{lmax}_nside{nside}_qest_gmv_A_yuuki_no_BTBE.npy')
    #plm_gmv_B = np.load(dir_out+f'/output/plm_healqest_seed{sim}_lmax{lmax}_nside{nside}_qest_gmv_B_yuuki_no_BTBE.npy')

    plm_original = np.zeros(len(plm_gmv), dtype=np.complex_)
    for i, est in enumerate(ests):
        plm_original += np.load(dir_out+f'/output/plm_{est}_healqest_seed{sim}_lmax{lmax}_nside{nside}_qest_original.npy')
    plm_original_TT = np.load(dir_out+f'/output/plm_TT_healqest_seed{sim}_lmax{lmax}_nside8192_qest_original.npy')
    plm_original_TE = np.load(dir_out+f'/output/plm_TE_healqest_seed{sim}_lmax{lmax}_nside8192_qest_original.npy')
    plm_original_EE = np.load(dir_out+f'/output/plm_EE_healqest_seed{sim}_lmax{lmax}_nside8192_qest_original.npy')
    plm_original_TB = np.load(dir_out+f'/output/plm_TB_healqest_seed{sim}_lmax{lmax}_nside8192_qest_original.npy')
    plm_original_EB = np.load(dir_out+f'/output/plm_EB_healqest_seed{sim}_lmax{lmax}_nside8192_qest_original.npy')

    resp_original_TT = get_analytic_response('TT',lmax,fwhm=1,nlev_t=5,nlev_p=5,clfile=clfile,unl=False)
    resp_original_TE = get_analytic_response('TE',lmax,fwhm=1,nlev_t=5,nlev_p=5,clfile=clfile,unl=False)# * 0.5
    resp_original_EE = get_analytic_response('EE',lmax,fwhm=1,nlev_t=5,nlev_p=5,clfile=clfile,unl=False)
    resp_original_TB = get_analytic_response('TB',lmax,fwhm=1,nlev_t=5,nlev_p=5,clfile=clfile,unl=False)# * 0.5
    resp_original_EB = get_analytic_response('EB',lmax,fwhm=1,nlev_t=5,nlev_p=5,clfile=clfile,unl=False)# * 0.5
    resp_original = resp_original_TT + resp_original_TE + resp_original_EE + resp_original_TB + resp_original_EB
    inv_resp_original_TT = np.zeros_like(l,dtype=np.complex_); inv_resp_original_TT[1:] = 1/(resp_original_TT)[1:]
    inv_resp_original_TE = np.zeros_like(l,dtype=np.complex_); inv_resp_original_TE[1:] = 1/(resp_original_TE)[1:]
    inv_resp_original_EE = np.zeros_like(l,dtype=np.complex_); inv_resp_original_EE[1:] = 1/(resp_original_EE)[1:]
    inv_resp_original_TB = np.zeros_like(l,dtype=np.complex_); inv_resp_original_TB[1:] = 1/(resp_original_TB)[1:]
    inv_resp_original_EB = np.zeros_like(l,dtype=np.complex_); inv_resp_original_EB[1:] = 1/(resp_original_EB)[1:]
    inv_resp_original = np.zeros_like(l,dtype=np.complex_); inv_resp_original[1:] = 1/(resp_original)[1:]
    # Response correct
    plm_original_resp_corr = hp.almxfl(plm_original,inv_resp_original)
    plm_TT_resp_corr = hp.almxfl(plm_original_TT,inv_resp_original_TT)
    plm_TE_resp_corr = hp.almxfl(plm_original_TE,inv_resp_original_TE)
    plm_EE_resp_corr = hp.almxfl(plm_original_EE,inv_resp_original_EE)
    plm_TB_resp_corr = hp.almxfl(plm_original_TB,inv_resp_original_TB)
    plm_EB_resp_corr = hp.almxfl(plm_original_EB,inv_resp_original_EB)

    gmv_resp_data = np.genfromtxt('True_variance_individual_custom_lmin0.0_lmaxT4096_lmaxP4096_beam1_noise5_50.txt')
    # Abhi's code calculates the reconstruction noise for d field rather than phi field, see GMV_QE.py, line 292 for example
    inv_resp_gmv = gmv_resp_data[:,3] / l**2
    inv_resp_gmv_A = gmv_resp_data[:,1] / l**2
    inv_resp_gmv_B = gmv_resp_data[:,2] / l**2
    # N is 1/R
    plm_gmv_resp_corr = hp.almxfl(plm_gmv,inv_resp_gmv)
    plm_gmv_A_resp_corr = hp.almxfl(plm_gmv_A,inv_resp_gmv_A)
    plm_gmv_B_resp_corr = hp.almxfl(plm_gmv_B,inv_resp_gmv_B)
    #plm_gmv_resp_corr_noBTBE = hp.almxfl(plm_gmv_noBTBE,inv_resp_gmv)

    # Get spectra
    auto_gmv = hp.alm2cl(plm_gmv_resp_corr, plm_gmv_resp_corr, lmax=lmax) * (l*(l+1))**2/4
    auto_gmv_A = hp.alm2cl(plm_gmv_A_resp_corr, plm_gmv_A_resp_corr, lmax=lmax) * (l*(l+1))**2/4
    auto_gmv_B = hp.alm2cl(plm_gmv_B_resp_corr, plm_gmv_B_resp_corr, lmax=lmax) * (l*(l+1))**2/4
    auto_original = hp.alm2cl(plm_original_resp_corr, plm_original_resp_corr, lmax=lmax) * (l*(l+1))**2/4
    #auto_gmv_noBTBE = hp.alm2cl(plm_gmv_resp_corr_noBTBE, plm_gmv_resp_corr_noBTBE, lmax=lmax) * (l*(l+1))**2/4
    auto_TT = hp.alm2cl(plm_TT_resp_corr, plm_TT_resp_corr) * (l*(l+1))**2/4
    auto_TE = hp.alm2cl(plm_TE_resp_corr, plm_TE_resp_corr) * (l*(l+1))**2/4
    auto_EE = hp.alm2cl(plm_EE_resp_corr, plm_EE_resp_corr) * (l*(l+1))**2/4
    auto_TB = hp.alm2cl(plm_TB_resp_corr, plm_TB_resp_corr) * (l*(l+1))**2/4
    auto_EB = hp.alm2cl(plm_EB_resp_corr, plm_EB_resp_corr) * (l*(l+1))**2/4

    # Theory spectrum
    clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
    ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
    clkk = slpp * (l*(l+1))**2/4

    # Get N0 bias
    #n0, n0_A, n0_B = get_n0(gmv=True, with_BTBE=True)
    #n0_original, n0_TT, n0_TE, n0_EE, n0_TB, n0_EB = get_n0(gmv=False)

    # Cross correlate with input plm
    input_plm = hp.read_alm(f'/scratch/users/yukanaka/full_res_maps/phi/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_phi1_seed{sim}.alm')
    input_plm = utils.reduce_lmax(input_plm,lmax=lmax)
    cross_gmv = hp.alm2cl(input_plm, plm_gmv_resp_corr, lmax=lmax) * (l*(l+1))**2/4
    cross_original = hp.alm2cl(input_plm, plm_original_resp_corr, lmax=lmax) * (l*(l+1))**2/4
    auto_input = hp.alm2cl(input_plm, input_plm, lmax=lmax) * (l*(l+1))**2/4
    cross_original_TT = hp.alm2cl(input_plm, plm_TT_resp_corr, lmax=lmax) * (l*(l+1))**2/4
    cross_original_TE = hp.alm2cl(input_plm, plm_TE_resp_corr, lmax=lmax) * (l*(l+1))**2/4
    cross_original_EE = hp.alm2cl(input_plm, plm_EE_resp_corr, lmax=lmax) * (l*(l+1))**2/4
    cross_original_TB = hp.alm2cl(input_plm, plm_TB_resp_corr, lmax=lmax) * (l*(l+1))**2/4
    cross_original_EB = hp.alm2cl(input_plm, plm_EB_resp_corr, lmax=lmax) * (l*(l+1))**2/4

    # Plot
    plt.figure(0)
    plt.clf()
    plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')
    plt.plot(l, auto_gmv, color='peru', label="Auto Spectrum (GMV)")
    #plt.plot(l, auto_gmv_A, color='forestgreen', label=f'Auto Spectrum (GMV [TT, EE, TE])')
    #plt.plot(l, auto_gmv_B, color='mediumorchid', label="Auto Spectrum (GMV [TB, EB])")
    #plt.plot(l, auto_gmv_noBTBE, color='slategrey', label="Auto Spectrum (GMV, no BT/BE)")
    plt.plot(l, auto_original, color='darkblue', label="Auto Spectrum (Original Total)")
    plt.plot(l, auto_TT, color='firebrick', label="Auto Spectrum (Original TT)")
    plt.plot(l, 2 * auto_TE, color='forestgreen', label="2 x Auto Spectrum (Original TE)")
    plt.plot(l, auto_EE, color='mediumorchid', label="Auto Spectrum (Original EE)")
    plt.plot(l, 2 * auto_TB, color='gold', label="2 x Auto Spectrum (Original TB)")
    plt.plot(l, 2 * auto_EB, color='orange', label="2 x Auto Spectrum (Original EB)")
    #plt.plot(l, auto_input, color='slategrey', label="Auto Spectrum (Input)")
    #plt.plot(l, cross_gmv, color='peru', label='Cross Spectrum (Input x GMV)')
    #plt.plot(l, cross_original, color='darkblue', label='Cross Spectrum (Input x Original Total)')
    #plt.plot(l, cross_original_TT, color='firebrick', label='Cross Spectrum (Input x Original TT)')
    #plt.plot(l, cross_original_TE, color='forestgreen', label='Cross Spectrum (Input x Original TE)')
    #plt.plot(l, cross_original_EE, color='mediumorchid', label='Cross Spectrum (Input x Original EE)')
    #plt.plot(l, cross_original_TB, color='gold', label='Cross Spectrum (Input x Original TB)')
    #plt.plot(l, cross_original_EB, color='orange', label='Cross Spectrum (Input x Original EB)')
    #plt.plot(l, auto_original - inv_resp_original * (l*(l+1))**2/4, color='darkblue', label="Auto Spectrum - 1/R (Original)")
    #plt.plot(l, auto_gmv - inv_resp_gmv * (l*(l+1))**2/4, color='firebrick', label="Auto Spectrum - 1/R (GMV)")
    #plt.plot(l, auto_gmv_A - inv_resp_gmv_A * (l*(l+1))**2/4, color='forestgreen', label=f'Auto Spectrum - 1/R (GMV [TT, EE, TE])')
    #plt.plot(l, auto_gmv_B - inv_resp_gmv_B * (l*(l+1))**2/4, color='mediumorchid', label="Auto Spectrum - 1/R (GMV [TB, EB])")
    #plt.plot(l, auto_original - n0_original * (l*(l+1))**2/4, color='darkblue', label="Auto Spectrum - $N_0$ (Original)")
    #plt.plot(l, auto_gmv - n0 * (l*(l+1))**2/4, color='firebrick', label="Auto Spectrum - $N_0$ (GMV)")
    #plt.plot(l, auto_gmv_A - n0_A * (l*(l+1))**2/4, color='forestgreen', label=f'Auto Spectrum - $N_0$ (GMV [TT, EE, TE])')
    #plt.plot(l, auto_gmv_B - n0_B * (l*(l+1))**2/4, color='mediumorchid', label="Auto Spectrum - $N_0$ (GMV [TB, EB])")
    plt.plot(l, inv_resp_gmv * (l*(l+1))**2/4, color='sandybrown', linestyle='--', label='1/R (GMV)')
    plt.plot(l, inv_resp_original * (l*(l+1))**2/4, color='cornflowerblue', linestyle='--', label='1/R (Original Total)')
    plt.plot(l, inv_resp_original_TT * (l*(l+1))**2/4, color='lightcoral', linestyle='--', label='1/R (Original TT)')
    plt.plot(l, inv_resp_original_TE * (l*(l+1))**2/4, color='lightgreen', linestyle='--', label='1/R (Original TE)')
    plt.plot(l, inv_resp_original_EE * (l*(l+1))**2/4, color='plum', linestyle='--', label='1/R (Original EE)')
    plt.plot(l, inv_resp_original_TB * (l*(l+1))**2/4, color='palegoldenrod', linestyle='--', label='1/R (Original TB)')
    plt.plot(l, inv_resp_original_EB * (l*(l+1))**2/4, color='bisque', linestyle='--', label='1/R (Original EB)')
    #plt.plot(l, inv_resp_gmv_A * (l*(l+1))**2/4, color='lightgreen', linestyle='--', label='1/R (GMV [TT, EE, TE])')
    #plt.plot(l, inv_resp_gmv_B * (l*(l+1))**2/4, color='plum', linestyle='--', label='1/R (GMV [TB, EB])')
    #plt.plot(l, inv_resp_gmv/inv_resp_original - 1, color='lightcoral', linestyle='--', label='1/R (GMV) / 1/R (Original) - 1')
    plt.ylabel("$C_\ell^{\kappa\kappa}$")
    plt.xlabel('$\ell$')
    plt.title(f'Spectra with Response Correction for Sim {sim}')
    #plt.title('1/R with 5 uK-arcmin Noise Levels for T/P')
    plt.legend(loc='upper right', fontsize='small')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,lmax)
    #plt.ylim(8e-9,1e-6)
    plt.ylim(8e-9,1e-5)
    #plt.ylim(-0.3,0.3)
    if save_fig:
        #plt.savefig(dir_out+f'/figs/gmv_comparison_spec_with_resp_len_cross.png')
        plt.savefig(dir_out+f'/figs/gmv_comparison_spec_with_resp_len_auto.png')
        #plt.savefig(dir_out+f'/figs/gmv_comparison_spec_with_resp_dominic.png')
        #plt.savefig(dir_out+f'/figs/gmv_comparison_spec_with_resp_1_over_R_subtracted.png')
        #plt.savefig(dir_out+f'/figs/gmv_comparison_spec_with_resp_N_0_subtracted.png')
    #plt.show()

def compare_gmv_unl(sims=np.arange(10)+1,lmax=4096,nside=8192,dir_out='/scratch/users/yukanaka/gmv/',save_fig=True):
    '''
    Plot N0 bias.
    '''
    unl = True
    l = np.arange(0,lmax+1)
    ests = ['TT', 'EE', 'TE', 'TB', 'EB']
    clfile = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat'

    # Load plms
    #clfile = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
    #clfile = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat'
    #plm_gmv = np.load(dir_out+f'/output/plm_healqest_seed{sim}_lmax{lmax}_nside{nside}_qest_gmv_unl_from_lensed_cls_withBTBE.npy')
    #plm_gmv_A = np.load(dir_out+f'/output/plm_healqest_seed{sim}_lmax{lmax}_nside{nside}_qest_gmv_unl_from_lensed_cls_withBTBE_A.npy')
    #plm_gmv_B = np.load(dir_out+f'/output/plm_healqest_seed{sim}_lmax{lmax}_nside{nside}_qest_gmv_unl_from_lensed_cls_withBTBE_B.npy')
    #plm_original = np.zeros(len(plm_gmv), dtype=np.complex_)
    #for i, est in enumerate(ests):
    #    plm_original += np.load(dir_out+f'/output/plm_{est}_healqest_seed{sim}_lmax{lmax}_nside{nside}_qest_original_unl_from_lensed_cls.npy')

    # Get N0 bias
    n0, n0_A, n0_B = get_n0(gmv=True, with_BTBE=True)
    n0_original, n0_TT, n0_TE, n0_EE, n0_TB, n0_EB = get_n0(gmv=False)

    # Response #TODO??
    resp_original_TT = get_analytic_response('TT',lmax,fwhm=1,nlev_t=5,nlev_p=5,clfile=clfile,unl=False)
    resp_original_TE = get_analytic_response('TE',lmax,fwhm=1,nlev_t=5,nlev_p=5,clfile=clfile,unl=False)
    resp_original_EE = get_analytic_response('EE',lmax,fwhm=1,nlev_t=5,nlev_p=5,clfile=clfile,unl=False)
    resp_original_TB = get_analytic_response('TB',lmax,fwhm=1,nlev_t=5,nlev_p=5,clfile=clfile,unl=False)
    resp_original_EB = get_analytic_response('EB',lmax,fwhm=1,nlev_t=5,nlev_p=5,clfile=clfile,unl=False)
    resp_original_ET = get_analytic_response('ET',lmax,fwhm=1,nlev_t=5,nlev_p=5,clfile=clfile,unl=False)
    resp_original_BT = get_analytic_response('BT',lmax,fwhm=1,nlev_t=5,nlev_p=5,clfile=clfile,unl=False)
    resp_original_BE = get_analytic_response('BE',lmax,fwhm=1,nlev_t=5,nlev_p=5,clfile=clfile,unl=False)
    #resp_original = resp_original_TT + resp_original_TE + resp_original_EE + resp_original_TB + resp_original_EB + resp_original_ET + resp_original_BT + resp_original_BE
    resp_original = resp_original_TT + resp_original_TE + resp_original_EE + resp_original_TB + resp_original_EB
    inv_resp_original_TT = np.zeros_like(l,dtype=np.complex_); inv_resp_original_TT[1:] = 1/(resp_original_TT)[1:]
    inv_resp_original_TE = np.zeros_like(l,dtype=np.complex_); inv_resp_original_TE[1:] = 1/(resp_original_TE)[1:]
    inv_resp_original_EE = np.zeros_like(l,dtype=np.complex_); inv_resp_original_EE[1:] = 1/(resp_original_EE)[1:]
    inv_resp_original_TB = np.zeros_like(l,dtype=np.complex_); inv_resp_original_TB[1:] = 1/(resp_original_TB)[1:]
    inv_resp_original_EB = np.zeros_like(l,dtype=np.complex_); inv_resp_original_EB[1:] = 1/(resp_original_EB)[1:]
    inv_resp_original = np.zeros_like(l,dtype=np.complex_); inv_resp_original[1:] = 1/(resp_original)[1:]

    gmv_resp_data = np.genfromtxt('True_variance_individual_custom_lmin0.0_lmaxT4096_lmaxP4096_beam1_noise5_50.txt')
    # Abhi's code calculates the reconstruction noise for d field rather than phi field, see GMV_QE.py, line 292 for example
    inv_resp_gmv = gmv_resp_data[:,3] / l**2
    inv_resp_gmv_A = gmv_resp_data[:,1] / l**2
    inv_resp_gmv_B = gmv_resp_data[:,2] / l**2

    # Theory spectrum
    clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
    ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
    clkk = slpp * (l*(l+1))**2/4

    # Plot
    plt.figure(0)
    plt.clf()
    plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')
    #plt.plot(l, n0_original * (l*(l+1))**2/4, color='darkblue', linestyle='-', label='$N_0$ from 10 Unlensed Sims (Original)')
    #plt.plot(l, n0_TT * (l*(l+1))**2/4, color='firebrick', linestyle='-', label='$N_0$ from 10 Unlensed Sims (TT)')
    #plt.plot(l, 2 * n0_TE * (l*(l+1))**2/4, color='forestgreen', linestyle='-', label='2 x $N_0$ from 10 Unlensed Sims (TE)')
    #plt.plot(l, n0_EE * (l*(l+1))**2/4, color='mediumorchid', linestyle='-', label='$N_0$ from 10 Unlensed Sims (EE)')
    #plt.plot(l, 2 * n0_TB * (l*(l+1))**2/4, color='gold', linestyle='-', label='2 x $N_0$ from 10 Unlensed Sims (TB)')
    #plt.plot(l, 2 * n0_EB * (l*(l+1))**2/4, color='orange', linestyle='-', label='2 x $N_0$ from 10 Unlensed Sims (EB)')
    plt.plot(l, n0_TT * (l*(l+1))**2/4, color='firebrick', linestyle='-', label='$N_0$ (TT)')
    plt.plot(l, 2 * n0_TE * (l*(l+1))**2/4, color='forestgreen', linestyle='-', label='$N_0$ (TE)')
    plt.plot(l, n0_EE * (l*(l+1))**2/4, color='mediumorchid', linestyle='-', label='$N_0$ (EE)')
    plt.plot(l, 2 * n0_TB * (l*(l+1))**2/4, color='gold', linestyle='-', label='$N_0$ (TB)')
    plt.plot(l, 2 * n0_EB * (l*(l+1))**2/4, color='orange', linestyle='-', label='$N_0$ (EB)')
    #plt.plot(l, n0 * (l*(l+1))**2/4, color='firebrick', linestyle='-', label='$N_0$ from 10 Unlensed Sims (GMV)')
    #plt.plot(l, n0_A * (l*(l+1))**2/4, color='forestgreen', linestyle='-', label='$N_0$ from 10 Unlensed Sims (GMV [TT, EE, TE])')
    #plt.plot(l, n0_B * (l*(l+1))**2/4, color='mediumorchid', linestyle='-', label='$N_0$ from 10 Unlensed Sims (GMV [TB, EB])')
    #plt.plot(l, inv_resp_original * (l*(l+1))**2/4, color='cornflowerblue', linestyle='--', label='1/R (Original)')
    #plt.plot(l, inv_resp_original_TT * (l*(l+1))**2/4, color='lightcoral', linestyle='--', label='1/R (TT)')
    #plt.plot(l, inv_resp_original_TE * (l*(l+1))**2/4, color='lightgreen', linestyle='--', label='1/R (TE)')
    #plt.plot(l, inv_resp_original_EE * (l*(l+1))**2/4, color='plum', linestyle='--', label='1/R (EE)')
    #plt.plot(l, inv_resp_original_TB * (l*(l+1))**2/4, color='palegoldenrod', linestyle='--', label='1/R (TB)')
    #plt.plot(l, inv_resp_original_EB * (l*(l+1))**2/4, color='bisque', linestyle='--', label='1/R (EB)')
    plt.plot(l, inv_resp_original_TT * (l*(l+1))**2/4, color='lightcoral', linestyle='--', label='1/R (TT)')
    plt.plot(l, inv_resp_original_TE * (l*(l+1))**2/4, color='lightgreen', linestyle='--', label='1/R (TE)')
    plt.plot(l, inv_resp_original_EE * (l*(l+1))**2/4, color='plum', linestyle='--', label='1/R (EE)')
    plt.plot(l, inv_resp_original_TB * (l*(l+1))**2/4, color='palegoldenrod', linestyle='--', label='1/R (TB)')
    plt.plot(l, inv_resp_original_EB * (l*(l+1))**2/4, color='bisque', linestyle='--', label='1/R (EB)')
    #plt.plot(l, inv_resp_gmv * (l*(l+1))**2/4, color='lightcoral', linestyle='--', label='1/R (GMV)')
    #plt.plot(l, inv_resp_gmv_A * (l*(l+1))**2/4, color='lightgreen', linestyle='--', label='1/R (GMV [TT, EE, TE])')
    #plt.plot(l, inv_resp_gmv_B * (l*(l+1))**2/4, color='plum', linestyle='--', label='1/R (GMV [TB, EB])')
    plt.ylabel("$C_\ell^{\kappa\kappa}$")
    plt.xlabel('$\ell$')
    #plt.title('$N_0$ with Response Correction')
    plt.title('$N_0$ from 10 Unlensed Sims, with Response Correction')
    plt.legend(loc='upper right', fontsize='small')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,lmax)
    plt.ylim(8e-9,1e-6)
    if save_fig:
        #plt.savefig(dir_out+f'/figs/gmv_comparison_spec_with_resp_test.png')
        #plt.savefig(dir_out+f'/figs/gmv_comparison_spec_with_resp_unl.png')
        #plt.savefig(dir_out+f'/figs/gmv_comparison_spec_with_resp_unl_original.png')
        plt.savefig(dir_out+f'/figs/gmv_comparison_spec_with_resp_unl_original_no_total.png')

def get_n0(sims=np.arange(10)+1,gmv=True,with_BTBE=True,lmax=4096,dir_out='/scratch/users/yukanaka/gmv/'):
    '''
    Get N0 bias.
    '''
    num = len(sims)
    if gmv:
        if with_BTBE:
            append = '_withBTBE'
        else:
            append = ''
        if os.path.isfile(f'/scratch/users/yukanaka/gmv/n0/n0_{num}sims_lmax{lmax}_nside8192_qest_gmv_unl_from_lensed_cls{append}.npy'):
            n0 = np.load(f'/scratch/users/yukanaka/gmv/n0/n0_{num}sims_lmax{lmax}_nside8192_qest_gmv_unl_from_lensed_cls{append}.npy')
            n0_A = np.load(f'/scratch/users/yukanaka/gmv/n0/n0_A_{num}sims_lmax{lmax}_nside8192_qest_gmv_unl_from_lensed_cls{append}.npy')
            n0_B = np.load(f'/scratch/users/yukanaka/gmv/n0/n0_B_{num}sims_lmax{lmax}_nside8192_qest_gmv_unl_from_lensed_cls{append}.npy')
        else:
            n0 = 0
            n0_A = 0
            n0_B = 0
            l = np.arange(0,lmax+1)
            for sim in sims:
                # Get unlensed sim
                plm_gmv = np.load(dir_out+f'/output/plm_healqest_seed{sim}_lmax{lmax}_nside8192_qest_gmv_unl_from_lensed_cls{append}.npy')
                plm_gmv_A = np.load(dir_out+f'/output/plm_healqest_seed{sim}_lmax{lmax}_nside8192_qest_gmv_unl_from_lensed_cls{append}_A.npy')
                plm_gmv_B = np.load(dir_out+f'/output/plm_healqest_seed{sim}_lmax{lmax}_nside8192_qest_gmv_unl_from_lensed_cls{append}_B.npy')
                gmv_resp_data = np.genfromtxt('True_variance_individual_custom_lmin0.0_lmaxT4096_lmaxP4096_beam1_noise5_50.txt')
                # Abhi's code calculates the reconstruction noise for d field rather than phi field, see GMV_QE.py, line 292 for example
                # N is 1/R
                inv_resp_gmv = gmv_resp_data[:,3] / l**2
                inv_resp_gmv_A = gmv_resp_data[:,1] / l**2
                inv_resp_gmv_B = gmv_resp_data[:,2] / l**2
                # Response correct
                plm_gmv_resp_corr = hp.almxfl(plm_gmv,inv_resp_gmv)
                plm_gmv_A_resp_corr = hp.almxfl(plm_gmv_A,inv_resp_gmv_A)
                plm_gmv_B_resp_corr = hp.almxfl(plm_gmv_B,inv_resp_gmv_B)
                # Get spectra
                auto_gmv = hp.alm2cl(plm_gmv_resp_corr, plm_gmv_resp_corr)
                auto_gmv_A = hp.alm2cl(plm_gmv_A_resp_corr, plm_gmv_A_resp_corr)
                auto_gmv_B = hp.alm2cl(plm_gmv_B_resp_corr, plm_gmv_B_resp_corr)
                n0 += auto_gmv
                n0_A += auto_gmv_A
                n0_B += auto_gmv_B
            n0 *= 1/num
            n0_A *= 1/num
            n0_B *= 1/num
            np.save(f'/scratch/users/yukanaka/gmv/n0/n0_{num}sims_lmax{lmax}_nside8192_qest_gmv_unl_from_lensed_cls{append}.npy', n0)
            np.save(f'/scratch/users/yukanaka/gmv/n0/n0_A_{num}sims_lmax{lmax}_nside8192_qest_gmv_unl_from_lensed_cls{append}.npy', n0_A)
            np.save(f'/scratch/users/yukanaka/gmv/n0/n0_B_{num}sims_lmax{lmax}_nside8192_qest_gmv_unl_from_lensed_cls{append}.npy', n0_B)
        return n0, n0_A, n0_B
    else:
        if os.path.isfile(f'/scratch/users/yukanaka/gmv/n0/n0_{num}sims_lmax{lmax}_nside8192_qest_original_unl_from_lensed_cls.npy'):
            n0 = np.load(f'/scratch/users/yukanaka/gmv/n0/n0_{num}sims_lmax{lmax}_nside8192_qest_original_unl_from_lensed_cls.npy')
            n0_TT = np.load(f'/scratch/users/yukanaka/gmv/n0/n0_TT_{num}sims_lmax{lmax}_nside8192_qest_original_unl_from_lensed_cls.npy')
            n0_TE = np.load(f'/scratch/users/yukanaka/gmv/n0/n0_TE_{num}sims_lmax{lmax}_nside8192_qest_original_unl_from_lensed_cls.npy')
            n0_EE = np.load(f'/scratch/users/yukanaka/gmv/n0/n0_EE_{num}sims_lmax{lmax}_nside8192_qest_original_unl_from_lensed_cls.npy')
            n0_TB = np.load(f'/scratch/users/yukanaka/gmv/n0/n0_TB_{num}sims_lmax{lmax}_nside8192_qest_original_unl_from_lensed_cls.npy')
            n0_EB = np.load(f'/scratch/users/yukanaka/gmv/n0/n0_EB_{num}sims_lmax{lmax}_nside8192_qest_original_unl_from_lensed_cls.npy')
        else:
            n0 = 0
            n0_TT = 0
            n0_TE = 0
            n0_EE = 0
            n0_TB = 0
            n0_EB = 0
            l = np.arange(0,lmax+1)
            ests = ['TT', 'EE', 'TE', 'TB', 'EB']
            clfile = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat'
            resp_original_TT = get_analytic_response('TT',lmax,fwhm=1,nlev_t=5,nlev_p=5,clfile=clfile,unl=False)
            resp_original_TE = get_analytic_response('TE',lmax,fwhm=1,nlev_t=5,nlev_p=5,clfile=clfile,unl=False)
            resp_original_EE = get_analytic_response('EE',lmax,fwhm=1,nlev_t=5,nlev_p=5,clfile=clfile,unl=False)
            resp_original_TB = get_analytic_response('TB',lmax,fwhm=1,nlev_t=5,nlev_p=5,clfile=clfile,unl=False)
            resp_original_EB = get_analytic_response('EB',lmax,fwhm=1,nlev_t=5,nlev_p=5,clfile=clfile,unl=False)
            resp_original_ET = get_analytic_response('ET',lmax,fwhm=1,nlev_t=5,nlev_p=5,clfile=clfile,unl=False)
            resp_original_BT = get_analytic_response('BT',lmax,fwhm=1,nlev_t=5,nlev_p=5,clfile=clfile,unl=False)
            resp_original_BE = get_analytic_response('BE',lmax,fwhm=1,nlev_t=5,nlev_p=5,clfile=clfile,unl=False)
            #resp_original = resp_original_TT + resp_original_TE + resp_original_EE + resp_original_TB + resp_original_EB + resp_original_ET + resp_original_BT + resp_original_BE
            resp_original = resp_original_TT + resp_original_TE + resp_original_EE + resp_original_TB + resp_original_EB
            inv_resp_original_TT = np.zeros_like(l,dtype=np.complex_); inv_resp_original_TT[1:] = 1/(resp_original_TT)[1:]
            inv_resp_original_TE = np.zeros_like(l,dtype=np.complex_); inv_resp_original_TE[1:] = 1/(resp_original_TE)[1:]
            inv_resp_original_EE = np.zeros_like(l,dtype=np.complex_); inv_resp_original_EE[1:] = 1/(resp_original_EE)[1:]
            inv_resp_original_TB = np.zeros_like(l,dtype=np.complex_); inv_resp_original_TB[1:] = 1/(resp_original_TB)[1:]
            inv_resp_original_EB = np.zeros_like(l,dtype=np.complex_); inv_resp_original_EB[1:] = 1/(resp_original_EB)[1:]
            inv_resp_original = np.zeros_like(l,dtype=np.complex_); inv_resp_original[1:] = 1/(resp_original)[1:]
            for sim in sims:
                # Get unlensed sim
                plm_original_TT = np.load(dir_out+f'/output/plm_TT_healqest_seed{sim}_lmax{lmax}_nside8192_qest_original_unl_from_lensed_cls.npy')
                plm_original_TE = np.load(dir_out+f'/output/plm_TE_healqest_seed{sim}_lmax{lmax}_nside8192_qest_original_unl_from_lensed_cls.npy')
                plm_original_EE = np.load(dir_out+f'/output/plm_EE_healqest_seed{sim}_lmax{lmax}_nside8192_qest_original_unl_from_lensed_cls.npy')
                plm_original_TB = np.load(dir_out+f'/output/plm_TB_healqest_seed{sim}_lmax{lmax}_nside8192_qest_original_unl_from_lensed_cls.npy')
                plm_original_EB = np.load(dir_out+f'/output/plm_EB_healqest_seed{sim}_lmax{lmax}_nside8192_qest_original_unl_from_lensed_cls.npy')
                #plm_original = plm_original_TT + 2*plm_original_TE + plm_original_EE + 2*plm_original_TB + 2*plm_original_EB
                plm_original = plm_original_TT + plm_original_TE + plm_original_EE + plm_original_TB + plm_original_EB
                # Response correct
                plm_original_resp_corr = hp.almxfl(plm_original,inv_resp_original)
                plm_TT_resp_corr = hp.almxfl(plm_original_TT,inv_resp_original_TT)
                plm_TE_resp_corr = hp.almxfl(plm_original_TE,inv_resp_original_TE)
                plm_EE_resp_corr = hp.almxfl(plm_original_EE,inv_resp_original_EE)
                plm_TB_resp_corr = hp.almxfl(plm_original_TB,inv_resp_original_TB)
                plm_EB_resp_corr = hp.almxfl(plm_original_EB,inv_resp_original_EB)
                # Get spectra
                auto_original = hp.alm2cl(plm_original_resp_corr, plm_original_resp_corr)
                auto_TT = hp.alm2cl(plm_TT_resp_corr, plm_TT_resp_corr)
                auto_TE = hp.alm2cl(plm_TE_resp_corr, plm_TE_resp_corr)
                auto_EE = hp.alm2cl(plm_EE_resp_corr, plm_EE_resp_corr)
                auto_TB = hp.alm2cl(plm_TB_resp_corr, plm_TB_resp_corr)
                auto_EB = hp.alm2cl(plm_EB_resp_corr, plm_EB_resp_corr)
                n0 += auto_original
                n0_TT += auto_TT
                n0_TE += auto_TE
                n0_EE += auto_EE
                n0_TB += auto_TB
                n0_EB += auto_EB
            n0 *= 1/num
            n0_TT *= 1/num
            n0_TE *= 1/num
            n0_EE *= 1/num
            n0_TB *= 1/num
            n0_EB *= 1/num
            #np.save(f'/scratch/users/yukanaka/gmv/n0/n0_{num}sims_lmax{lmax}_nside8192_qest_original_unl_from_lensed_cls.npy', n0)
            #np.save(f'/scratch/users/yukanaka/gmv/n0/n0_TT_{num}sims_lmax{lmax}_nside8192_qest_original_unl_from_lensed_cls.npy', n0_TT)
            #np.save(f'/scratch/users/yukanaka/gmv/n0/n0_TE_{num}sims_lmax{lmax}_nside8192_qest_original_unl_from_lensed_cls.npy', n0_TE)
            #np.save(f'/scratch/users/yukanaka/gmv/n0/n0_EE_{num}sims_lmax{lmax}_nside8192_qest_original_unl_from_lensed_cls.npy', n0_EE)
            #np.save(f'/scratch/users/yukanaka/gmv/n0/n0_TB_{num}sims_lmax{lmax}_nside8192_qest_original_unl_from_lensed_cls.npy', n0_TB)
            #np.save(f'/scratch/users/yukanaka/gmv/n0/n0_EB_{num}sims_lmax{lmax}_nside8192_qest_original_unl_from_lensed_cls.npy', n0_EB)
        return n0, n0_TT, n0_TE, n0_EE, n0_TB, n0_EB

def get_analytic_response(est, lmax=4096, fwhm=1, nlev_t=5, nlev_p=5, u=None,
                          clfile='/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat',
                          from_quicklens=False,unl=False,filename=None,qeZA=None):
    '''
    NEEDS PYTHON2 if the analytic response is not already saved and from_quicklens is True.
    See https://github.com/dhanson/quicklens/blob/master/examples/plot_lens_reconstruction_noise_levels.py.
    '''
    if filename is None:
        if from_quicklens:
            filename = '/scratch/users/yukanaka/gmv/resp/an_resp_{}_quicklens_lmax{}_fwhm{}_nlevt{}_nlevp{}.npy'.format(est,lmax,fwhm,nlev_t,nlev_p)
        elif unl:
            filename = '/scratch/users/yukanaka/gmv/resp/an_resp_{}_healqest_lmax{}_fwhm{}_nlevt{}_nlevp{}_unl.npy'.format(est,lmax,fwhm,nlev_t,nlev_p)
        else:
            filename = '/scratch/users/yukanaka/gmv/resp/an_resp_{}_healqest_lmax{}_fwhm{}_nlevt{}_nlevp{}.npy'.format(est,lmax,fwhm,nlev_t,nlev_p)
    if os.path.isfile(filename):
        R = np.load(filename)
    else:
        # First get the theory spectra and filter functions
        #pars = camb.CAMBparams()
        #pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
        #pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
        #pars.set_for_lmax(2500, lens_potential_accuracy=0)
        #results = camb.get_results(pars)
        #sltt,slee,slbb,slte = results.get_cmb_power_spectra(pars,lmax=lmax, CMB_unit='muK',raw_cl=True)['lensed_scalar'].T
        #ell = np.arange(lmax+1)
        if unl:
            ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile,lmax)
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
            if from_quicklens:
                pass
                #q = ql.qest.lens.phi_TT(sltt)
            flX = flt
            flY = flt
        elif est == 'EE':
            if from_quicklens:
                pass
                #q = ql.qest.lens.phi_EE(slee)
            flX = fle
            flY = fle
        elif est == 'TE':
            if from_quicklens:
                pass
                #q = ql.qest.lens.phi_TE(slte)
            flX = flt
            flY = fle
        elif est == 'TB':
            if from_quicklens:
                pass
                #q = ql.qest.lens.phi_TB(slte)
            flX = flt
            flY = flb
        elif est == 'BT':
            if from_quicklens:
                pass
                #q = ql.qest.lens.phi_BT(slte)
            flX = flb
            flY = flt
        elif est == 'EB':
            if from_quicklens:
                pass
                #q = ql.qest.lens.phi_EB(slee)
            flX = fle
            flY = flb
        elif est == 'ET':
            flX = fle
            flY = flt
        elif est == 'BE':
            flX = flb
            flY = fle
        if from_quicklens:
            pass
            #R = q.fill_resp(q, np.zeros(lmax+1, dtype=np.complex), flX, flY)
        else:
            #pass
            R = resp.fill_resp(weights_combined.weights(est,lmax,clfile,u=u), np.zeros(lmax+1, dtype=np.complex_), flX, flY, qeZA=qeZA)
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

