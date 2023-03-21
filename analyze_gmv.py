#!/usr/bin/env python3
import numpy as np
import healpy as hp
import camb
import os, sys
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
from astropy.io import fits
import utils
import matplotlib.pyplot as plt
import weights_gmv
import qest
import wignerd
import resp

def compare_gmv(sim=2,lmax=4096,nside=8192,dir_out='/scratch/users/yukanaka/gmv/',save_fig=True,unl=True):
    l = np.arange(0,lmax+1)
    ests = ['TT', 'EE', 'TE', 'TB', 'EB']

    # Load plms
    if unl:
        #clfile = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
        clfile = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat'
        plm_gmv = np.load(dir_out+f'/output/plm_healqest_seed{sim}_lmax{lmax}_nside{nside}_qest_gmv_unl_from_lensed_cls_withBTBE.npy')
        plm_gmv_A = np.load(dir_out+f'/output/plm_healqest_seed{sim}_lmax{lmax}_nside{nside}_qest_gmv_unl_from_lensed_cls_withBTBE_A.npy')
        plm_gmv_B = np.load(dir_out+f'/output/plm_healqest_seed{sim}_lmax{lmax}_nside{nside}_qest_gmv_unl_from_lensed_cls_withBTBE_B.npy')
        plm_gmv_noBTBE = np.load(dir_out+f'/output/plm_healqest_seed{sim}_lmax{lmax}_nside{nside}_qest_gmv_unl_from_lensed_cls.npy')
    else:
        clfile = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat'
        plm_gmv = np.load(dir_out+f'/output/plm_healqest_seed{sim}_lmax{lmax}_nside{nside}_qest_gmv_withBTBE.npy')
        plm_gmv_A = np.load(dir_out+f'/output/plm_healqest_seed{sim}_lmax{lmax}_nside{nside}_qest_gmv_withBTBE_A.npy')
        plm_gmv_B = np.load(dir_out+f'/output/plm_healqest_seed{sim}_lmax{lmax}_nside{nside}_qest_gmv_withBTBE_B.npy')
        plm_gmv_noBTBE = np.load(dir_out+f'/output/plm_healqest_seed{sim}_lmax{lmax}_nside{nside}_qest_gmv.npy')
        #plm_gmv = np.load(dir_out+f'/output/plm_healqest_seed{sim}_lmax{lmax}_nside{nside}_qest_gmv_yuuki.npy')
        #plm_gmv_A = np.load(dir_out+f'/output/plm_healqest_seed{sim}_lmax{lmax}_nside{nside}_qest_gmv_A_yuuki.npy')
        #plm_gmv_B = np.load(dir_out+f'/output/plm_healqest_seed{sim}_lmax{lmax}_nside{nside}_qest_gmv_B_yuuki.npy')
        #plm_gmv = np.load(dir_out+f'/output/plm_healqest_seed{sim}_lmax{lmax}_nside{nside}_qest_gmv_yuuki_no_BTBE.npy')
        #plm_gmv_A = np.load(dir_out+f'/output/plm_healqest_seed{sim}_lmax{lmax}_nside{nside}_qest_gmv_A_yuuki_no_BTBE.npy')
        #plm_gmv_B = np.load(dir_out+f'/output/plm_healqest_seed{sim}_lmax{lmax}_nside{nside}_qest_gmv_B_yuuki_no_BTBE.npy')

    plm_original = np.zeros(len(plm_gmv), dtype=np.complex_)
    for i, est in enumerate(ests):
        if unl:
            plm_original += np.load(dir_out+f'/output/plm_{est}_healqest_seed{sim}_lmax{lmax}_nside{nside}_qest_original_unl_from_lensed_cls.npy')
        else:
            plm_original += np.load(dir_out+f'/output/plm_{est}_healqest_seed{sim}_lmax{lmax}_nside{nside}_qest_original.npy')

    # Response correct
    resp_original = np.zeros_like(l, dtype=np.complex_)
    for i, est in enumerate(ests):
        resp_original += get_analytic_response(est,lmax,fwhm=1,nlev_t=5,nlev_p=5,clfile=clfile,unl=False)
    #resp_original_TT = get_analytic_response('TT',lmax,fwhm=1,nlev_t=5,nlev_p=5,clfile=clfile,unl=False)
    #resp_original_EB = get_analytic_response('EB',lmax,fwhm=1,nlev_t=5,nlev_p=5,clfile=clfile,unl=False)
    inv_resp_original = np.zeros_like(l,dtype=np.complex_); inv_resp_original[1:] = 1/(resp_original)[1:]
    #inv_resp_original_TT = np.zeros_like(l,dtype=np.complex_); inv_resp_original_TT[1:] = 1/(resp_original_TT)[1:]
    #inv_resp_original_EB = np.zeros_like(l,dtype=np.complex_); inv_resp_original_EB[1:] = 1/(resp_original_EB)[1:]
    plm_original_resp_corr = hp.almxfl(plm_original,inv_resp_original)

    gmv_resp_data = np.genfromtxt('True_variance_individual_custom_lmin0.0_lmaxT4096_lmaxP4096_beam1_noise5_50.txt')
    # Abhi's code calculates the reconstruction noise for d field rather than phi field, see GMV_QE.py, line 292 for example
    inv_resp_gmv = gmv_resp_data[:,3] / l**2
    inv_resp_gmv_A = gmv_resp_data[:,1] / l**2
    inv_resp_gmv_B = gmv_resp_data[:,2] / l**2
    # N is 1/R
    plm_gmv_resp_corr = hp.almxfl(plm_gmv,inv_resp_gmv)
    plm_gmv_A_resp_corr = hp.almxfl(plm_gmv_A,inv_resp_gmv_A)
    plm_gmv_B_resp_corr = hp.almxfl(plm_gmv_B,inv_resp_gmv_B)
    plm_gmv_resp_corr_noBTBE = hp.almxfl(plm_gmv_noBTBE,inv_resp_gmv)

    # Get spectra
    cross = hp.alm2cl(plm_gmv_resp_corr, plm_original_resp_corr, lmax=lmax) * (l*(l+1))**2/4
    auto_gmv = hp.alm2cl(plm_gmv_resp_corr, plm_gmv_resp_corr, lmax=lmax) * (l*(l+1))**2/4
    auto_gmv_A = hp.alm2cl(plm_gmv_A_resp_corr, plm_gmv_A_resp_corr, lmax=lmax) * (l*(l+1))**2/4
    auto_gmv_B = hp.alm2cl(plm_gmv_B_resp_corr, plm_gmv_B_resp_corr, lmax=lmax) * (l*(l+1))**2/4
    auto_original = hp.alm2cl(plm_original_resp_corr, plm_original_resp_corr, lmax=lmax) * (l*(l+1))**2/4
    auto_gmv_noBTBE = hp.alm2cl(plm_gmv_resp_corr_noBTBE, plm_gmv_resp_corr_noBTBE, lmax=lmax) * (l*(l+1))**2/4

    # Theory spectrum
    clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
    ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
    clkk = slpp * (l*(l+1))**2/4

    # Get N0 bias
    n0, n0_A, n0_B = get_n0(gmv=True, with_BTBE=True)
    n0_original = get_n0(gmv=False)

    input_plm = hp.read_alm(f'/scratch/users/yukanaka/full_res_maps/phi/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_phi1_seed{sim}.alm')
    #TODO
    #input_plm_resp_corr = hp.almxfl(input_plm,inv_resp_gmv)
    #cross_input_output = hp.alm2cl(input_plm, plm_gmv_resp_corr)

    #fudgefac_A = 0.85
    #fudgefac_B = 2

    # Plot
    plt.figure(0)
    plt.clf()
    plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')
    #plt.plot(l, cross, color='palegoldenrod', label='Cross Spectrum (Original x GMV)')
    #plt.plot(l, auto_original, color='darkblue', label="Auto Spectrum (Original)")
    #plt.plot(l, auto_gmv, color='firebrick', label="Auto Spectrum (GMV)")
    #plt.plot(l, auto_gmv_A, color='forestgreen', label=f'Auto Spectrum (GMV [TT, EE, TE])')
    #plt.plot(l, auto_gmv_B, color='mediumorchid', label="Auto Spectrum (GMV [TB, EB])")
    #plt.plot(l, auto_gmv_noBTBE, color='slategrey', label="Auto Spectrum (GMV, no BT/BE)")
    #plt.plot(l, n0_original * (l*(l+1))**2/4, color='darkblue', linestyle='-', label='$N_0$ from 10 Unlensed Sims (Original)')
    plt.plot(l, n0 * (l*(l+1))**2/4, color='firebrick', linestyle='-', label='$N_0$ from 10 Unlensed Sims (GMV)')
    plt.plot(l, n0_A * (l*(l+1))**2/4, color='forestgreen', linestyle='-', label='$N_0$ from 10 Unlensed Sims (GMV [TT, EE, TE])')
    plt.plot(l, n0_B * (l*(l+1))**2/4, color='mediumorchid', linestyle='-', label='$N_0$ from 10 Unlensed Sims (GMV [TB, EB])')
    #plt.plot(l, auto_original - inv_resp_original * (l*(l+1))**2/4, color='darkblue', label="Auto Spectrum - 1/R (Original)")
    #plt.plot(l, auto_gmv - inv_resp_gmv * (l*(l+1))**2/4, color='firebrick', label="Auto Spectrum - 1/R (GMV)")
    #plt.plot(l, auto_gmv_A - inv_resp_gmv_A * (l*(l+1))**2/4, color='forestgreen', label=f'Auto Spectrum - 1/R (GMV [TT, EE, TE])')
    #plt.plot(l, auto_gmv_B - inv_resp_gmv_B * (l*(l+1))**2/4, color='mediumorchid', label="Auto Spectrum - 1/R (GMV [TB, EB])")
    #plt.plot(l, auto_original - n0_original * (l*(l+1))**2/4, color='darkblue', label="Auto Spectrum - $N_0$ (Original)")
    #plt.plot(l, auto_gmv - n0 * (l*(l+1))**2/4, color='firebrick', label="Auto Spectrum - $N_0$ (GMV)")
    #plt.plot(l, auto_gmv_A - n0_A * (l*(l+1))**2/4, color='forestgreen', label=f'Auto Spectrum - $N_0$ (GMV [TT, EE, TE])')
    #plt.plot(l, auto_gmv_B - n0_B * (l*(l+1))**2/4, color='mediumorchid', label="Auto Spectrum - $N_0$ (GMV [TB, EB])")
    #plt.plot(l, inv_resp_original * (l*(l+1))**2/4, color='cornflowerblue', linestyle='--', label='1/R (Original)')
    #plt.plot(l, inv_resp_original_TT * (l*(l+1))**2/4, color='cornflowerblue', linestyle='--', label='1/R (Original, TT)')
    #plt.plot(l, inv_resp_original_EB * (l*(l+1))**2/4, color='lightgreen', linestyle='--', label='1/R (Original, EB)')
    plt.plot(l, inv_resp_gmv * (l*(l+1))**2/4, color='lightcoral', linestyle='--', label='1/R (GMV)')
    plt.plot(l, inv_resp_gmv_A * (l*(l+1))**2/4, color='lightgreen', linestyle='--', label='1/R (GMV [TT, EE, TE])')
    plt.plot(l, inv_resp_gmv_B * (l*(l+1))**2/4, color='plum', linestyle='--', label='1/R (GMV [TB, EB])')
    plt.ylabel("$C_\ell^{\kappa\kappa}$")
    plt.xlabel('$\ell$')
    plt.title('Spectra with Response Correction')
    #plt.title('1/R with 5 uK-arcmin Noise Levels for T/P')
    plt.legend(loc='upper right', fontsize='small')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,lmax)
    plt.ylim(8e-9,1e-6)
    if save_fig:
        #plt.savefig(dir_out+f'/figs/gmv_comparison_spec_with_resp_test.png')
        plt.savefig(dir_out+f'/figs/gmv_comparison_spec_with_resp_unl.png')
        #plt.savefig(dir_out+f'/figs/gmv_comparison_spec_with_resp_dominic.png')
        #plt.savefig(dir_out+f'/figs/gmv_comparison_spec_with_resp_unl_gmv_and_original_only.png')
        #plt.savefig(dir_out+f'/figs/gmv_comparison_spec_with_resp_1_over_R_subtracted.png')
        #plt.savefig(dir_out+f'/figs/gmv_comparison_spec_with_resp_N_0_subtracted.png')
        #plt.savefig(dir_out+f'/figs/gmv_comparison_spec_with_resp.png')
    #plt.show()

def get_n0(sims=np.arange(10)+1,gmv=True,with_BTBE=False,lmax=4096,dir_out='/scratch/users/yukanaka/gmv/'):
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
        else:
            n0 = 0
            l = np.arange(0,lmax+1)
            ests = ['TT', 'EE', 'TE', 'TB', 'EB']
            clfile = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat'
            resp_original = np.zeros_like(l, dtype=np.complex_)
            for i, est in enumerate(ests):
                resp_original += get_analytic_response(est,lmax,fwhm=1,nlev_t=5,nlev_p=5,clfile=clfile,unl=False)
            inv_resp_original = np.zeros_like(l,dtype=np.complex_); inv_resp_original[1:] = 1/(resp_original)[1:]
            for sim in sims:
                # Get unlensed sim
                plm_original = np.load(dir_out+f'/output/plm_TT_healqest_seed{sim}_lmax{lmax}_nside8192_qest_original_unl_from_lensed_cls.npy')
                plm_original = np.zeros_like(plm_original, dtype=np.complex_)
                for i, est in enumerate(ests):
                    plm_original += np.load(dir_out+f'/output/plm_{est}_healqest_seed{sim}_lmax{lmax}_nside8192_qest_original_unl_from_lensed_cls.npy')
                # Response correct
                plm_original_resp_corr = hp.almxfl(plm_original,inv_resp_original)
                # Get spectra
                auto_original = hp.alm2cl(plm_original_resp_corr, plm_original_resp_corr)
                n0 += auto_original
            n0 *= 1/num
            np.save(f'/scratch/users/yukanaka/gmv/n0/n0_{num}sims_lmax{lmax}_nside8192_qest_original_unl_from_lensed_cls.npy', n0)
        return n0

def get_analytic_response(est, lmax=4096, fwhm=1, nlev_t=5, nlev_p=5,
                          clfile='/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat',
                          from_quicklens=False,unl=False):
    '''
    NEEDS PYTHON2 if the analytic response is not already saved and from_quicklens is True.
    See https://github.com/dhanson/quicklens/blob/master/examples/plot_lens_reconstruction_noise_levels.py.
    '''
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
        if est == 'TT':
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
        if from_quicklens:
            pass
            #R = q.fill_resp(q, np.zeros(lmax+1, dtype=np.complex), flX, flY)
        else:
            #pass
            R = resp.fill_resp(weights_gmv.weights(est,lmax,clfile,unl=unl), np.zeros(lmax+1, dtype=np.complex_), flX, flY)
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
