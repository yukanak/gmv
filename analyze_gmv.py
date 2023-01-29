import numpy as np
import healpy as hp
import camb
import os, sys
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
from astropy.io import fits
import utils
import matplotlib.pyplot as plt
import weights
import qest
import wignerd
import resp

def compare_gmv(sim=1,lmax=4096,nside=8192,dir_out='/scratch/users/yukanaka/gmv/',save_fig=True):
    l = np.arange(0,lmax+1)
    ests = ['TT', 'EE', 'TE', 'TB', 'EB']
    # Load plms
    plm_gmv = np.load(dir_out+f'/output/plm_healqest_seed{sim}_lmax{lmax}_nside{nside}_qest_gmv_nonzeronlte_floattypealm.npy')
    plm_original = np.zeros((len(plm_gmv),5), dtype=np.complex_)
    for i, est in enumerate(ests):
        plm_original[:,i] = np.load(dir_out+f'/output/plm_{est}_healqest_seed{sim}_lmax{lmax}_nside{nside}_qest_original.npy')
    #TODO: response for GMV? should i bother for this comparison?
    #TODO: is this the correct way to get the "unbiased" estimator combining all estimators for the HO method?
    plm_original_tot = np.zeros(len(plm_gmv), dtype=np.complex_)
    for i, est in enumerate(ests):
        R_an = get_analytic_response(est,lmax,fwhm=1,nlev_t=5,nlev_p=5)
        inv_R = np.zeros_like(l,dtype=np.complex_); inv_R[1:] = 1/(R_an)[1:]
        plm_original_tot += hp.almxfl(plm_original[:,i],inv_R)
    # Convert to kappa
    #klm_gmv = hp.almxfl(plm_gmv, l*(l+1)/2)
    #klm_original = hp.almxfl(plm_original_tot, l*(l+1)/2)

    # Get spectra
    cross = hp.alm2cl(plm_gmv, plm_original_tot, lmax=lmax) * (l*(l+1))**2/4
    auto_gmv = hp.alm2cl(plm_gmv, plm_gmv, lmax=lmax) * (l*(l+1))**2/4
    auto_original = hp.alm2cl(plm_original_tot, plm_original_tot, lmax=lmax) * (l*(l+1))**2/4

    # Theory spectrum
    clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
    ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
    clkk = slpp * (l*(l+1))**2/4

    # Plot
    plt.figure(0)
    plt.clf()
    #plt.plot(l, cross, 'b', label='Cross Spectrum')
    #plt.plot(l, auto_gmv, 'y--', label="Auto Spectrum (GMV)")
    plt.plot(l, auto_original, 'g:', label="Auto Spectrum (Original)")
    plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')
    plt.ylabel("$C_\ell^{\kappa\kappa}$")
    plt.xlabel('$\ell$')
    plt.title('Spectra with No Response Correction')
    plt.legend(loc='upper right', fontsize='small')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,lmax)
    if save_fig:
        #plt.savefig(dir_out+f'/figs/gmv_comparison_spec_no_resp.png')
        plt.savefig(dir_out+f'/figs/gmv_comparison_spec_no_resp_auto_original_only.png')
    #plt.show()

def get_analytic_response(est, lmax=4096, fwhm=1, nlev_t=5, nlev_p=5,
                          clfile='/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat',
                          from_quicklens=False):
    '''
    NEEDS PYTHON2 if the analytic response is not already saved and from_quicklens is True.
    See https://github.com/dhanson/quicklens/blob/master/examples/plot_lens_reconstruction_noise_levels.py.
    '''
    if from_quicklens:
        filename = '/scratch/users/yukanaka/gmv/resp/an_resp_{}_quicklens_lmax{}_fwhm{}_nlevt{}_nlevp{}.npy'.format(est,lmax,fwhm,nlev_t,nlev_p)
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
            R = resp.fill_resp(weights.weights(est,lmax,clfile), np.zeros(lmax+1, dtype=np.complex_), flX, flY)
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
