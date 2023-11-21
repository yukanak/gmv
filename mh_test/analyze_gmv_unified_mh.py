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
            n0=False,n1=False,resp_from_sims=False,
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
    ests = ['TT', 'EE', 'TE', 'TB', 'EB']
    resps_original = np.zeros((len(l),len(ests)), dtype=np.complex_)
    inv_resps_original = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
    if resp_from_sims:
        pass
    else: 
        for i, est in enumerate(ests):
            resps_original[:,i] = get_analytic_response(est,config,gmv=False)
            inv_resps_original[1:,i] = 1/(resps_original)[1:,i]
        resp_original = resps_original[:,0]+resps_original[:,1]+2*resps_original[:,2]+2*resps_original[:,3]+2*resps_original[:,4]
    inv_resp_original = np.zeros_like(l,dtype=np.complex_); inv_resp_original[1:] = 1/(resp_original)[1:]

    # GMV response
    if resp_from_sims:
        pass
    else:
        resp_gmv = get_analytic_response('all',config,gmv=True)
        resp_gmv_TTEETE = get_analytic_response('TTEETE',config,gmv=True)
        resp_gmv_TBEB = get_analytic_response('TBEB',config,gmv=True)
    inv_resp_gmv = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv[1:] = 1./(resp_gmv)[1:]
    inv_resp_gmv_TTEETE = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TTEETE[1:] = 1./(resp_gmv_TTEETE)[1:]
    inv_resp_gmv_TBEB = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TBEB[1:] = 1./(resp_gmv_TBEB)[1:]

    if n0:
        pass

    if n1:
        pass

    auto_gmv_all = 0
    auto_original_all = 0
    cross_gmv_all = 0
    cross_original_all = 0

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

        # Response correct
        plm_gmv_resp_corr = hp.almxfl(plm_gmv,inv_resp_gmv)
        plm_original_resp_corr = hp.almxfl(plm_original,inv_resp_original)

        # Get spectra
        auto_gmv = hp.alm2cl(plm_gmv_resp_corr, plm_gmv_resp_corr, lmax=lmax) * (l*(l+1))**2/4
        auto_original = hp.alm2cl(plm_original_resp_corr, plm_original_resp_corr, lmax=lmax) * (l*(l+1))**2/4

        # N0 and N1 subtract
        if n0 and n1:
            pass

        elif n0:
            pass

        # Sum the auto spectra
        auto_gmv_all += auto_gmv
        auto_original_all += auto_original
        if n0:
            pass

        if not unl:
            input_plm = hp.read_alm(f'/scratch/users/yukanaka/lensing19-20/inputcmb/phi/phi_lmax_{lmax}/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_phi1_seed{sim}_lmax{lmax}.alm')
            # Cross correlate with input plm
            cross_gmv_all += hp.alm2cl(input_plm, plm_gmv_resp_corr, lmax=lmax) * (l*(l+1))**2/4
            cross_original_all += hp.alm2cl(input_plm, plm_original_resp_corr, lmax=lmax) * (l*(l+1))**2/4
            # If debiasing, get the binned ratio against input
            if n0:
                pass

    # Average
    auto_gmv_avg = auto_gmv_all / num
    auto_original_avg = auto_original_all / num
    if n0:
        pass

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
        plt.savefig(dir_out+f'/figs/{num}_sims_comparison_{append}.png',bbox_inches='tight')
        #plt.savefig(dir_out+f'/figs/{num}_sims_comparison_{append}_n0n1subtracted.png',bbox_inches='tight')

def compare_resp(config_file='mh_yuka.yaml',
                 save_fig=True):
    config = utils.parse_yaml(config_file)
    lmax = config['Lmax']
    lmin = config['lminT']
    dir_out = config['dir_out']
    l = np.arange(0,lmax+1)

    # SQE response
    ests = ['TT', 'EE', 'TE', 'TB', 'EB']
    resps_original = np.zeros((len(l),len(ests)), dtype=np.complex_)
    inv_resps_original = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
    for i, est in enumerate(ests):
        resps_original[:,i] = get_analytic_response(est,config,gmv=False)
        inv_resps_original[1:,i] = 1/(resps_original)[1:,i]
    resp_original = resps_original[:,0]+resps_original[:,1]+2*resps_original[:,2]+2*resps_original[:,3]+2*resps_original[:,4]
    inv_resp_original = np.zeros_like(l,dtype=np.complex_); inv_resp_original[1:] = 1/(resp_original)[1:]

    # GMV response
    resp_gmv = get_analytic_response('all',config,gmv=True)
    resp_gmv_TTEETE = get_analytic_response('TTEETE',config,gmv=True)
    resp_gmv_TBEB = get_analytic_response('TBEB',config,gmv=True)
    inv_resp_gmv = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv[1:] = 1./(resp_gmv)[1:]
    inv_resp_gmv_TTEETE = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TTEETE[1:] = 1./(resp_gmv_TTEETE)[1:]
    inv_resp_gmv_TBEB = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TBEB[1:] = 1./(resp_gmv_TBEB)[1:]

    # SQE response from before (2019/2020 ILC noise curves that are NOT correlated between frequencies, no foregrounds)
    resps_original_old = np.zeros((len(l),len(ests)), dtype=np.complex_)
    inv_resps_original_old = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
    for i, est in enumerate(ests):
        resps_original_old[:,i] = np.load(f'/scratch/users/yukanaka/gmv/resp/an_resp_sqe_est{est}_lmaxT3000_lmaxP4096_lmin300_cltypelcmb_added_noise_from_file.npy')
        inv_resps_original_old[1:,i] = 1/(resps_original_old)[1:,i]
    resp_original_old = resps_original_old[:,0]+resps_original_old[:,1]+2*resps_original_old[:,2]+2*resps_original_old[:,3]+2*resps_original_old[:,4]
    inv_resp_original_old = np.zeros_like(l,dtype=np.complex_); inv_resp_original_old[1:] = 1/(resp_original_old)[1:]

    # GMV response from before (2019/2020 ILC noise curves that are NOT correlated between frequencies, no foregrounds)
    resp_gmv_old = np.load(f'/scratch/users/yukanaka/gmv/resp/an_resp_gmv_estall_lmaxT3000_lmaxP4096_lmin300_cltypelcmb_added_noise_from_file.npy')
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
    plt.plot(l, inv_resps_original[:,0] * (l*(l+1))**2/4, color='sienna', linestyle='-', label='$1/R$ (SQE, TT)')
    plt.plot(l, inv_resps_original[:,1] * (l*(l+1))**2/4, color='mediumorchid', linestyle='-', label='$1/R$ (SQE, EE)')
    plt.plot(l, 0.5*inv_resps_original[:,2] * (l*(l+1))**2/4, color='forestgreen', linestyle='-', label='$1/(2R)$ (SQE, TE)')
    plt.plot(l, 0.5*inv_resps_original[:,3] * (l*(l+1))**2/4, color='gold', linestyle='-', label='$1/(2R)$ (SQE, TB)')
    plt.plot(l, 0.5*inv_resps_original[:,4] * (l*(l+1))**2/4, color='orange', linestyle='-', label='$1/(2R$) (SQE, EB)')

    plt.plot(l, inv_resp_original_old * (l*(l+1))**2/4, color='lightcoral', linestyle='--', label='$1/R$ (SQE, old)')
    plt.plot(l, inv_resps_original_old[:,0] * (l*(l+1))**2/4, color='sandybrown', linestyle='--', label='$1/R$ (SQE, TT old)')
    plt.plot(l, inv_resps_original_old[:,1] * (l*(l+1))**2/4, color='plum', linestyle='--', label='$1/R$ (SQE, EE old)')
    plt.plot(l, 0.5*inv_resps_original_old[:,2] * (l*(l+1))**2/4, color='lightgreen', linestyle='--', label='$1/(2R)$ (SQE, TE old)')
    plt.plot(l, 0.5*inv_resps_original_old[:,3] * (l*(l+1))**2/4, color='palegoldenrod', linestyle='--', label='$1/(2R)$ (SQE, TB old)')
    plt.plot(l, 0.5*inv_resps_original_old[:,4] * (l*(l+1))**2/4, color='bisque', linestyle='--', label='$1/(2R$) (SQE, EB old)')

    #plt.plot(l, inv_resp_gmv * (l*(l+1))**2/4, color='darkblue', linestyle='-', label='$1/R$ (GMV)')
    #plt.plot(l, inv_resp_gmv_TTEETE * (l*(l+1))**2/4, color='forestgreen', linestyle='-', label='1/R (GMV, TTEETE)')
    #plt.plot(l, inv_resp_gmv_TBEB * (l*(l+1))**2/4, color='blueviolet', linestyle='-', label='1/R (GMV, TBEB)')

    #plt.plot(l, inv_resp_gmv_old * (l*(l+1))**2/4, color='cornflowerblue', linestyle='--', label='$1/R$ (GMV, old)')
    #plt.plot(l, inv_resp_gmv_TTEETE_old * (l*(l+1))**2/4, color='lightgreen', linestyle='--', label='1/R (GMV, TTEETE old)')
    #plt.plot(l, inv_resp_gmv_TBEB_old * (l*(l+1))**2/4, color='thistle', linestyle='--', label='1/R (GMV, TBEB old)')

    plt.ylabel("$C_\ell^{\kappa\kappa}$")
    plt.xlabel('$\ell$')
    plt.title('$1/R$')
    plt.legend(loc='upper left', fontsize='small', bbox_to_anchor=(1,0.5))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,lmax)
    plt.ylim(8e-9,1e-5)
    if save_fig:
        #plt.savefig(dir_out+f'/figs/mh_response_comparison.png',bbox_inches='tight')
        plt.savefig(dir_out+f'/figs/mh_response_comparison_sqe_only.png',bbox_inches='tight')
        #plt.savefig(dir_out+f'/figs/mh_response_comparison_gmv_only.png',bbox_inches='tight')

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
        filename = f'/scratch/users/yukanaka/gmv/resp/an_resp{append}.npy'

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

