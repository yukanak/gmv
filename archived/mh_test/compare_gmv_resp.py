#!/usr/bin/env python3
import numpy as np
import pickle
import healpy as hp
import camb
import os, sys
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import gmv_resp
import gmv_resp_numericalinv
import gmv_resp_alt
import healqest_utils as utils
import matplotlib.pyplot as plt
import weights
import qest
import wignerd
import resp

def analyze(sims=np.arange(40)+1,config_file='mh_yuka.yaml'):
    '''
    Compare GMV response calculated from different M inverse matrices.
    '''
    config = utils.parse_yaml(config_file)
    lmax = config['lensrec']['Lmax']
    lmin = config['lensrec']['lminT']
    lmaxT = config['lensrec']['lmaxT']
    lmaxP = config['lensrec']['lmaxP']
    nside = config['lensrec']['nside']
    dir_out = config['dir_out']
    l = np.arange(0,lmax+1)

    # GMV sim response
    #sim_resp_gmv = get_sim_response('all',config,gmv=True,sims=sims)
    #sim_resp_gmv_TTEETE = get_sim_response('TTEETE',config,gmv=True,sims=sims)
    #sim_resp_gmv_TBEB = get_sim_response('TBEB',config,gmv=True,sims=sims)
    #inv_sim_resp_gmv = np.zeros(len(l),dtype=np.complex_); inv_sim_resp_gmv[1:] = 1./(sim_resp_gmv)[1:]
    #inv_sim_resp_gmv_TTEETE = np.zeros(len(l),dtype=np.complex_); inv_sim_resp_gmv_TTEETE[1:] = 1./(sim_resp_gmv_TTEETE)[1:]
    #inv_sim_resp_gmv_TBEB = np.zeros(len(l),dtype=np.complex_); inv_sim_resp_gmv_TBEB[1:] = 1./(sim_resp_gmv_TBEB)[1:]

    # GMV response, Abhi's version, NOT CROSS-ILC, has noise and foregrounds
    resp_gmv = get_gmv_analytic_response('all',config)
    resp_gmv_TTEETE = get_gmv_analytic_response('TTEETE',config)
    resp_gmv_TBEB = get_gmv_analytic_response('TBEB',config)
    inv_resp_gmv = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv[1:] = 1./(resp_gmv)[1:]
    inv_resp_gmv_TTEETE = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TTEETE[1:] = 1./(resp_gmv_TTEETE)[1:]
    inv_resp_gmv_TBEB = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TBEB[1:] = 1./(resp_gmv_TBEB)[1:]

    # GMV response, my version, NOT CROSS-ILC
    resp_gmv_alt = get_gmv_analytic_response('all',config,alt=True)
    resp_gmv_TTEETE_alt = get_gmv_analytic_response('TTEETE',config,alt=True)
    resp_gmv_TBEB_alt = get_gmv_analytic_response('TBEB',config,alt=True)
    inv_resp_gmv_alt = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_alt[1:] = 1./(resp_gmv_alt)[1:]
    inv_resp_gmv_TTEETE_alt = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TTEETE_alt[1:] = 1./(resp_gmv_TTEETE_alt)[1:]
    inv_resp_gmv_TBEB_alt = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TBEB_alt[1:] = 1./(resp_gmv_TBEB_alt)[1:]

    # GMV response, my version, CROSS-ILC
    resp_gmv_alt_crossilc = get_gmv_analytic_response('all',config,alt=True,crossilc=True)
    resp_gmv_TTEETE_alt_crossilc = get_gmv_analytic_response('TTEETE',config,alt=True,crossilc=True)
    resp_gmv_TBEB_alt_crossilc = get_gmv_analytic_response('TBEB',config,alt=True,crossilc=True)
    inv_resp_gmv_alt_crossilc = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_alt_crossilc[1:] = 1./(resp_gmv_alt_crossilc)[1:]
    inv_resp_gmv_TTEETE_alt_crossilc = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TTEETE_alt_crossilc[1:] = 1./(resp_gmv_TTEETE_alt_crossilc)[1:]
    inv_resp_gmv_TBEB_alt_crossilc = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TBEB_alt_crossilc[1:] = 1./(resp_gmv_TBEB_alt_crossilc)[1:]

    # GMV response from before (2019/2020 ILC noise curves that are NOT correlated between frequencies, no foregrounds)
    resp_gmv_old = np.load(dir_out+f'/resp/an_resp_gmv_estall_lmaxT3000_lmaxP4096_lmin300_cltypelcmb_added_noise_from_file.npy')
    resp_gmv_TTEETE_old = resp_gmv_old[:,1]
    resp_gmv_TBEB_old = resp_gmv_old[:,2]
    resp_gmv_old = resp_gmv_old[:,3]
    inv_resp_gmv_old = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_old[1:] = 1./(resp_gmv_old)[1:]
    inv_resp_gmv_TTEETE_old = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TTEETE_old[1:] = 1./(resp_gmv_TTEETE_old)[1:]
    inv_resp_gmv_TBEB_old = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TBEB_old[1:] = 1./(resp_gmv_TBEB_old)[1:]

    # SQE response, MH
    ests = ['T1T2', 'T2T1', 'EE', 'TE', 'ET', 'TB', 'BT', 'EB', 'BE']
    resps_original_crossilc = np.zeros((len(l),len(ests)), dtype=np.complex_)
    inv_resps_original_crossilc = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
    for i, est in enumerate(ests):
        resps_original_crossilc[:,i] = get_sqe_analytic_response(est,config,crossilc=True)
        inv_resps_original_crossilc[1:,i] = 1/(resps_original_crossilc)[1:,i]
    resp_original_crossilc = 0.5*resps_original_crossilc[:,0]+0.5*resps_original_crossilc[:,1]+np.sum(resps_original_crossilc[:,2:], axis=1)
    TT_resp_original_crossilc = 0.5*resps_original_crossilc[:,0]+0.5*resps_original_crossilc[:,1]
    inv_resp_original_crossilc = np.zeros_like(l,dtype=np.complex_); inv_resp_original_crossilc[1:] = 1/(resp_original_crossilc)[1:]
    inv_TT_resp_original_crossilc = np.zeros_like(l,dtype=np.complex_); inv_TT_resp_original_crossilc[1:] = 1/(TT_resp_original_crossilc)[1:]

    # SQE response, non-MH
    ests = ['TT', 'EE', 'TE', 'ET', 'TB', 'BT', 'EB', 'BE']
    resps_original = np.zeros((len(l),len(ests)), dtype=np.complex_)
    inv_resps_original = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
    for i, est in enumerate(ests):
        resps_original[:,i] = get_sqe_analytic_response(est,config,crossilc=False)
        inv_resps_original[1:,i] = 1/(resps_original)[1:,i]
    resp_original = np.sum(resps_original, axis=1)
    TT_resp_original = resps_original[:,0]
    inv_resp_original = np.zeros_like(l,dtype=np.complex_); inv_resp_original[1:] = 1/(resp_original)[1:]
    inv_TT_resp_original = np.zeros_like(l,dtype=np.complex_); inv_TT_resp_original[1:] = 1/(TT_resp_original)[1:]

    # Theory spectrum
    clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
    ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
    clkk = slpp * (l*(l+1))**2/4

    # Plot
    plt.figure(0)
    plt.clf()
    plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')

    #plt.plot(l, inv_resp_gmv_alt_crossilc * (l*(l+1))**2/4, color='darkblue', linestyle='-', label='1/R (GMV, MH)')
    #plt.plot(l, inv_resp_gmv_TTEETE_alt_crossilc * (l*(l+1))**2/4, color='forestgreen', linestyle='-', label='1/R (GMV, TTEETE, MH)')
    #plt.plot(l, inv_resp_gmv_TBEB_alt_crossilc * (l*(l+1))**2/4, color='blueviolet', linestyle='-', label='1/R (GMV, TBEB, MH)')

    #plt.plot(l, inv_resp_gmv * (l*(l+1))**2/4, color='cornflowerblue', linestyle='--', label='1/R (GMV, old)')
    #plt.plot(l, inv_resp_gmv_TTEETE * (l*(l+1))**2/4, color='lightgreen', linestyle='--', label='1/R (GMV, TTEETE, old)')
    #plt.plot(l, inv_resp_gmv_TBEB * (l*(l+1))**2/4, color='thistle', linestyle='--', label='1/R (GMV, TBEB, old)')

    plt.plot(l, inv_resp_original_crossilc * (l*(l+1))**2/4, color='darkblue', linestyle='-', label='1/R (SQE, MH)')
    plt.plot(l, inv_TT_resp_original_crossilc * (l*(l+1))**2/4, color='forestgreen', linestyle='-', label='1/R (SQE, TT, MH)')

    plt.plot(l, inv_resp_original * (l*(l+1))**2/4, color='cornflowerblue', linestyle='--', label='1/R (SQE, non-MH)')
    plt.plot(l, inv_TT_resp_original * (l*(l+1))**2/4, color='lightgreen', linestyle='--', label='1/R (SQE, TT, non-MH)')

    plt.ylabel("$C_\ell^{\kappa\kappa}$")
    plt.xlabel('$\ell$')
    plt.title(f'SQE 1/Response Comparison')
    plt.legend(loc='upper left', fontsize='x-small', bbox_to_anchor=(1,0.5))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,lmax)
    plt.ylim(8e-9,1e-5)

    plt.savefig(dir_out+f'/figs/sqe_resp_comparison.png',bbox_inches='tight')

def get_sim_response(est, config, gmv, sims=np.arange(40)+1,
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
        append = ''
        if gmv:
            append += f'_gmv_est{est}'
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

def get_gmv_analytic_response(est, config, alt=False, crossilc=False,
                              filename=None):
    '''
    ONLY for computing GMV analytic response, for comparing different inverse M.
    '''
    print(f'Computing analytic response for est {est}')
    lmax = config['lensrec']['Lmax']
    lmaxT = config['lensrec']['lmaxT']
    lmaxP = config['lensrec']['lmaxP']
    lmin = config['lensrec']['lminT']
    nside = config['lensrec']['nside']
    cltype = config['lensrec']['cltype']
    cls = config['cls']
    sl = {ee:config['cls'][cltype][ee] for ee in config['cls'][cltype].keys()}
    ell = np.arange(lmax+1,dtype=np.float_)
    dir_out = config['dir_out']

    if filename is None:
        append = ''
        if est=='all' or est=='TTEETE' or est=='TBEB':
            append += '_gmv_estall'
        else:
            append += f'_gmv_est{est}'
        if not crossilc:
            append += f'_lmaxT{lmaxT}_lmaxP{lmaxP}_lmin{lmin}_cltype{cltype}_notmh'
            if alt:
                append += '_numericalinv'
        else:
            append += f'_lmaxT{lmaxT}_lmaxP{lmaxP}_lmin{lmin}_cltype{cltype}_mh_numericalinv_nanzeros'
        filename = dir_out+f'/resp/an_resp{append}.npy'

    if os.path.isfile(filename):
        print('Loading from existing file!')
        R = np.load(filename)
    else:
        # File doesn't exist!
        # Load total Cls; these are for the MH test, obtained from alm2cl and averaging over 40 sims
        totalcls = np.load(dir_out+f'totalcls/totalcls_average_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_mh.npy')
        cltt1 = totalcls[:,4]; cltt2 = totalcls[:,5]; clee = totalcls[:,1]; clbb = totalcls[:,2]; clte = totalcls[:,3]
        if not crossilc:
            totalcls = np.vstack((cltt1,clee,clbb,clte)).T

        if crossilc:
            # GMV response using alternate inverse M matrix
            gmv_r = gmv_resp_alt.gmv_resp(config,cltype,totalcls,u=None,crossilc=crossilc,save_path=filename)
            if est == 'TTEETE' or est == 'TBEB' or est == 'all':
                gmv_r.calc_tvar()
            R = np.load(filename)
        elif not alt:
            # GMV response
            gmv_r = gmv_resp.gmv_resp(config,cltype,totalcls,u=None,crossilc=False,save_path=filename)
            if est == 'TTEETE' or est == 'TBEB' or est == 'all':
                gmv_r.calc_tvar()
            R = np.load(filename)
        else:
            # GMV response using alternate inverse M matrix
            gmv_r = gmv_resp_numericalinv.gmv_resp(config,cltype,totalcls,u=None,crossilc=False,save_path=filename)
            if est == 'TTEETE' or est == 'TBEB' or est == 'all':
                gmv_r.calc_tvar()
            R = np.load(filename)

    # If GMV, save file has columns L, TTEETE, TBEB, all
    if est == 'TTEETE' or est == 'TTEETEprf' or est == 'TTEETETTEETEprf':
        R = R[:,1]
    elif est == 'TBEB':
        R = R[:,2]
    elif est == 'all':
        R = R[:,3]

    return R

def get_sqe_analytic_response(est, config, crossilc=False,
                              filename=None):
    print(f'Computing analytic response for est {est}')
    lmax = config['lensrec']['Lmax']
    lmaxT = config['lensrec']['lmaxT']
    lmaxP = config['lensrec']['lmaxP']
    lmin = config['lensrec']['lminT']
    nside = config['lensrec']['nside']
    cltype = config['lensrec']['cltype']
    cls = config['cls']
    sl = {ee:config['cls'][cltype][ee] for ee in config['cls'][cltype].keys()}
    ell = np.arange(lmax+1,dtype=np.float_)
    dir_out = config['dir_out']

    if filename is None:
        append = f'_sqe_est{est}'
        if not crossilc:
            append += f'_lmaxT{lmaxT}_lmaxP{lmaxP}_lmin{lmin}_cltype{cltype}_notmh'
        else:
            append += f'_lmaxT{lmaxT}_lmaxP{lmaxP}_lmin{lmin}_cltype{cltype}_mh'
        filename = dir_out+f'/resp/an_resp{append}.npy'

    if os.path.isfile(filename):
        print('Loading from existing file!')
        R = np.load(filename)
    else:
        # File doesn't exist!
        # Load total Cls; these are for the MH test, obtained from alm2cl and averaging over 40 sims
        totalcls = np.load(dir_out+f'totalcls/totalcls_average_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_mh.npy')
        cltt1 = totalcls[:,4]; cltt2 = totalcls[:,5]; clee = totalcls[:,1]; clbb = totalcls[:,2]; clte = totalcls[:,3]
        if not crossilc:
            totalcls = np.vstack((cltt1,clee,clbb,clte)).T
        if crossilc:
            # Create 1/Nl filters
            flt1 = np.zeros(lmax+1); flt1[lmin:] = 1./cltt1[lmin:]
            flt2 = np.zeros(lmax+1); flt2[lmin:] = 1./cltt2[lmin:]
            fle = np.zeros(lmax+1); fle[lmin:] = 1./clee[lmin:]
            flb = np.zeros(lmax+1); flb[lmin:] = 1./clbb[lmin:]

            if est[:2] == 'T1': flX = flt1
            elif est[:2] == 'T2': flX = flt2
            elif est[0] == 'T': flX = flt1
            elif est[0] == 'E': flX = fle
            elif est[0] == 'B': flX = flb

            if est[2:4] == 'T1': flY = flt1
            elif est[2:4] == 'T2': flY = flt2
            elif est[1] == 'T': flY = flt1
            elif est[1] == 'E': flY = fle
            elif est[1] == 'B': flY = flb

            if est == 'T1T2' or  est == 'T2T1': est='TT'
            qeXY = weights.weights(est,cls[cltype],lmax,u=None)
            qeZA = None
            R = resp.fill_resp(qeXY,np.zeros(lmax+1, dtype=np.complex_),flX,flY,qeZA=qeZA)
            np.save(filename, R)
        else:
            # Create 1/Nl filters
            flt = np.zeros(lmax+1); flt[lmin:] = 1./cltt1[lmin:]
            fle = np.zeros(lmax+1); fle[lmin:] = 1./clee[lmin:]
            flb = np.zeros(lmax+1); flb[lmin:] = 1./clbb[lmin:]

            if est[0] == 'T': flX = flt
            elif est[0] == 'E': flX = fle
            elif est[0] == 'B': flX = flb

            if est[1] == 'T': flY = flt
            elif est[1] == 'E': flY = fle
            elif est[1] == 'B': flY = flb

            qeXY = weights.weights(est,cls[cltype],lmax,u=None)
            qeZA = None
            R = resp.fill_resp(qeXY,np.zeros(lmax+1, dtype=np.complex_),flX,flY,qeZA=qeZA)
            np.save(filename, R)

    return R

####################

analyze()
