#!/usr/bin/env python3
import numpy as np
import pickle
import healpy as hp
import camb
import os, sys
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import healqest_utils as utils
import matplotlib.pyplot as plt
import weights
import wignerd
import resp

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def get_analytic_response(est, config, gmv=True, append='standard', filename=None):
    '''
    If not gmv, assume sqe and est should be 'TT'/'EE'/'TE'/'TB'/'EB'.
    Also, we are taking lmax values from the config file, so make sure those are right.
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
        fn = ''
        if gmv and (est=='all' or est=='TTEETE' or est=='TBEB'):
            fn += '_gmv_estall'
        elif gmv:
            fn += f'_gmv_est{est}'
        else:
            fn += f'_sqe_est{est}'
        fn += f'_lmaxT{lmaxT}_lmaxP{lmaxP}_lmin{lmin}_cltype{cltype}_agora_{append}'
        filename = dir_out+f'/resp/an_resp{fn}.npy'

    if os.path.isfile(filename):
        print('Loading from existing file!')
        R = np.load(filename)
    else:
        # File doesn't exist!
        # Load total Cls
        if append == 'standard':
            totalcls = np.load(dir_out+f'totalcls/totalcls_average_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_agora_standard.npy')
            cltt = totalcls[:,0]; clee = totalcls[:,1]; clbb = totalcls[:,2]; clte = totalcls[:,3]
        else:
            pass

        if not gmv:
            # Create 1/Nl filters
            flt = np.zeros(lmax+1); flt[lmin:] = 1./cltt[lmin:]
            fle = np.zeros(lmax+1); fle[lmin:] = 1./clee[lmin:]
            flb = np.zeros(lmax+1); flb[lmin:] = 1./clbb[lmin:]

            if est[0] == 'T': flX = flt
            elif est[0] == 'E': flX = fle
            elif est[0] == 'B': flX = flb

            if est[1] == 'T': flY = flt
            elif est[1] == 'E': flY = fle
            elif est[1] == 'B': flY = flb

            qeXY = weights.weights(est,cls[cltype],lmax)
            qeZA = None
            R = resp.fill_resp(qeXY,np.zeros(lmax+1, dtype=np.complex_),flX,flY,qeZA=qeZA)
            np.save(filename, R)

        else:
            # GMV response
            print('Doing the 1/Dl for GMV...')
            invDl = np.zeros(lmax+1, dtype=np.complex_)
            invDl[lmin:] = 1./(cltt[lmin:]*clee[lmin:] - clte[lmin:]**2)
            flb = np.zeros(lmax+1); flb[lmin:] = 1./clbb[lmin:]

            qeXY = weights.weights(est,cls[cltype],lmax)
            #qeZA = None
            R = 0
            if est == 'TT':
                # Eq. A13 in my paper draft
                flX = clee * invDl; flY = clee * invDl; qeZA = weights.weights('TT',cls[cltype],lmax); R += resp.fill_resp(qeXY,np.zeros(lmax+1, dtype=np.complex_),flX,flY,qeZA=qeZA)
                flX = clee * invDl; flY = clte * invDl; qeZA = weights.weights('TE',cls[cltype],lmax); R += -1*resp.fill_resp(qeXY,np.zeros(lmax+1, dtype=np.complex_),flX,flY,qeZA=qeZA)
                flX = clte * invDl; flY = clee * invDl; qeZA = weights.weights('TE',cls[cltype],lmax); R += -1*resp.fill_resp(qeXY,np.zeros(lmax+1, dtype=np.complex_),flX,flY,qeZA=qeZA)
                flX = clte * invDl; flY = clte * invDl; qeZA = weights.weights('EE',cls[cltype],lmax); R += resp.fill_resp(qeXY,np.zeros(lmax+1, dtype=np.complex_),flX,flY,qeZA=qeZA)
            elif est == 'EE':
                flX = cltt * invDl; flY = cltt * invDl; qeZA = weights.weights('EE',cls[cltype],lmax); R += resp.fill_resp(qeXY,np.zeros(lmax+1, dtype=np.complex_),flX,flY,qeZA=qeZA)
                flX = cltt * invDl; flY = clte * invDl; qeZA = weights.weights('TE',cls[cltype],lmax); R += -1*resp.fill_resp(qeXY,np.zeros(lmax+1, dtype=np.complex_),flX,flY,qeZA=qeZA)
                flX = clte * invDl; flY = cltt * invDl; qeZA = weights.weights('TE',cls[cltype],lmax); R += -1*resp.fill_resp(qeXY,np.zeros(lmax+1, dtype=np.complex_),flX,flY,qeZA=qeZA)
                flX = clte * invDl; flY = clte * invDl; qeZA = weights.weights('TT',cls[cltype],lmax); R += resp.fill_resp(qeXY,np.zeros(lmax+1, dtype=np.complex_),flX,flY,qeZA=qeZA)
            elif est == 'TE':
                flX = clee * invDl; flY = cltt * invDl; qeZA = weights.weights('TE',cls[cltype],lmax); R += resp.fill_resp(qeXY,np.zeros(lmax+1, dtype=np.complex_),flX,flY,qeZA=qeZA)
                flX = clee * invDl; flY = clte * invDl; qeZA = weights.weights('TT',cls[cltype],lmax); R += -1*resp.fill_resp(qeXY,np.zeros(lmax+1, dtype=np.complex_),flX,flY,qeZA=qeZA)
                flX = clte * invDl; flY = cltt * invDl; qeZA = weights.weights('EE',cls[cltype],lmax); R += -1*resp.fill_resp(qeXY,np.zeros(lmax+1, dtype=np.complex_),flX,flY,qeZA=qeZA)
                flX = clte * invDl; flY = clte * invDl; qeZA = weights.weights('TE',cls[cltype],lmax); R += resp.fill_resp(qeXY,np.zeros(lmax+1, dtype=np.complex_),flX,flY,qeZA=qeZA)
            elif est == 'ET':
                flX = cltt * invDl; flY = clee * invDl; qeZA = weights.weights('TE',cls[cltype],lmax); R += resp.fill_resp(qeXY,np.zeros(lmax+1, dtype=np.complex_),flX,flY,qeZA=qeZA)
                flX = cltt * invDl; flY = clte * invDl; qeZA = weights.weights('EE',cls[cltype],lmax); R += -1*resp.fill_resp(qeXY,np.zeros(lmax+1, dtype=np.complex_),flX,flY,qeZA=qeZA)
                flX = clte * invDl; flY = clee * invDl; qeZA = weights.weights('TT',cls[cltype],lmax); R += -1*resp.fill_resp(qeXY,np.zeros(lmax+1, dtype=np.complex_),flX,flY,qeZA=qeZA)
                flX = clte * invDl; flY = clte * invDl; qeZA = weights.weights('TE',cls[cltype],lmax); R += resp.fill_resp(qeXY,np.zeros(lmax+1, dtype=np.complex_),flX,flY,qeZA=qeZA)
            elif est == 'TB':
                flX = clee * invDl; flY = flb; qeZA = weights.weights('TB',cls[cltype],lmax); R += resp.fill_resp(qeXY,np.zeros(lmax+1, dtype=np.complex_),flX,flY,qeZA=qeZA)
                flX = clte * invDl; flY = flb; qeZA = weights.weights('EB',cls[cltype],lmax); R += -1*resp.fill_resp(qeXY,np.zeros(lmax+1, dtype=np.complex_),flX,flY,qeZA=qeZA)
            elif est == 'BT':
                flX = flb; flY = clee * invDl; qeZA = weights.weights('TB',cls[cltype],lmax); R += resp.fill_resp(qeXY,np.zeros(lmax+1, dtype=np.complex_),flX,flY,qeZA=qeZA)
                flX = flb; flY = clte * invDl; qeZA = weights.weights('EB',cls[cltype],lmax); R += -1*resp.fill_resp(qeXY,np.zeros(lmax+1, dtype=np.complex_),flX,flY,qeZA=qeZA)
            elif est == 'EB':
                flX = cltt * invDl; flY = flb; qeZA = weights.weights('EB',cls[cltype],lmax); R += resp.fill_resp(qeXY,np.zeros(lmax+1, dtype=np.complex_),flX,flY,qeZA=qeZA)
                flX = clte * invDl; flY = flb; qeZA = weights.weights('TB',cls[cltype],lmax); R += -1*resp.fill_resp(qeXY,np.zeros(lmax+1, dtype=np.complex_),flX,flY,qeZA=qeZA)
            elif est == 'BE':
                flX = flb; flY = cltt * invDl; qeZA = weights.weights('EB',cls[cltype],lmax); R += resp.fill_resp(qeXY,np.zeros(lmax+1, dtype=np.complex_),flX,flY,qeZA=qeZA)
                flX = flb; flY = clte * invDl; qeZA = weights.weights('TB',cls[cltype],lmax); R += -1*resp.fill_resp(qeXY,np.zeros(lmax+1, dtype=np.complex_),flX,flY,qeZA=qeZA)
            np.save(filename, R)

    return R

def compare():
    '''
    cinv = False NOT IMPLEMENTED.
    '''
    config_file = 'test_yuka_lmaxT3500.yaml'
    cinv = True
    config = utils.parse_yaml(config_file)
    lmax = config['lensrec']['Lmax']
    lmin = config['lensrec']['lminT']
    lmaxT = config['lensrec']['lmaxT']
    lmaxP = config['lensrec']['lmaxP']
    nside = config['lensrec']['nside']
    dir_out = config['dir_out']
    l = np.arange(0,lmax+1)
    append = 'standard'
    if not cinv:
        print('NOT IMPLEMENTED')
        return None

    # Full sky, no masking
    # Sims are signal + Agora foregrounds + SPT3G 2019-2020 noise levels frequency correlated noise realizations generated from frequency separated noise spectra

    #==================== NO ====================#
    # Standard SQE
    # lmaxT = 3000
    filename = dir_out+f'/n0/n0_249simpairs_healqest_sqe_lmaxT3000_lmaxP4096_nside2048_agora_standard_resp_from_sims.pkl'
    n0_sqe_3000 = pickle.load(open(filename,'rb'))
    n0_sqe_3000_total = n0_sqe_3000['total'] * (l*(l+1))**2/4
    # lmaxT = 3500
    filename = dir_out+f'/n0/n0_249simpairs_healqest_sqe_lmaxT3500_lmaxP4096_nside2048_agora_standard_resp_from_sims.pkl'
    n0_sqe_3500 = pickle.load(open(filename,'rb'))
    n0_sqe_3500_total = n0_sqe_3500['total'] * (l*(l+1))**2/4

    # Standard GMV
    # lmaxT = 3000
    filename = dir_out+f'/n0/n0_249simpairs_healqest_gmv_cinv_lmaxT3000_lmaxP4096_nside2048_agora_standard_resp_from_sims.pkl'
    n0_standard_3000 = pickle.load(open(filename,'rb'))
    n0_standard_3000_total = n0_standard_3000['total'] * (l*(l+1))**2/4
    # lmaxT = 3500
    filename = dir_out+f'/n0/n0_249simpairs_healqest_gmv_cinv_lmaxT3500_lmaxP4096_nside2048_agora_standard_resp_from_sims.pkl'
    n0_standard_3500 = pickle.load(open(filename,'rb'))
    n0_standard_3500_total = n0_standard_3500['total'] * (l*(l+1))**2/4
    # lmaxT = 4000
    filename = dir_out+f'/n0/n0_249simpairs_healqest_gmv_cinv_lmaxT4000_lmaxP4096_nside2048_agora_standard_resp_from_sims.pkl'
    n0_standard_4000 = pickle.load(open(filename,'rb'))
    n0_standard_4000_total = n0_standard_4000['total'] * (l*(l+1))**2/4
    # lmaxT = 3500, RDN0
    filename = dir_out+f'/n0/rdn0_249simpairs_healqest_gmv_cinv_lmaxT3500_lmaxP4096_nside2048_agora_standard_resp_from_sims.pkl'
    rdn0_standard_3500 = pickle.load(open(filename,'rb'))
    rdn0_standard_3500_total = rdn0_standard_3500 * (l*(l+1))**2/4
    #============================================#

    #==================== RESPONSE ====================#
    ests = ['TT', 'EE', 'TE', 'ET', 'TB', 'BT', 'EB', 'BE']
    # Get sim response
    resps = np.zeros((len(l),len(ests)), dtype=np.complex_)
    resps_sqe = np.zeros((len(l),len(ests)), dtype=np.complex_)
    inv_resps = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
    inv_resps_sqe = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
    for i, est in enumerate(ests):
        resps[:,i] = np.load(dir_out+f'/resp/sim_resp_250sims_gmv_cinv_est{est}_lmaxT{lmaxT}_lmaxP{lmaxP}_lmin{lmin}_cltypelcmb_agora_standard.npy')
        resps_sqe[:,i] = np.load(dir_out+f'/resp/sim_resp_250sims_sqe_est{est}_lmaxT{lmaxT}_lmaxP{lmaxP}_lmin{lmin}_cltypelcmb_agora_standard.npy')
        inv_resps[1:,i] = 1/(resps)[1:,i]
        inv_resps_sqe[1:,i] = 1/(resps_sqe)[1:,i]
    resp = np.sum(resps, axis=1) #resp = 0.5*np.sum(resps[:,:4], axis=1)+np.sum(resps[:,4:], axis=1)
    resp_sqe = np.sum(resps_sqe, axis=1)
    inv_resp = np.zeros_like(l,dtype=np.complex_); inv_resp[1:] = 1/(resp)[1:]
    inv_resp_sqe = np.zeros_like(l,dtype=np.complex_); inv_resp_sqe[1:] = 1/(resp_sqe)[1:]
    sim_resp_TTEETE = np.load(dir_out+f'/resp/sim_resp_250sims_gmv_cinv_estTTEETE_lmaxT{lmaxT}_lmaxP{lmaxP}_lmin{lmin}_cltypelcmb_agora_standard.npy')
    inv_resp_TTEETE = np.zeros_like(l,dtype=np.complex_); inv_resp_TTEETE[1:] = 1/(sim_resp_TTEETE)[1:]
    resp_sqe_TTEETE = np.sum(resps_sqe[:,:4], axis=1)
    inv_resp_sqe_TTEETE = np.zeros_like(l,dtype=np.complex_); inv_resp_sqe_TTEETE[1:] = 1/(resp_sqe_TTEETE)[1:]

    # Get analytic response
    an_resps = np.zeros((len(l),len(ests)), dtype=np.complex_)
    inv_an_resps = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
    for i, est in enumerate(ests):
        an_resps[:,i] = get_analytic_response(est,config,append=append)
        inv_an_resps[1:,i] = 1/(an_resps)[1:,i]
    an_resp = np.sum(an_resps, axis=1) #an_resp = 0.5*np.sum(an_resps[:,:4], axis=1)+np.sum(an_resps[:,4:], axis=1)
    inv_an_resp = np.zeros_like(l,dtype=np.complex_); inv_an_resp[1:] = 1/(an_resp)[1:]

    ests = ['TT', 'EE', 'TE', 'ET']
    an_resp_TTEETE = 0
    for i, est in enumerate(ests):
        an_resp_TTEETE += get_analytic_response(est,config,append=append)
    inv_an_resp_TTEETE = np.zeros_like(l,dtype=np.complex_); inv_an_resp_TTEETE[1:] = 1/(an_resp_TTEETE)[1:]
    #==================================================#

    # Theory spectrum
    clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
    ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
    clkk = slpp * (l*(l+1))**2/4

    # Plot
    plt.clf()
    plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')
    #plt.plot(l, n0_standard_3500_total, color='forestgreen', alpha=0.8, linestyle='-',label='N0 Standard GMV')
    plt.plot(l, inv_resp * (l*(l+1))**2/4, color='darkblue', alpha=0.8, linestyle='-',label='Full-Sky Test Sim Response Total')
    plt.plot(l, inv_an_resp * (l*(l+1))**2/4, color='lightsteelblue', alpha=0.8, linestyle='-',label='Analytic Response Total')
    plt.plot(l, inv_resps[:,0] * (l*(l+1))**2/4, color='firebrick', alpha=0.8, linestyle='-',label='Full-Sky Test Sim Response TT')
    plt.plot(l, inv_an_resps[:,0] * (l*(l+1))**2/4, color='pink', alpha=0.8, linestyle='-',label='Analytic Response TT')
    plt.plot(l, inv_resps[:,1] * (l*(l+1))**2/4, color='forestgreen', alpha=0.8, linestyle='-',label='Full-Sky Test Sim Response EE')
    plt.plot(l, inv_an_resps[:,1] * (l*(l+1))**2/4, color='lightgreen', alpha=0.8, linestyle='-',label='Analytic Response EE')
    plt.plot(l, inv_resps[:,2] * (l*(l+1))**2/4, color='darkorange', alpha=0.8, linestyle='-',label='Full-Sky Test Sim Response TE')
    plt.plot(l, inv_an_resps[:,2] * (l*(l+1))**2/4, color='bisque', alpha=0.8, linestyle='-',label='Analytic Response TE')
    #plt.plot(l, inv_resps[:,3] * (l*(l+1))**2/4, color='', alpha=0.8, linestyle='-',label='Full-Sky Test Sim Response ET')
    #plt.plot(l, inv_an_resps[:,3] * (l*(l+1))**2/4, color='', alpha=0.8, linestyle='-',label='Analytic Response ET')
    plt.plot(l, inv_resps[:,4] * (l*(l+1))**2/4, color='blueviolet', alpha=0.8, linestyle='-',label='Full-Sky Test Sim Response TB')
    plt.plot(l, inv_an_resps[:,4] * (l*(l+1))**2/4, color='thistle', alpha=0.8, linestyle='-',label='Analytic Response TB')
    #plt.plot(l, inv_resps[:,5] * (l*(l+1))**2/4, color='', alpha=0.8, linestyle='-',label='Full-Sky Test Sim Response BT')
    #plt.plot(l, inv_an_resps[:,5] * (l*(l+1))**2/4, color='', alpha=0.8, linestyle='-',label='Analytic Response BT')
    plt.plot(l, inv_resps[:,6] * (l*(l+1))**2/4, color='gold', alpha=0.8, linestyle='-',label='Full-Sky Test Sim Response EB')
    plt.plot(l, inv_an_resps[:,6] * (l*(l+1))**2/4, color='palegoldenrod', alpha=0.8, linestyle='-',label='Analytic Response EB')
    #plt.plot(l, inv_resps[:,7] * (l*(l+1))**2/4, color='', alpha=0.8, linestyle='-',label='Full-Sky Test Sim Response BE')
    #plt.plot(l, inv_an_resps[:,7] * (l*(l+1))**2/4, color='', alpha=0.8, linestyle='-',label='Analytic Response BE')
    plt.xlabel('$\ell$')
    plt.title(f'GMV Response Comparison, Standard')
    #plt.legend(loc='upper left', fontsize='small')
    plt.legend(loc='upper left', fontsize='x-small', bbox_to_anchor=(1,0.5))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,lmax)
    plt.ylim(6e-9,1e-4)
    plt.savefig(dir_out+f'/figs/check_response_gmv.png',bbox_inches='tight')

    plt.clf()
    plt.axhline(y=1, color='k', linestyle='--')
    #plt.plot(l, inv_resp/inv_an_resp, color='darkblue', alpha=0.8, linestyle='-',label='Ratio Total')
    plt.plot(l, inv_resps[:,0]/inv_an_resps[:,0], color='firebrick', alpha=0.8, linestyle='-',label='Ratio TT')
    plt.plot(l, inv_resps[:,1]/inv_an_resps[:,1], color='forestgreen', alpha=0.8, linestyle='-',label='Ratio EE')
    plt.plot(l, inv_resps[:,2]/inv_an_resps[:,2], color='darkorange', alpha=0.8, linestyle='-',label='Ratio TE')
    plt.plot(l, inv_resp_TTEETE/(inv_an_resp_TTEETE), color='darkblue', alpha=0.8, linestyle='-',label='Ratio TTEETE')
    #plt.plot(l, inv_resps[:,4]/inv_an_resps[:,4], color='blueviolet', alpha=0.8, linestyle='-',label='Ratio TB')
    #plt.plot(l, inv_resps[:,6]/inv_an_resps[:,6], color='gold', alpha=0.8, linestyle='-',label='Ratio EB')
    plt.xlabel('$\ell$')
    plt.title(f'GMV Response Comparison, Standard, Sim 1/R / Analytic 1/R')
    plt.legend(loc='upper left', fontsize='small')
    plt.xscale('log')
    plt.ylim(0,2)
    plt.xlim(10,lmax)
    plt.savefig(dir_out+f'/figs/check_response_gmv_ratio.png',bbox_inches='tight')

    '''
    plt.clf()
    # Check input kappa just in case
    # Input kappa
    klm = hp.read_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_spt3g_input_klm_lmax{lmax}.fits')
    input_clkk = hp.alm2cl(klm,klm,lmax=lmax)
    # Input kappa sims
    mean_input_clkk = 0
    for sim in np.arange(250)+1:
        input_plm = hp.read_alm(f'/oak/stanford/orgs/kipac//users/yukanaka/lensing19-20/inputcmb/phi/phi_lmax_{lmax}/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_phi1_seed{sim}_lmax{lmax}.alm')
        mean_input_clkk += hp.alm2cl(input_plm) * (l*(l+1))**2/4
    mean_input_clkk /= 250
    input_clkk = moving_average(input_clkk, window_size=50)
    mean_input_clkk = moving_average(mean_input_clkk, window_size=50)
    plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')
    plt.plot(l, input_clkk, color='darkblue', alpha=0.8, linestyle='-',label='Input kappa spectrum Agora')
    plt.plot(l, mean_input_clkk, color='firebrick', alpha=0.8, linestyle='-', label='Mean of input kappa sims')
    plt.xlabel('$\ell$')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,lmax)
    plt.ylim(1e-8,1e-7)
    plt.savefig(dir_out+f'/figs/test_input_kappa.png',bbox_inches='tight')

    plt.clf()
    ratio_input_kappa_spec_agora = input_clkk/clkk
    ratio_mean_from_sims = mean_input_clkk/clkk
    # Ratios with error bars
    plt.axhline(y=1, color='k', linestyle='--')
    plt.plot(l, ratio_input_kappa_spec_agora, color='darkblue', alpha=0.8, linestyle='-',label='Ratio Kappa Spec from Agora Map / Fiducial')
    plt.plot(l, ratio_mean_from_sims, color='firebrick', alpha=0.8, linestyle='-',label='Ratio Mean of Input Kappa Sims / Fiducial')
    plt.xlabel('$\ell$')
    plt.legend(loc='upper left', fontsize='small')
    plt.xscale('log')
    #plt.ylim(0.9,1.4)
    plt.xlim(10,lmax)
    plt.savefig(dir_out+f'/figs/test_input_kappa_ratio.png',bbox_inches='tight')
    '''

#compare()
