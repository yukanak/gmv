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
import qest
import wignerd
import resp

def plot_profhrd_maps(config_file='test_yuka.yaml'):
    config = utils.parse_yaml(config_file)
    lmax = config['lensrec']['Lmax']
    lmin = config['lensrec']['lminT']
    lmaxT = config['lensrec']['lmaxT']
    lmaxP = config['lensrec']['lmaxP']
    nside = config['lensrec']['nside']
    dir_out = config['dir_out']
    l = np.arange(0,lmax+1)
    sim = 1
    noise_file = None
    fsky_corr = 1
    u=np.ones(4096+1, dtype=np.complex_)
    fluxlim = 0.200
    append = f'tsrc_fluxlim{fluxlim:.3f}'
    num = 100
    fwhm=0; nlev_t=0; nlev_p=0

    # Response
    ests = ['TT', 'EE', 'TE', 'TB', 'EB']
    resps_original = np.zeros((len(l),len(ests)), dtype=np.complex_)
    inv_resps_original = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
    for i, est in enumerate(ests):
        resps_original[:,i] = get_analytic_response(est,config,gmv=False,
                                                    fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,u=u,
                                                    noise_file=noise_file,fsky_corr=fsky_corr)
        #resps_original[:,i] = np.load(dir_out+f'/resp/sim_resp_sqe_est{est}_{num}sims_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_tsrc_fluxlim{fluxlim:.3f}.npy')
        inv_resps_original[1:,i] = 1/(resps_original)[1:,i]
    resp_original = resps_original[:,0]+resps_original[:,1]+2*resps_original[:,2]+2*resps_original[:,3]+2*resps_original[:,4]
    inv_resp_original = np.zeros_like(l,dtype=np.complex_); inv_resp_original[1:] = 1/(resp_original)[1:]

    # Get the profile response and weight
    # SQE
    resp_original_TT_ss = get_analytic_response('TTprf',config,gmv=False,
                                                fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,u=u,
                                                noise_file=noise_file,fsky_corr=fsky_corr)
    resp_original_TT_sk = get_analytic_response('TTTTprf',config,gmv=False,
                                                fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,u=u,
                                                noise_file=noise_file,fsky_corr=fsky_corr)
    weight_original = -1 * resp_original_TT_sk / resp_original_TT_ss
    resp_original_hrd = resp_original + weight_original*resp_original_TT_sk # Equivalent to resp_original_TT (hardened) + np.sum(resps_original[:,1:], axis=1)
    resp_original_TT_hrd = resps_original[:,0] + weight_original*resp_original_TT_sk
    #resp_original_hrd = np.load(f'/oak/stanford/orgs/kipac/users/yukanaka/outputs/resp/sim_resp_sqe_estall_hrd_{num}sims_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_tsrc_fluxlim{fluxlim:.3f}.npy')
    #resp_original_TT_hrd = np.load(f'/oak/stanford/orgs/kipac/users/yukanaka/outputs/resp/sim_resp_sqe_estTT_hrd_{num}sims_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_tsrc_fluxlim{fluxlim:.3f}.npy')
    inv_resp_original_hrd = np.zeros_like(l,dtype=np.complex_); inv_resp_original_hrd[1:] = 1/(resp_original_hrd)[1:]
    inv_resp_original_TT_hrd = np.zeros_like(l,dtype=np.complex_); inv_resp_original_TT_hrd[1:] = 1/(resp_original_TT_hrd)[1:]

    plms_original = np.zeros((len(np.load(dir_out+f'/outputs_lensing19-20_with_foregrounds_no_noise/plm_TT_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')),5), dtype=np.complex_)
    for i, est in enumerate(ests):
        plms_original[:,i] = np.load(dir_out+f'/outputs_lensing19-20_with_foregrounds_no_noise/plm_{est}_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
    plm_original = plms_original[:,0]+plms_original[:,1]+2*plms_original[:,2]+2*plms_original[:,3]+2*plms_original[:,4]

    # Harden!
    glm_prf_TT = np.load(dir_out+f'/outputs_lensing19-20_with_foregrounds_no_noise/plm_TTprf_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
    plm_original_TT_hrd = plms_original[:,0] + hp.almxfl(glm_prf_TT, weight_original)
    plm_original_hrd = plm_original_TT_hrd + plms_original[:,1]+2*plms_original[:,2]+2*plms_original[:,3]+2*plms_original[:,4]

    # Contaminated reconstructed kappa map
    plm_TT = plms_original[:,0].copy()
    #plm_TT = hp.almxfl(plms_original[:,0],inv_resps_original[:,0])
    #plm_TT = alm_cutlmax(plm_TT,300)
    phi = hp.alm2map(hp.almxfl(plm_TT,(l*(l+1))/2),nside=nside)
    plm_sqe = hp.almxfl(plm_original,inv_resp_original)
    plm_sqe = alm_cutlmax(plm_sqe,300)
    phi_total = hp.alm2map(hp.almxfl(plm_sqe,(l*(l+1))/2),nside=nside)

    # Source estimator output
    glm_prf_TT_weighted = -1*hp.almxfl(glm_prf_TT, weight_original)
    #glm_prf_TT_weighted = hp.almxfl(glm_prf_TT_weighted,inv_resp_original_TT_hrd)
    #glm_prf_TT = alm_cutlmax(glm_prf_TT,300)
    #source = hp.alm2map(hp.almxfl(glm_prf_TT_weighted,(l*(l+1))/2),nside=nside)
    source = hp.alm2map(hp.almxfl(glm_prf_TT,(l*(l+1))/2),nside=nside)

    # Bias-hardened kappa map
    #plm_TT_hrd = hp.almxfl(plm_original_TT_hrd,inv_resp_original_TT_hrd)
    plm_TT_hrd = plm_original_TT_hrd.copy()
    #plm_TT_hrd = alm_cutlmax(plm_TT_hrd,300)
    phi_bh = hp.alm2map(hp.almxfl(plm_TT_hrd,(l*(l+1))/2),nside=nside)
    plm_sqe_hrd = hp.almxfl(plm_original_hrd,inv_resp_original_hrd)
    plm_sqe_hrd = alm_cutlmax(plm_sqe_hrd,300)
    phi_bh_total = hp.alm2map(hp.almxfl(plm_sqe_hrd,(l*(l+1))/2),nside=nside)

    # Input
    input_plm = hp.read_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/phi/phi_lmax_{lmax}/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_phi1_seed{sim}_lmax{lmax}.alm')
    input_plm = alm_cutlmax(input_plm,300)
    input_phi = hp.alm2map(hp.almxfl(input_plm,(l*(l+1))/2),nside=nside)

    # Plot
    scale = 0.2
    plt.figure(0)
    plt.clf()
    hp.gnomview(phi,title='Reconstructed Kappa Map',rot=[0,-60],notext=True,xsize=300,ysize=300,reso=3,)#min=-1*scale*3,max=scale*3)#,unit="uK")
    #plt.savefig(dir_out+f'/figs/poisson_profhrd_reconstructed_kappa_map.png',bbox_inches='tight')
    plt.savefig(dir_out+f'/figs/gaussian_profhrd_reconstructed_kappa_map.png',bbox_inches='tight')

    plt.clf()
    hp.gnomview(source,title='Reconstructed Source Map',rot=[0,-60],notext=True,xsize=300,ysize=300,reso=3,)#min=-1*scale*3,max=scale*3)
    #plt.savefig(dir_out+f'/figs/poisson_profhrd_reconstructed_source_map.png',bbox_inches='tight')
    plt.savefig(dir_out+f'/figs/gaussian_profhrd_reconstructed_source_map.png',bbox_inches='tight')

    plt.clf()
    hp.gnomview(phi_bh_total,title='Bias-Hardened Kappa Map',rot=[0,-60],notext=True,xsize=300,ysize=300,reso=3,)#min=-1*scale,max=scale)
    #plt.savefig(dir_out+f'/figs/poisson_profhrd_bias_hardened_kappa_map.png',bbox_inches='tight')
    plt.savefig(dir_out+f'/figs/gaussian_profhrd_bias_hardened_kappa_map.png',bbox_inches='tight')

    plt.clf()
    hp.gnomview(input_phi,title='Input Kappa Map',rot=[0,-60],notext=True,xsize=300,ysize=300,reso=3,min=-1*scale,max=scale)
    #plt.savefig(dir_out+f'/figs/poisson_profhrd_input_kappa_map.png',bbox_inches='tight')
    plt.savefig(dir_out+f'/figs/gaussian_profhrd_input_kappa_map.png',bbox_inches='tight')

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

"""
def get_analytic_response(est, config, gmv,
                          fwhm=0, nlev_t=0, nlev_p=0, u=None,
                          noise_file=None, fsky_corr=1,
                          filename=None,):
    '''
    If gmv, est should be 'TTEETE'/'TBEB'/'all'/'TTEETEprf'/'TTEETETTEETEprf'.
    If not gmv, assume sqe and est should be 'TT'/'EE'/'TE'/'TB'/'EB'/'TTprf'/'TTTTprf'.
    Also, we are taking lmax values from the config file, so make sure those are right.
    Note we are also assuming that if we are loading from a noise file, we won't also add
    noise according to nlev_t and nlev_p.
    '''
    print(f'Computing analytic response for est {est}')
    lmax = config['lensrec']['Lmax']
    lmin = config['lensrec']['lminT']
    lmaxT = config['lensrec']['lmaxT']
    lmaxP = config['lensrec']['lmaxP']
    nside = config['lensrec']['nside']
    dir_out = config['dir_out']
    cltype = config['lensrec']['cltype']
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
        filename = f'/oak/stanford/orgs/kipac/users/yukanaka/outputs/resp/an_resp{append}.npy'

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
        #if gmv:
        # If lmaxT != lmaxP, we add artificial noise in TT for ell > lmaxT
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
"""
