#!/usr/bin/env python3
from scipy.ndimage import gaussian_filter1d as gf1
import numpy as np
import pickle
import healpy as hp
import camb
import os, sys
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
from astropy.io import fits
import gmv_resp
import utils
import matplotlib.pyplot as plt
import weights
import qest
import wignerd
import resp




def get_n1(sims=np.arange(10)+101,qetype='gmv',fluxlim=0.200,
           lmax=4096,nside=8192,u=np.ones(4097,dtype=np.complex_),
           config_file='profhrd_yuka.yaml',dir_out='/scratch/users/yukanaka/gmv/'):
    '''
    Get N1 bias. qetype should be 'gmv' or 'original'.
    '''
    num = len(sims)
    l = np.arange(0,lmax+1)
    config = utils.parse_yaml(config_file)
    append = 'cmbonly'
    clfile = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat'
    #filename = f'/scratch/users/yukanaka/gmv/n0/n1_{num}simpairs_healqest_{qetype}_lmax{lmax}_nside{nside}_{append}.npy'
    filename = f'/scratch/users/yukanaka/gmv/n0/n1_{num}simpairs_healqest_{qetype}_lmax{lmax}_nside{nside}_{append}.pkl'
    if os.path.isfile(filename):
        n1 = pickle.load(open(filename,'rb'))
        #n1 = np.load(filename,allow_pickle=True)
    elif qetype == 'gmv':
        gmv_resp_data = np.genfromtxt('gmv_resp/True_variance_individual_custom_lmin2.0_lmaxT4096_lmaxP4096_beam0_noise0_50.txt')
        inv_resp_gmv = gmv_resp_data[:,3] / l**2
        inv_resp_gmv_A = gmv_resp_data[:,1] / l**2
        inv_resp_gmv_B = gmv_resp_data[:,2] / l**2
        # Abhi's code calculates the reconstruction noise for d field rather than phi field, see GMV_QE.py, line 292 for example
        resp_gmv = 1/inv_resp_gmv
        resp_gmv_A = 1/inv_resp_gmv_A
        resp_gmv_B = 1/inv_resp_gmv_B
        n1 = {'total':0, 'TTEETE':0, 'TBEB':0}
        for i, sim in enumerate(sims):
            # These are reconstructions using sims that were lensed with the same phi but different CMB realizations, no foregrounds
            plm_gmv_ij = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim}_seed2_{sim}_lmax{lmax}_nside{nside}_cmbonly_phi1_tqu1tqu2.npy')
            plm_gmv_A_ij = np.load(dir_out+f'/plm_TTEETE_healqest_gmv_seed1_{sim}_seed2_{sim}_lmax{lmax}_nside{nside}_cmbonly_phi1_tqu1tqu2.npy')
            plm_gmv_B_ij = np.load(dir_out+f'/plm_TBEB_healqest_gmv_seed1_{sim}_seed2_{sim}_lmax{lmax}_nside{nside}_cmbonly_phi1_tqu1tqu2.npy')
            # Response correct GMV
            # N is 1/R
            plm_all_ij = hp.almxfl(plm_gmv_ij,inv_resp_gmv)
            plm_A_ij = hp.almxfl(plm_gmv_A_ij,inv_resp_gmv_A)
            plm_B_ij = hp.almxfl(plm_gmv_B_ij,inv_resp_gmv_B)
            # Get ij auto spectra <ijij>
            auto = hp.alm2cl(plm_all_ij, plm_all_ij, lmax=lmax)
            auto_A = hp.alm2cl(plm_A_ij, plm_A_ij, lmax=lmax)
            auto_B = hp.alm2cl(plm_B_ij, plm_B_ij, lmax=lmax)
            # Now get the ji sims
            plm_gmv_ji = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim}_seed2_{sim}_lmax{lmax}_nside{nside}_cmbonly_phi1_tqu2tqu1.npy')
            plm_gmv_A_ji = np.load(dir_out+f'/plm_TTEETE_healqest_gmv_seed1_{sim}_seed2_{sim}_lmax{lmax}_nside{nside}_cmbonly_phi1_tqu2tqu1.npy')
            plm_gmv_B_ji = np.load(dir_out+f'/plm_TBEB_healqest_gmv_seed1_{sim}_seed2_{sim}_lmax{lmax}_nside{nside}_cmbonly_phi1_tqu2tqu1.npy')
            # Response correct GMV
            # N is 1/R
            plm_all_ji = hp.almxfl(plm_gmv_ji,inv_resp_gmv)
            plm_A_ji = hp.almxfl(plm_gmv_A_ji,inv_resp_gmv_A)
            plm_B_ji = hp.almxfl(plm_gmv_B_ji,inv_resp_gmv_B)
            # Get cross spectra <ijji>
            cross = hp.alm2cl(plm_all_ij, plm_all_ji, lmax=lmax)
            cross_A = hp.alm2cl(plm_A_ij, plm_A_ji, lmax=lmax)
            cross_B = hp.alm2cl(plm_B_ij, plm_B_ji, lmax=lmax)
            n1['total'] += auto + cross
            n1['TTEETE'] += auto_A + cross_A
            n1['TBEB'] += auto_B + cross_B
        n1['total'] *= 1/num
        n1['TTEETE'] *= 1/num
        n1['TBEB'] *= 1/num
        n0 = get_n0(sims=sims,qetype=qetype,fluxlim=fluxlim,lmax=lmax,
                    nside=nside,u=u,config_file=config_file,dir_out=dir_out,
                    withfg=False)
        n1['total'] -= n0['total']
        n1['TTEETE'] -= n0['TTEETE']
        n1['TBEB'] -= n0['TBEB']
        #np.save(filename, n1)
        with open(filename, 'wb') as f:
            pickle.dump(n1, f)
    elif qetype == 'original':
        ests = ['TT', 'EE', 'TE', 'TB', 'EB']
        resps_original = np.zeros((len(l),5), dtype=np.complex_)
        inv_resps_original = np.zeros((len(l),5) ,dtype=np.complex_)
        for i, est in enumerate(ests):
            resps_original[:,i] = get_analytic_response(est,config,lmax,fwhm=0,nlev_t=0,nlev_p=0,u=u,clfile=clfile)
            inv_resps_original[1:,i] = 1/(resps_original)[1:,i]
        resp_original = np.sum(resps_original, axis=1)
        inv_resp_original = np.zeros_like(l,dtype=np.complex_); inv_resp_original[1:] = 1/(resp_original)[1:]
        n1 = {'total':0, 'TT':0, 'EE':0, 'TE':0, 'TB':0, 'EB':0}
        for i, sim in enumerate(sims):
            plm_TT_ij = np.load(dir_out+f'/plm_TT_healqest_seed1_{sim}_seed2_{sim}_lmax{lmax}_nside{nside}_cmbonly_phi1_tqu1tqu2.npy')
            plm_EE_ij = np.load(dir_out+f'/plm_EE_healqest_seed1_{sim}_seed2_{sim}_lmax{lmax}_nside{nside}_cmbonly_phi1_tqu1tqu2.npy')
            plm_TE_ij = np.load(dir_out+f'/plm_TE_healqest_seed1_{sim}_seed2_{sim}_lmax{lmax}_nside{nside}_cmbonly_phi1_tqu1tqu2.npy')
            plm_TB_ij = np.load(dir_out+f'/plm_TB_healqest_seed1_{sim}_seed2_{sim}_lmax{lmax}_nside{nside}_cmbonly_phi1_tqu1tqu2.npy')
            plm_EB_ij = np.load(dir_out+f'/plm_EB_healqest_seed1_{sim}_seed2_{sim}_lmax{lmax}_nside{nside}_cmbonly_phi1_tqu1tqu2.npy')
            plm_total_ij = plm_TT_ij + plm_EE_ij + plm_TE_ij + plm_TB_ij + plm_EB_ij
            # Response correct healqest
            plm_total_ij = hp.almxfl(plm_total_ij,inv_resp_original)
            plm_TT_ij = hp.almxfl(plm_TT_ij,inv_resps_original[:,0])
            plm_EE_ij = hp.almxfl(plm_EE_ij,inv_resps_original[:,1])
            plm_TE_ij = hp.almxfl(plm_TE_ij,inv_resps_original[:,2])
            plm_TB_ij = hp.almxfl(plm_TB_ij,inv_resps_original[:,3])
            plm_EB_ij = hp.almxfl(plm_EB_ij,inv_resps_original[:,4])
            # Get ij auto spectra <ijij>
            auto = hp.alm2cl(plm_total_ij, plm_total_ij, lmax=lmax)
            auto_TT = hp.alm2cl(plm_TT_ij, plm_TT_ij, lmax=lmax)
            auto_TE = hp.alm2cl(plm_TE_ij, plm_TE_ij, lmax=lmax)
            auto_EE = hp.alm2cl(plm_EE_ij, plm_EE_ij, lmax=lmax)
            auto_TB = hp.alm2cl(plm_TB_ij, plm_TB_ij, lmax=lmax)
            auto_EB = hp.alm2cl(plm_EB_ij, plm_EB_ij, lmax=lmax)
            # Now get the ji sims
            plm_TT_ji = np.load(dir_out+f'/plm_TT_healqest_seed1_{sim}_seed2_{sim}_lmax{lmax}_nside{nside}_cmbonly_phi1_tqu2tqu1.npy')
            plm_EE_ji = np.load(dir_out+f'/plm_EE_healqest_seed1_{sim}_seed2_{sim}_lmax{lmax}_nside{nside}_cmbonly_phi1_tqu2tqu1.npy')
            plm_TE_ji = np.load(dir_out+f'/plm_TE_healqest_seed1_{sim}_seed2_{sim}_lmax{lmax}_nside{nside}_cmbonly_phi1_tqu2tqu1.npy')
            plm_TB_ji = np.load(dir_out+f'/plm_TB_healqest_seed1_{sim}_seed2_{sim}_lmax{lmax}_nside{nside}_cmbonly_phi1_tqu2tqu1.npy')
            plm_EB_ji = np.load(dir_out+f'/plm_EB_healqest_seed1_{sim}_seed2_{sim}_lmax{lmax}_nside{nside}_cmbonly_phi1_tqu2tqu1.npy')
            plm_total_ji = plm_TT_ji + plm_EE_ji + plm_TE_ji + plm_TB_ji + plm_EB_ji
            # Response correct healqest
            plm_total_ji = hp.almxfl(plm_total_ji,inv_resp_original)
            plm_TT_ji = hp.almxfl(plm_TT_ji,inv_resps_original[:,0])
            plm_EE_ji = hp.almxfl(plm_EE_ji,inv_resps_original[:,1])
            plm_TE_ji = hp.almxfl(plm_TE_ji,inv_resps_original[:,2])
            plm_TB_ji = hp.almxfl(plm_TB_ji,inv_resps_original[:,3])
            plm_EB_ji = hp.almxfl(plm_EB_ji,inv_resps_original[:,4])
            # Get cross spectra <ijji>
            cross = hp.alm2cl(plm_total_ij, plm_total_ji, lmax=lmax)
            cross_TT = hp.alm2cl(plm_TT_ij, plm_TT_ji, lmax=lmax)
            cross_TE = hp.alm2cl(plm_TE_ij, plm_TE_ji, lmax=lmax)
            cross_EE = hp.alm2cl(plm_EE_ij, plm_EE_ji, lmax=lmax)
            cross_TB = hp.alm2cl(plm_TB_ij, plm_TB_ji, lmax=lmax)
            cross_EB = hp.alm2cl(plm_EB_ij, plm_EB_ji, lmax=lmax)
            n1['total'] += auto + cross
            n1['TT'] += auto_TT + cross_TT
            n1['EE'] += auto_TE + cross_TE
            n1['TE'] += auto_EE + cross_EE
            n1['TB'] += auto_TB + cross_TB
            n1['EB'] += auto_EB + cross_EB
        n1['total'] *= 1/num
        n1['TT'] *= 1/num
        n1['EE'] *= 1/num
        n1['TE'] *= 1/num
        n1['TB'] *= 1/num
        n1['EB'] *= 1/num
        n0 = get_n0(sims=sims,qetype=qetype,fluxlim=fluxlim,lmax=lmax,
                    nside=nside,u=u,config_file=config_file,dir_out=dir_out,
                    withfg=False)
        n1['total'] -= n0['total']
        n1['TT'] -= n0['TT']
        n1['EE'] -= n0['EE']
        n1['TE'] -= n0['TE']
        n1['TB'] -= n0['TB']
        n1['EB'] -= n0['EB']
        #np.save(filename, n1)
        with open(filename, 'wb') as f:
            pickle.dump(n1, f)
    else:
        print('Invalid argument qetype.')
    return n1

########## Below here is good ##########

def analyze(sims=np.arange(29)+100,n0_sims=np.arange(29)+100,n1_sims=np.arange(29)+100,
            u=None,fluxlim=0.200,
            config_file='profhrd_yuka.yaml',
            #config_file='test_yuka.yaml',
            fwhm=0,nlev_t=0,nlev_p=0,
            #fwhm=1,nlev_t=5,nlev_p=5,
            noise_file='nl_cmbmv_20192020.dat',fsky_corr=25.308939726920805,
            #noise_file=None,fsky_corr=1,
            cltype='len',dir_out='/scratch/users/yukanaka/gmv/',
            save_fig=True,unl=False,
            n0=False,n1=False):
    '''
    Compare with N0/N1 subtraction.
    '''
    #sims = np.delete(sims,[10,12,14,20,22])
    sims = np.delete(sims,[3])
    n0_sims = np.delete(n0_sims,[3])
    config = utils.parse_yaml(config_file)
    lmax = config['Lmax']
    lmin = config['lmin']
    lmaxT = config['lmaxT']
    lmaxP = config['lmaxP']
    l = np.arange(0,lmax+1)
    num = len(sims)
    if u is not None:
        append = f'tsrc_fluxlim{fluxlim:.3f}'
    elif unl is True:
        append = 'unl'
    else:
        append = 'cmbonly'
    if nlev_t!=0 or nlev_p!=0:
        append += f'_fwhm{fwhm}_nlevt{nlev_t}_nlevp{nlev_p}'

    # Get SQE analytic response
    ests = ['TT', 'EE', 'TE', 'TB', 'EB']
    #ests = ['EB']
    resps_original = np.zeros((len(l),len(ests)), dtype=np.complex_)
    inv_resps_original = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
    for i, est in enumerate(ests):
        resps_original[:,i] = get_analytic_response(est,config,gmv=False,cltype=cltype,
                                                    fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,
                                                    noise_file=noise_file,fsky_corr=fsky_corr)
        inv_resps_original[1:,i] = 1/(resps_original)[1:,i]
    resp_original = np.sum(resps_original, axis=1)
    inv_resp_original = np.zeros_like(l,dtype=np.complex_); inv_resp_original[1:] = 1/(resp_original)[1:]

    # Get GMV analytic response
    resp_gmv = get_analytic_response('all',config,gmv=True,cltype=cltype,
                                     fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,
                                     noise_file=noise_file,fsky_corr=fsky_corr)
    resp_gmv_TTEETE = get_analytic_response('TTEETE',config,gmv=True,cltype=cltype,
                                            fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,
                                            noise_file=noise_file,fsky_corr=fsky_corr)
    inv_resp_gmv = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv[1:] = 1./(resp_gmv)[1:]
    inv_resp_gmv_TTEETE = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TTEETE[1:] = 1./(resp_gmv_TTEETE)[1:]

    if u is not None:
        # If we are hardening, get the profile response and weight
        # SQE
        resp_original_TT_ss = get_analytic_response('TTprf',config,gmv=False,cltype=cltype,
                                                    fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,u=u,
                                                    noise_file=noise_file,fsky_corr=fsky_corr)
        resp_original_TT_sk = get_analytic_response('TTTTprf',config,gmv=False,cltype=cltype,
                                                    fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,u=u,
                                                    noise_file=noise_file,fsky_corr=fsky_corr)
        weight_original = -1 * resp_original_TT_sk / resp_original_TT_ss
        resp_original_hrd = resp_original + weight_original*resp_original_TT_sk # Equivalent to resp_original_TT (hardened) + np.sum(resps_original[:,1:], axis=1)
        resp_original_TT_hrd = resps_original[:,0] + weight_original*resp_original_TT_sk
        inv_resp_original_hrd = np.zeros_like(l,dtype=np.complex_); inv_resp_original_hrd[1:] = 1/(resp_original_hrd)[1:]
        inv_resp_original_TT_hrd = np.zeros_like(l,dtype=np.complex_); inv_resp_original_TT_hrd = 1/(resp_original_TT_hrd)[1:]

        # GMV
        resp_gmv_TTEETE_ss = get_analytic_response('TTEETEprf',config,gmv=True,cltype=cltype,
                                                   fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,u=u[lmin:],
                                                   noise_file=noise_file,fsky_corr=fsky_corr)
        resp_gmv_TTEETE_sk = get_analytic_response('TTEETETTEETEprf',config,gmv=True,cltype=cltype,
                                                   fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,u=u[lmin:],
                                                   noise_file=noise_file,fsky_corr=fsky_corr)
        weight_gmv = -1 * resp_gmv_TTEETE_sk / resp_gmv_TTEETE_ss
        resp_gmv_hrd = resp_gmv + weight_gmv*resp_gmv_TTEETE_sk # Equivalent to resp_gmv_TTEETE (hardened) + resp_gmv_TBEB (unhardened)
        resp_gmv_TTEETE_hrd = resp_gmv_TTEETE + weight_gmv*resp_gmv_TTEETE_sk
        inv_resp_gmv_hrd = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_hrd[1:] = 1./(resp_gmv_hrd)[1:]
        inv_resp_gmv_TTEETE_hrd = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TTEETE_hrd[1:] = 1./(resp_gmv_TTEETE_hrd)[1:]

    if n0:
        # Get N0 correction (remember these still need (l*(l+1))**2/4 factor to convert to kappa)
        n0_gmv = get_n0(sims=n0_sims,qetype='gmv',config=config,cltype=cltype,dir_out=dir_out,u=None,fluxlim=fluxlim)
        n0_gmv_total = n0_gmv['total'] * (l*(l+1))**2/4
        n0_gmv_TTEETE = n0_gmv['TTEETE'] * (l*(l+1))**2/4
        n0_gmv_TBEB = n0_gmv['TBEB'] * (l*(l+1))**2/4
        n0_original = get_n0(sims=n0_sims,qetype='sqe',config=config,cltype=cltype,dir_out=dir_out,u=None,fluxlim=fluxlim)
        n0_original_total = n0_original['total'] * (l*(l+1))**2/4
        n0_original_TT = n0_original['TT'] * (l*(l+1))**2/4
        n0_original_EE = n0_original['EE'] * (l*(l+1))**2/4
        n0_original_TE = n0_original['TE'] * (l*(l+1))**2/4
        n0_original_TB = n0_original['TB'] * (l*(l+1))**2/4
        n0_original_EB = n0_original['EB'] * (l*(l+1))**2/4
        if u is not None:
            n0_gmv_hrd = get_n0(sims=n0_sims,qetype='gmv',config=config,cltype=cltype,dir_out=dir_out,u=u,fluxlim=fluxlim)
            n0_gmv_total_hrd = n0_gmv_hrd['total'] * (l*(l+1))**2/4
            n0_gmv_TTEETE_hrd = n0_gmv_hrd['TTEETE'] * (l*(l+1))**2/4
            n0_original_hrd = get_n0(sims=n0_sims,qetype='sqe',config=config,cltype=cltype,dir_out=dir_out,u=u,fluxlim=fluxlim)
            n0_original_total_hrd = n0_original_hrd['total'] * (l*(l+1))**2/4
            n0_original_TT_hrd = n0_original_hrd['TT'] * (l*(l+1))**2/4

    if n1:
        # Get N1 correction (remember these still need (l*(l+1))**2/4 factor to convert to kappa)
        n1_gmv = get_n1(sims=n1_sims,qetype='gmv',config=config,cltype=cltype,dir_out=dir_out)
        n1_gmv_total = n1_gmv['total'] * (l*(l+1))**2/4
        n1_gmv_TTEETE = n1_gmv['TTEETE'] * (l*(l+1))**2/4
        n1_gmv_TBEB = n1_gmv['TBEB'] * (l*(l+1))**2/4
        n1_original = get_n1(sims=n1_sims,qetype='sqe',config=config,cltype=cltype,dir_out=dir_out)
        n1_original_total = n1_original['total'] * (l*(l+1))**2/4
        n1_original_TT = n1_original['TT'] * (l*(l+1))**2/4
        n1_original_EE = n1_original['EE'] * (l*(l+1))**2/4
        n1_original_TE = n1_original['TE'] * (l*(l+1))**2/4
        n1_original_TB = n1_original['TB'] * (l*(l+1))**2/4
        n1_original_EB = n1_original['EB'] * (l*(l+1))**2/4

    auto_gmv_all = 0 #np.zeros((len(l),len(sims)))
    auto_original_all = 0 #np.zeros((len(l),len(sims)))
    cross_gmv_all = 0
    cross_original_all = 0
    auto_input_all = 0
    cross_gmv_uncorrected_all = 0
    cross_original_uncorrected_all = 0
    #cross_original_uncorrected_all_EE = 0
    auto_original_all_TT = 0
    auto_original_all_EE = 0
    auto_original_all_TE = 0
    auto_original_all_TB = 0
    auto_original_all_EB = 0
    auto_original_all_hrd = 0
    auto_gmv_all_hrd = 0
    auto_original_all_TT_hrd = 0
    auto_gmv_all_TTEETE_hrd = 0
    auto_gmv_debiased_all = 0
    auto_original_debiased_all = 0

    #resp_gmv = np.load(dir_out+f'/resp/sim_resp_gmv_estall_{num}sims_lmaxT{lmaxT}_lmaxP{lmaxP}_nside8192_cmbonly.npy')
    #resp_original = np.load(dir_out+f'/resp/sim_resp_sqe_estall_{num}sims_lmaxT{lmaxT}_lmaxP{lmaxP}_nside8192_cmbonly.npy')
    #inv_resp_original = np.zeros_like(l,dtype=np.complex_); inv_resp_original[1:] = 1/(resp_original)[1:]
    #inv_resp_gmv = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv[1:] = 1./(resp_gmv)[1:]

    for ii, sim in enumerate(sims):
        # Load GMV plms
        plm_gmv = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside8192_{append}.npy')
        plm_gmv_TTEETE = np.load(dir_out+f'/plm_TTEETE_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside8192_{append}.npy')

        # Load SQE plms
        plms_original = np.zeros((len(np.load(dir_out+f'/plm_TT_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside8192_{append}.npy')),5), dtype=np.complex_)
        for i, est in enumerate(ests):
            plms_original[:,i] = np.load(dir_out+f'/plm_{est}_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside8192_{append}.npy')
        plm_original = np.sum(plms_original, axis=1)
        #plm_original_EE = np.load(dir_out+f'/plm_EE_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside8192_{append}.npy')

        if u is not None:
            # Harden!
            # GMV
            glm_prf_TTEETE = np.load(dir_out+f'/plm_TTEETEprf_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside8192_{append}.npy')
            plm_gmv_hrd = plm_gmv + hp.almxfl(glm_prf_TTEETE, weight_gmv) # Equivalent to plm_gmv_TTEETE_hrd + plm_gmv_TBEB
            plm_gmv_TTEETE_hrd = plm_gmv_TTEETE + hp.almxfl(glm_prf_TTEETE, weight_gmv)

            # SQE
            glm_prf_TT = np.load(dir_out+f'/plm_TTprf_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside8192_{append}.npy')
            plm_original_TT_hrd = plms_original[:,0] + hp.almxfl(glm_prf_TT, weight_original)
            plm_original_hrd = plm_TT_hrd + np.sum(plms_original[:,1:], axis=1)

        # Response correct
        plm_gmv_resp_corr = hp.almxfl(plm_gmv,inv_resp_gmv)
        plm_original_resp_corr = hp.almxfl(plm_original,inv_resp_original)
        plm_original_resp_corr_TT = hp.almxfl(plms_original[:,0],inv_resps_original[:,0])
        plm_original_resp_corr_EE = hp.almxfl(plms_original[:,1],inv_resps_original[:,1])
        plm_original_resp_corr_TE = hp.almxfl(plms_original[:,2],inv_resps_original[:,2])
        plm_original_resp_corr_TB = hp.almxfl(plms_original[:,3],inv_resps_original[:,3])
        plm_original_resp_corr_EB = hp.almxfl(plms_original[:,4],inv_resps_original[:,4])
        if u is not None:
            plm_gmv_resp_corr_hrd = hp.almxfl(plm_gmv_hrd,inv_resp_gmv_hrd)
            plm_gmv_TTEETE_resp_corr_hrd = hp.almxfl(plm_gmv_TTEETE_hrd,inv_resp_gmv_TTEETE_hrd)
            plm_original_resp_corr_hrd = hp.almxfl(plm_original_hrd,inv_resp_original_hrd)
            plm_original_TT_resp_corr_hrd = hp.almxfl(plm_original_TT_hrd,inv_resp_original_TT_hrd)

        # Get spectra
        auto_gmv = hp.alm2cl(plm_gmv_resp_corr, plm_gmv_resp_corr, lmax=lmax) * (l*(l+1))**2/4
        auto_original = hp.alm2cl(plm_original_resp_corr, plm_original_resp_corr, lmax=lmax) * (l*(l+1))**2/4
        auto_original_TT = hp.alm2cl(plm_original_resp_corr_TT, plm_original_resp_corr_TT, lmax=lmax) * (l*(l+1))**2/4
        auto_original_EE = hp.alm2cl(plm_original_resp_corr_EE, plm_original_resp_corr_EE, lmax=lmax) * (l*(l+1))**2/4
        auto_original_TE = hp.alm2cl(plm_original_resp_corr_TE, plm_original_resp_corr_TE, lmax=lmax) * (l*(l+1))**2/4
        auto_original_TB = hp.alm2cl(plm_original_resp_corr_TB, plm_original_resp_corr_TB, lmax=lmax) * (l*(l+1))**2/4
        auto_original_EB = hp.alm2cl(plm_original_resp_corr_EB, plm_original_resp_corr_EB, lmax=lmax) * (l*(l+1))**2/4
        if u is not None:
            auto_gmv_hrd = hp.alm2cl(plm_gmv_resp_corr_hrd, plm_gmv_resp_corr_hrd, lmax=lmax) * (l*(l+1))**2/4
            auto_gmv_TTEETE_hrd = hp.alm2cl(plm_gmv_TTEETE_resp_corr_hrd, plm_gmv_TTEETE_resp_corr_hrd, lmax=lmax) * (l*(l+1))**2/4
            auto_original_hrd = hp.alm2cl(plm_original_resp_corr_hrd, plm_original_resp_corr_hrd, lmax=lmax) * (l*(l+1))**2/4
            auto_original_TT_hrd = hp.alm2cl(plm_original_TT_resp_corr_hrd, plm_original_TT_resp_corr_hrd, lmax=lmax) * (l*(l+1))**2/4

        # N0 and N1 subtract
        if n0 and n1:
            auto_gmv_debiased = auto_gmv - n0_gmv_total - n1_gmv_total
            auto_original_debiased = auto_original - n0_original_total - n1_original_total
        elif n0:
            auto_gmv_debiased = auto_gmv - n0_gmv_total
            auto_original_debiased = auto_original - n0_original_total

        auto_gmv_all += auto_gmv
        auto_original_all += auto_original
        auto_original_all_TT += auto_original_TT
        auto_original_all_EE += auto_original_EE
        auto_original_all_TE += auto_original_TE
        auto_original_all_TB += auto_original_TB
        auto_original_all_EB += auto_original_EB
        if n0:
            auto_gmv_debiased_all += auto_gmv_debiased
            auto_original_debiased_all += auto_original_debiased
        if u is not None:
            auto_original_all_hrd += auto_original_hrd
            auto_gmv_all_hrd += auto_gmv_hrd
            auto_original_all_TT_hrd += auto_original_TT_hrd
            auto_gmv_all_TTEETE_hrd += auto_gmv_TTEETE_hrd
    
        # Cross correlate with input plm
        if not unl:
            if os.path.isfile(f'/scratch/users/yukanaka/input_phi1/phi_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_seed{sim}_alm_lmax{lmax}.fits'):
                input_plm = hp.read_alm(f'/scratch/users/yukanaka/input_phi1/phi_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_seed{sim}_alm_lmax{lmax}.fits')
            else:
                input_phi = hp.read_map(f'/scratch/users/yukanaka/input_phi1/phi_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_seed{sim}.fits')
                input_phi = hp.pixelfunc.ud_grade(input_phi,8192)
                input_plm = hp.map2alm(input_phi,lmax=lmax)
                # Saving as alms because these maps are huge
                hp.fitsfunc.write_alm(f'/scratch/users/yukanaka/input_phi1/phi_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_seed{sim}_alm_lmax{lmax}.fits',input_plm)
            cross_gmv_all += hp.alm2cl(input_plm, plm_gmv_resp_corr, lmax=lmax) * (l*(l+1))**2/4
            cross_original_all += hp.alm2cl(input_plm, plm_original_resp_corr, lmax=lmax) * (l*(l+1))**2/4
            # For response from sims, want to use plms that are not response corrected
            cross_gmv_uncorrected_all += hp.alm2cl(input_plm, plm_gmv, lmax=lmax) * (l*(l+1))**2/4
            cross_original_uncorrected_all += hp.alm2cl(input_plm, plm_original, lmax=lmax) * (l*(l+1))**2/4
            auto_input_all += hp.alm2cl(input_plm, input_plm, lmax=lmax) * (l*(l+1))**2/4
            #cross_original_uncorrected_all_EE += hp.alm2cl(input_plm, plm_original_EE, lmax=lmax) * (l*(l+1))**2/4

    # Save, because I'm paranoid
    #np.save(dir_out+f'/auto_gmv_all_{num}sims_healqest_gmv_lmaxT{lmaxT}_lmaxP{lmaxP}_nside8192_cmbonly.npy',auto_gmv_all)
    #np.save(dir_out+f'/auto_original_all_{num}sims_healqest_sqe_lmaxT{lmaxT}_lmaxP{lmaxP}_nside8192_cmbonly.npy',auto_original_all)

    # Average
    auto_gmv_avg = auto_gmv_all / num
    auto_original_avg = auto_original_all / num
    auto_original_avg_TT = auto_original_all_TT / num
    auto_original_avg_EE = auto_original_all_EE / num
    auto_original_avg_TE = auto_original_all_TE / num
    auto_original_avg_TB = auto_original_all_TB / num
    auto_original_avg_EB = auto_original_all_EB / num
    if n0:
        auto_gmv_debiased_avg = auto_gmv_debiased_all / num
        auto_original_debiased_avg = auto_original_debiased_all / num
    if u is not None:
        auto_original_avg_hrd += auto_original_all_hrd
        auto_gmv_avg_hrd += auto_gmv_all_hrd
        auto_original_avg_TT_hrd += auto_original_all_TT_hrd
        auto_gmv_avg_TTEETE_hrd += auto_gmv_all_TTEETE_hrd

    if not unl:
        cross_gmv_avg = cross_gmv_all / num
        cross_original_avg = cross_original_all / num
        auto_input_avg = auto_input_all / num
        cross_gmv_uncorrected_avg = cross_gmv_uncorrected_all / num
        cross_original_uncorrected_avg = cross_original_uncorrected_all / num
        #cross_original_uncorrected_avg_EE = cross_original_uncorrected_all_EE / num
    
        # Get "response from sims" calculated the same way as the MC response
        sim_resp_gmv = cross_gmv_uncorrected_avg / auto_input_avg
        sim_resp_original = cross_original_uncorrected_avg / auto_input_avg
        np.save(dir_out+f'/resp/sim_resp_gmv_estall_{num}sims_lmaxT{lmaxT}_lmaxP{lmaxP}_nside8192_cmbonly.npy',sim_resp_gmv)
        np.save(dir_out+f'/resp/sim_resp_sqe_estall_{num}sims_lmaxT{lmaxT}_lmaxP{lmaxP}_nside8192_cmbonly.npy',sim_resp_original)
        #sim_resp_original_EE = cross_original_uncorrected_avg_EE / auto_input_avg

    # Theory spectrum
    clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
    ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
    clkk = slpp * (l*(l+1))**2/4

    # Plot
    plt.figure(0)
    plt.clf()
    plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')
    plt.plot(l, auto_original_avg, color='firebrick', linestyle='-', label=f'Auto Spectrum (SQE)')
    plt.plot(l, auto_gmv_avg, color='darkblue', linestyle='-', label="Auto Spectrum (GMV)")
    #plt.plot(l, auto_original_debiased_avg, color='firebrick', linestyle='-', label=f'Auto Spectrum (SQE)')
    #plt.plot(l, auto_gmv_debiased_avg, color='darkblue', linestyle='-', label="Auto Spectrum (GMV)")
    plt.plot(l, inv_resp_original * (l*(l+1))**2/4, color='lightcoral', linestyle='--', label='1/R (SQE)')
    plt.plot(l, inv_resp_gmv * (l*(l+1))**2/4, color='cornflowerblue', linestyle='--', label='1/R (GMV)')

    #plt.plot(l, 2*auto_original_avg_TE, color='forestgreen', linestyle='-', label=f'2 x Auto Spectrum (SQE, TE)')
    #plt.plot(l, auto_original_avg_EE, color='mediumorchid', linestyle='-', label=f'Auto Spectrum (SQE, EE)')
    #plt.plot(l, 2*auto_original_avg_TB, color='gold', linestyle='-', label=f'2 x Auto Spectrum (SQE, TB)')
    #plt.plot(l, 2*auto_original_avg_EB, color='orange', linestyle='-', label=f'2 x Auto Spectrum (SQE, EB)')
    #plt.plot(l, inv_resps_original[:,2] * (l*(l+1))**2/4, color='lightgreen', linestyle='--', label='$1/R^{KK}$ (SQE, TE)')
    #plt.plot(l, inv_resps_original[:,1] * (l*(l+1))**2/4, color='plum', linestyle='--', label='$1/R^{KK}$ (SQE, EE)')
    #plt.plot(l, inv_resps_original[:,3] * (l*(l+1))**2/4, color='palegoldenrod', linestyle='--', label='$1/R^{KK}$ (SQE, TB)')
    #plt.plot(l, inv_resps_original[:,4] * (l*(l+1))**2/4, color='bisque', linestyle='--', label='$1/R^{KK}$ (SQE, EB)')
    plt.ylabel("$C_\ell^{\kappa\kappa}$")
    plt.xlabel('$\ell$')
    plt.title(f'Spectra Averaged over {num} Sims')
    plt.legend(loc='lower left', fontsize='x-small')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,lmax)
    #plt.ylim(5e-9,1e-6)
    plt.ylim(8e-9,1e-6)
    if save_fig:
        #plt.savefig(dir_out+f'/figs/{num}_sims_no_hardening_comparison.png')
        plt.savefig(dir_out+f'/figs/{num}_sims_no_hardening_comparison_{append}.png',bbox_inches='tight')
        #plt.savefig(dir_out+f'/figs/{num}_sims_no_hardening_comparison_{append}_n0subtracted.png',bbox_inches='tight')

    """
    plt.figure(1)
    plt.clf()
    plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')
    plt.plot(l, cross_original_avg, color='firebrick', linestyle='-', label=f'Cross Spectrum with Input (SQE)')
    plt.plot(l, cross_gmv_avg, color='darkblue', linestyle='-', label="Cross Spectrum with Input (GMV)")
    plt.plot(l, inv_resp_original * (l*(l+1))**2/4, color='lightcoral', linestyle='--', label='1/R (SQE)')
    plt.plot(l, inv_resp_gmv * (l*(l+1))**2/4, color='cornflowerblue', linestyle='--', label='1/R (GMV)')
    plt.ylabel("$C_\ell^{\kappa\kappa}$")
    plt.xlabel('$\ell$')
    plt.title(f'Spectra Averaged over {num} Sims')
    plt.legend(loc='upper right', fontsize='small')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,lmax)
    plt.ylim(5e-9,1e-6)
    if save_fig:
        plt.savefig(dir_out+f'/figs/{num}_sims_no_hardening_cross_with_input_comparison.png')

    plt.figure(2)
    plt.clf()
    plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')
    plt.plot(l, 1/sim_resp_original * (l*(l+1))**2/4, color='firebrick', linestyle='-', label=f'1/Sim Response (SQE)')
    #plt.plot(l, 1/sim_resp_original_EE * (l*(l+1))**2/4, color='darkgreen', linestyle='-', label=f'1/Sim Response (SQE, EE ONLY)')
    plt.plot(l, 1/sim_resp_gmv * (l*(l+1))**2/4, color='darkblue', linestyle='-', label="1/Sim Response (GMV)")
    plt.plot(l, inv_resp_original * (l*(l+1))**2/4, color='lightcoral', linestyle='--', label='1/R (SQE)')
    #plt.plot(l, inv_resp_original_EE * (l*(l+1))**2/4, color='lightgreen', linestyle='--', label='1/R (SQE, EE ONLY)')
    plt.plot(l, inv_resp_gmv * (l*(l+1))**2/4, color='cornflowerblue', linestyle='--', label='1/R (GMV)')
    plt.ylabel("$C_\ell^{\kappa\kappa}$")
    plt.xlabel('$\ell$')
    plt.title(f'Spectra Averaged over {num} Sims')
    plt.legend(loc='upper right', fontsize='small')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,lmax)
    plt.ylim(5e-9,1e-6)
    if save_fig:
        plt.savefig(dir_out+f'/figs/{num}_sims_no_hardening_sim_response_comparison.png')

    plt.figure(3)
    plt.clf()
    plt.plot(l, sim_resp_original/resp_original, color='firebrick', linestyle='-', label=f'Sim Response/Analytic Response (SQE)')
    #plt.plot(l, sim_resp_original_EE/resp_original_EE, color='darkgreen', linestyle='-', label=f'Sim Response/Analytic Response (SQE, EE ONLY)')
    plt.plot(l, sim_resp_gmv/resp_gmv, color='darkblue', linestyle='-', label="Sim Response/Analytic Response (GMV)")
    plt.xlabel('$\ell$')
    plt.title(f'Spectra Averaged over {num} Sims')
    plt.legend(loc='upper right', fontsize='small')
    plt.xlim(10,lmax)
    plt.ylim(0.8,1.2)
    if save_fig:
        plt.savefig(dir_out+f'/figs/{num}_sims_no_hardening_sim_response_comparison_ratio.png')
    """

def compare_profile_hardening_resp(u=None,cltype='len',dir_out='/scratch/users/yukanaka/gmv/',
                                   config_file='profhrd_yuka.yaml',
                                   fwhm=1,nlev_t=5,nlev_p=5,
                                   noise_file='nl_cmbmv_20192020.dat',fsky_corr=25.308939726920805,
                                   save_fig=True):
    config = utils.parse_yaml(config_file)
    lmax = config['Lmax']
    lmin = config['lmin']
    l = np.arange(0,lmax+1)
    if u is None:
        #u=np.ones(lmax+1,dtype=np.complex_)
        u = hp.sphtfunc.gauss_beam(1*(np.pi/180.)/60., lmax=lmax)

    # Flat sky healqest response
    resp_healqest_TT_ss = get_analytic_response('TTprf',config,gmv=False,cltype=cltype,
                                                fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,u=u,
                                                noise_file=noise_file,fsky_corr=fsky_corr)
    inv_resp_healqest_TT_ss = np.zeros_like(l,dtype=np.complex_); inv_resp_healqest_TT_ss[1:] = 1/(resp_healqest_TT_ss)[1:]
    resp_healqest_TT_sk = get_analytic_response('TTTTprf',config,gmv=False,cltype=cltype,
                                                fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,u=u,
                                                noise_file=noise_file,fsky_corr=fsky_corr)
    inv_resp_healqest_TT_sk = np.zeros_like(l,dtype=np.complex_); inv_resp_healqest_TT_sk[1:] = 1/(resp_healqest_TT_sk)[1:]
    resp_healqest_TT_kk = get_analytic_response('TT',config,gmv=False,cltype=cltype,
                                                fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,u=u,
                                                noise_file=noise_file,fsky_corr=fsky_corr)
    inv_resp_healqest_TT_kk = np.zeros_like(l,dtype=np.complex_); inv_resp_healqest_TT_kk[1:] = 1/(resp_healqest_TT_kk)[1:]

    # GMV response
    u = u[lmin:]
    resp_gmv_TTEETE_ss = get_analytic_response('TTEETEprf',config,gmv=True,cltype=cltype,
                                                fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,u=u,
                                                noise_file=noise_file,fsky_corr=fsky_corr)
    inv_resp_gmv_TTEETE_ss = np.zeros(len(l), dtype=np.complex_); inv_resp_gmv_TTEETE_ss[1:] = 1./(resp_gmv_TTEETE_ss)[1:]
    resp_gmv_TTEETE_sk = get_analytic_response('TTEETETTEETEprf',config,gmv=True,cltype=cltype,
                                                fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,u=u,
                                                noise_file=noise_file,fsky_corr=fsky_corr)
    inv_resp_gmv_TTEETE_sk = np.zeros(len(l), dtype=np.complex_); inv_resp_gmv_TTEETE_sk[1:] = 1./(resp_gmv_TTEETE_sk)[1:]
    resp_gmv_TTEETE_kk = get_analytic_response('TTEETE',config,gmv=True,cltype=cltype,
                                                fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,u=u,
                                                noise_file=noise_file,fsky_corr=fsky_corr)
    inv_resp_gmv_TTEETE_kk = np.zeros(len(l), dtype=np.complex_); inv_resp_gmv_TTEETE_kk[1:] = 1./(resp_gmv_TTEETE_kk)[1:]

    # Theory spectrum
    clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
    ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
    clkk = slpp * (l*(l+1))**2/4

    plt.figure(0)
    plt.clf()
    plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')
    plt.plot(l, inv_resp_healqest_TT_ss * (l*(l+1))**2/4, color='firebrick', linestyle='-', label='$1/R^{SS}$ (Healqest TT)')
    plt.plot(l, inv_resp_healqest_TT_sk * (l*(l+1))**2/4, color='lightcoral', linestyle='--', label='$1/R^{SK}$ (Healqest TT)')
    plt.plot(l, inv_resp_healqest_TT_kk * (l*(l+1))**2/4, color='maroon', linestyle=':', label='$1/R^{KK}$ (Healqest TT)')
    plt.plot(l, inv_resp_gmv_TTEETE_ss * (l*(l+1))**2/4, color='seagreen', linestyle='-', label='$1/R^{SS}$ (GMV [TT, EE, TE])')
    plt.plot(l, inv_resp_gmv_TTEETE_sk * (l*(l+1))**2/4, color='lightgreen', linestyle='--', label='$1/R^{SK}$ (GMV [TT, EE, TE])')
    plt.plot(l, inv_resp_gmv_TTEETE_kk * (l*(l+1))**2/4, color='darkgreen', linestyle=':', label='$1/R^{KK}$ (GMV [TT, EE, TE])')
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
    plt.plot(l, np.abs(inv_resp_gmv_TTEETE_ss/inv_resp_healqest_TT_ss - 1)*100, color='seagreen', linestyle='-', label='$R^{SS}$ Comparison')
    plt.plot(l, np.abs(inv_resp_gmv_TTEETE_sk/inv_resp_healqest_TT_sk - 1)*100, color='lightgreen', linestyle='--', label='$R^{SK}$ Comparison')
    plt.plot(l, np.abs(inv_resp_gmv_TTEETE_kk/inv_resp_healqest_TT_kk - 1)*100, color='darkgreen', linestyle=':', label='$R^{KK}$ Comparison')
    plt.xlabel('$\ell$')
    plt.title("$|{R_{healqest}}/{R_{GMV,A}} - 1|$ x 100%")
    plt.legend(loc='upper right', fontsize='small')
    plt.xlim(10,lmax)
    plt.ylim(0,30)
    #plt.ylim(0,50)
    if save_fig:
        plt.savefig(dir_out+f'/figs/profile_response_comparison_frac_diff.png')
        #plt.savefig(dir_out+f'/figs/profile_response_comparison_frac_diff_noiseless.png')

def compare_lensing_resp(cltype='len',dir_out='/scratch/users/yukanaka/gmv/',
                         #config_file='profhrd_yuka.yaml',
                         config_file='test_yuka.yaml',
                         fwhm=1,nlev_t=5,nlev_p=5,
                         #fwhm=0,nlev_t=0,nlev_p=0,
                         noise_file=None,fsky_corr=1,
                         #noise_file='nl_cmbmv_20192020.dat',fsky_corr=25.308939726920805,
                         TTEETE_only=False,TBEB_only=False,save_fig=True):

    config = utils.parse_yaml(config_file)
    lmax = config['Lmax']
    l = np.arange(0,lmax+1)
    # Flat sky healqest response
    if TTEETE_only:
        ests = ['TT', 'EE', 'TE']
    elif TBEB_only:
        ests = ['TB', 'EB']
    else:
        ests = ['TT', 'EE', 'TE', 'TB', 'EB']
    resps_original = np.zeros((len(l),len(ests)), dtype=np.complex_)
    inv_resps_original = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
    for i, est in enumerate(ests):
        resps_original[:,i] = get_analytic_response(est,config,gmv=False,cltype=cltype,
                                                    fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,
                                                    noise_file=noise_file,fsky_corr=fsky_corr)
        inv_resps_original[1:,i] = 1/(resps_original)[1:,i]
    resp_original = np.sum(resps_original, axis=1)
    inv_resp_original = np.zeros_like(l,dtype=np.complex_); inv_resp_original[1:] = 1/(resp_original)[1:]

    # GMV response
    if TTEETE_only:
        resp_gmv = get_analytic_response('TTEETE',config,gmv=True,cltype=cltype,
                                          fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,
                                          noise_file=noise_file,fsky_corr=fsky_corr)
    elif TBEB_only:
        resp_gmv = get_analytic_response('TBEB',config,gmv=True,cltype=cltype,
                                          fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,
                                          noise_file=noise_file,fsky_corr=fsky_corr)
    else:
        resp_gmv = get_analytic_response('all',config,gmv=True,cltype=cltype,
                                         fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,
                                         noise_file=noise_file,fsky_corr=fsky_corr)
    inv_resp_gmv = np.zeros(len(l), dtype=np.complex_)
    inv_resp_gmv[1:] = 1./(resp_gmv)[1:]

    # Theory spectrum
    clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
    ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
    clkk = slpp * (l*(l+1))**2/4

    # Thing from Yuuki
    #v = (l*(l+1)/2)**2
    #sumresp = np.load('sum_aresp.npy')

    plt.figure(0)
    plt.clf()
    plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')
    plt.plot(l, inv_resp_original * (l*(l+1))**2/4, color='darkblue', linestyle='--', label='$1/R^{KK}$ (Healqest)')
    plt.plot(l, inv_resps_original[:,0] * (l*(l+1))**2/4, color='firebrick', linestyle='--', label='$1/R^{KK}$ (Healqest, TT)')
    plt.plot(l, inv_resps_original[:,2] * (l*(l+1))**2/4, color='forestgreen', linestyle='--', label='$1/R^{KK}$ (Healqest, TE)')
    plt.plot(l, inv_resps_original[:,1] * (l*(l+1))**2/4, color='mediumorchid', linestyle='--', label='$1/R^{KK}$ (Healqest, EE)')
    plt.plot(l, inv_resps_original[:,3] * (l*(l+1))**2/4, color='gold', linestyle='--', label='$1/R^{KK}$ (Healqest, TE)')
    plt.plot(l, inv_resps_original[:,4] * (l*(l+1))**2/4, color='orange', linestyle='--', label='$1/R^{KK}$ (Healqest, EB)')
    #plt.plot(l, inv_resp_gmv * (l*(l+1))**2/4, color='darkblue', linestyle='--', label='$1/R^{KK}$ (GMV)')
    #plt.plot(l,gf1(v*1/(sumresp[:lmax+1]),10),color='crimson',alpha=1.0,label='MV')
    plt.ylabel("$1/R^{\kappa\kappa}$")
    plt.xlabel('$\ell$')
    #plt.title('$1/R$ Comparison with 2019+2020 ILC Noise Curves')
    plt.title('$1/R$ Comparison with 5 uK-arcmin Noise')
    plt.legend(loc='lower right', fontsize='x-small')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,lmax)
    #plt.ylim(-0.3,0)
    #plt.ylim(-0.2,0.2)
    plt.ylim(8e-9,1e-6)
    if save_fig:
        #plt.savefig(dir_out+f'/figs/lensing_response_comparison_ilc_noise.png',bbox_inches='tight')
        plt.savefig(dir_out+f'/figs/lensing_response_comparison_fwhm{fwhm}_nlevt{nlev_t}_nlevp{nlev_p}.png',bbox_inches='tight')

    """
    plt.figure(1)
    plt.clf()
    plt.plot(l, (inv_resp_gmv/inv_resp_original)-1, color='maroon', linestyle='-')
    plt.ylabel("$(N_0^{GMV}/N_0^{healqest})-1$")
    plt.xlabel('$\ell$')
    plt.title('$1/R$ Comparison with 2019+2020 ILC Noise Curves')
    plt.xlim(10,lmax)
    plt.ylim(-0.3,0)
    #plt.ylim(-0.2,0.2)
    #plt.ylim(-0.6,-0.2)
    if save_fig:
        plt.savefig(dir_out+f'/figs/lensing_response_comparison_ilc_noise_frac_diff.png',bbox_inches='tight')
    """

def get_n0(sims,qetype,config,cltype='len',
           fwhm=0,nlev_t=0,nlev_p=0,
           noise_file='nl_cmbmv_20192020.dat',fsky_corr=25.308939726920805,
           dir_out='/scratch/users/yukanaka/gmv/',u=None,fluxlim=0.200):
    '''
    Get N0 bias. qetype should be 'gmv' or 'sqe'.
    Hardens if u is not None.
    Returns dictionary containing keys 'total', 'TTEETE', and 'TBEB' for GMV, but note that
    if we are hardening, 'total' and 'TTEETE' will be hardened but 'TBEB' will not.
    Similarly for SQE.
    '''
    lmax = config['Lmax']
    lmaxT = config['lmaxT']
    lmaxP = config['lmaxP']
    lmin = config['lmin']
    tdict = {'grad':'gcmb', 'len':'lcmb', 'unl':'ucmb'}
    sl = {ee:config['cls'][tdict[cltype]][ee] for ee in config['cls'][tdict[cltype]].keys()}
    l = np.arange(lmax+1,dtype=np.float_)
    num = len(sims)
    if u is not None:
        append = f'tsrc_fluxlim{fluxlim:.3f}'
    else:
        append = 'cmbonly'
    filename = f'/scratch/users/yukanaka/gmv/n0/n0_{num}simpairs_healqest_{qetype}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside8192_{append}.pkl'

    if os.path.isfile(filename):
        n0 = pickle.load(open(filename,'rb'))

    elif qetype == 'gmv':
        # Get GMV analytic response
        resp_gmv = get_analytic_response('all',config,gmv=True,cltype=cltype,
                                         fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,
                                         noise_file=noise_file,fsky_corr=fsky_corr)
        resp_gmv_TTEETE = get_analytic_response('TTEETE',config,gmv=True,cltype=cltype,
                                                fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,
                                                noise_file=noise_file,fsky_corr=fsky_corr)
        resp_gmv_TBEB = get_analytic_response('TBEB',config,gmv=True,cltype=cltype,
                                              fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,
                                              noise_file=noise_file,fsky_corr=fsky_corr)
        if u is not None:
            # If we are hardening, get the profile response and weight
            u = u[lmin:]
            resp_gmv_TTEETE_ss = get_analytic_response('TTEETEprf',config,gmv=True,cltype=cltype,
                                                       fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,u=u,
                                                       noise_file=noise_file,fsky_corr=fsky_corr)
            resp_gmv_TTEETE_sk = get_analytic_response('TTEETETTEETEprf',config,gmv=True,cltype=cltype,
                                                       fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,u=u,
                                                       noise_file=noise_file,fsky_corr=fsky_corr)
            weight_gmv = -1 * resp_gmv_TTEETE_sk / resp_gmv_TTEETE_ss
            resp_gmv = resp_gmv + weight_gmv*resp_gmv_TTEETE_sk # Equivalent to resp_gmv_TTEETE (hardened) + resp_gmv_TBEB (unhardened)
            resp_gmv_TTEETE = resp_gmv_TTEETE + weight_gmv*resp_gmv_TTEETE_sk
        inv_resp_gmv = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv[1:] = 1./(resp_gmv)[1:]
        inv_resp_gmv_TTEETE = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TTEETE[1:] = 1./(resp_gmv_TTEETE)[1:]
        inv_resp_gmv_TBEB = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TBEB[1:] = 1./(resp_gmv_TBEB)[1:]

        n0 = {'total':0, 'TTEETE':0, 'TBEB':0}
        for i, sim1 in enumerate(sims):
            sim2 = sim1 + 1

            # Get the lensed ij sims
            plm_gmv_ij = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside8192_{append}.npy')
            plm_gmv_TTEETE_ij = np.load(dir_out+f'/plm_TTEETE_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside8192_{append}.npy')
            plm_gmv_TBEB_ij = np.load(dir_out+f'/plm_TBEB_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside8192_{append}.npy')

            # Now get the ji sims
            plm_gmv_ji = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside8192_{append}.npy')
            plm_gmv_TTEETE_ji = np.load(dir_out+f'/plm_TTEETE_healqest_gmv_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside8192_{append}.npy')
            plm_gmv_TBEB_ji = np.load(dir_out+f'/plm_TBEB_healqest_gmv_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside8192_{append}.npy')

            if u is not None:
                # Harden!
                glm_prf_TTEETE_ij = np.load(dir_out+f'/plm_TTEETEprf_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside8192_{append}.npy')
                plm_gmv_ij = plm_gmv_ij + hp.almxfl(glm_prf_TTEETE_ij, weight_gmv) # Equivalent to plm_gmv_TTEETE_ij (hardened) + plm_gmv_TBEB_ij (unhardened)
                plm_gmv_TTEETE_ij = plm_gmv_TTEETE_ij + hp.almxfl(glm_prf_TTEETE_ij, weight_gmv)

                glm_prf_TTEETE_ji = np.load(dir_out+f'/plm_TTEETEprf_healqest_gmv_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside8192_{append}.npy')
                plm_gmv_ji = plm_gmv_ji + hp.almxfl(glm_prf_TTEETE_ji, weight_gmv) # Equivalent to plm_gmv_TTEETE_ji (hardened) + plm_gmv_TBEB_ji (unhardened)
                plm_gmv_TTEETE_ji = plm_gmv_TTEETE_ji + hp.almxfl(glm_prf_TTEETE_ji, weight_gmv)

            # Response correct
            plm_gmv_resp_corr_ij = hp.almxfl(plm_gmv_ij,inv_resp_gmv)
            plm_gmv_resp_corr_TTEETE_ij = hp.almxfl(plm_gmv_TTEETE_ij,inv_resp_gmv_TTEETE)
            plm_gmv_resp_corr_TBEB_ij = hp.almxfl(plm_gmv_TBEB_ij,inv_resp_gmv_TBEB)

            plm_gmv_resp_corr_ji = hp.almxfl(plm_gmv_ji,inv_resp_gmv)
            plm_gmv_resp_corr_TTEETE_ji = hp.almxfl(plm_gmv_TTEETE_ji,inv_resp_gmv_TTEETE)
            plm_gmv_resp_corr_TBEB_ji = hp.almxfl(plm_gmv_TBEB_ji,inv_resp_gmv_TBEB)

            # Get ij auto spectra <ijij>
            auto = hp.alm2cl(plm_gmv_resp_corr_ij, plm_gmv_resp_corr_ij, lmax=lmax)
            auto_A = hp.alm2cl(plm_gmv_resp_corr_TTEETE_ij, plm_gmv_resp_corr_TTEETE_ij, lmax=lmax)
            auto_B = hp.alm2cl(plm_gmv_resp_corr_TBEB_ij, plm_gmv_resp_corr_TBEB_ij, lmax=lmax)

            # Get cross spectra <ijji>
            cross = hp.alm2cl(plm_gmv_resp_corr_ij, plm_gmv_resp_corr_ji, lmax=lmax)
            cross_A = hp.alm2cl(plm_gmv_resp_corr_TTEETE_ij, plm_gmv_resp_corr_TTEETE_ji, lmax=lmax)
            cross_B = hp.alm2cl(plm_gmv_resp_corr_TBEB_ij, plm_gmv_resp_corr_TBEB_ji, lmax=lmax)

            n0['total'] += auto + cross
            n0['TTEETE'] += auto_A + cross_A
            n0['TBEB'] += auto_B + cross_B

        n0['total'] *= 1/num
        n0['TTEETE'] *= 1/num
        n0['TBEB'] *= 1/num

        with open(filename, 'wb') as f:
            pickle.dump(n0, f)

    elif qetype == 'sqe':
        # Get SQE analytic response
        ests = ['TT', 'EE', 'TE', 'TB', 'EB']
        resps_original = np.zeros((len(l),len(ests)), dtype=np.complex_)
        inv_resps_original = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
        for i, est in enumerate(ests):
            resps_original[:,i] = get_analytic_response(est,config,gmv=False,cltype=cltype,
                                                        fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,
                                                        noise_file=noise_file,fsky_corr=fsky_corr)
            inv_resps_original[1:,i] = 1/(resps_original)[1:,i]
        resp_original = np.sum(resps_original, axis=1)
        if u is not None:
            # If we are hardening, get the profile response and weight
            resp_original_TT_ss = get_analytic_response('TTprf',config,gmv=False,cltype=cltype,
                                                        fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,u=u,
                                                        noise_file=noise_file,fsky_corr=fsky_corr)
            resp_original_TT_sk = get_analytic_response('TTTTprf',config,gmv=False,cltype=cltype,
                                                        fwhm=fwhm,nlev_t=nlev_t,nlev_p=nlev_p,u=u,
                                                        noise_file=noise_file,fsky_corr=fsky_corr)
            weight_original = -1 * resp_original_TT_sk / resp_original_TT_ss
            resp_original = resp_original + weight_original*resp_original_TT_sk # Equivalent to resp_original_TT (hardened) + np.sum(resps_original[:,1:], axis=1)
            resps_original[:,0] = resps_original[:,0] + weight_original*resp_original_TT_sk
            inv_resps_original[1:,0] = 1/(resps_original)[1:,0]
        inv_resp_original = np.zeros_like(l,dtype=np.complex_); inv_resp_original[1:] = 1/(resp_original)[1:]

        n0 = {'total':0, 'TT':0, 'EE':0, 'TE':0, 'TB':0, 'EB':0}
        for i, sim1 in enumerate(sims):
            sim2 = sim1 + 1

            # Get the lensed ij sims
            plm_TT_ij = np.load(dir_out+f'/plm_TT_healqest_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside8192_{append}.npy')
            plm_EE_ij = np.load(dir_out+f'/plm_EE_healqest_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside8192_{append}.npy')
            plm_TE_ij = np.load(dir_out+f'/plm_TE_healqest_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside8192_{append}.npy')
            plm_TB_ij = np.load(dir_out+f'/plm_TB_healqest_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside8192_{append}.npy')
            plm_EB_ij = np.load(dir_out+f'/plm_EB_healqest_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside8192_{append}.npy')

            # Now get the ji sims
            plm_TT_ji = np.load(dir_out+f'/plm_TT_healqest_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside8192_{append}.npy')
            plm_EE_ji = np.load(dir_out+f'/plm_EE_healqest_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside8192_{append}.npy')
            plm_TE_ji = np.load(dir_out+f'/plm_TE_healqest_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside8192_{append}.npy')
            plm_TB_ji = np.load(dir_out+f'/plm_TB_healqest_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside8192_{append}.npy')
            plm_EB_ji = np.load(dir_out+f'/plm_EB_healqest_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside8192_{append}.npy')

            if u is not None:
                # Harden!
                glm_prf_TT_ij = np.load(dir_out+f'/plm_TTprf_healqest_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside8192_{append}.npy')
                plm_TT_ij = plm_TT_ij + hp.almxfl(glm_prf_TT_ij, weight_original)

                glm_prf_TT_ji = np.load(dir_out+f'/plm_TTprf_healqest_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside8192_{append}.npy')
                plm_TT_ji = plm_TT_ji + hp.almxfl(glm_prf_TT_ji, weight_original)

            plm_total_ij = plm_TT_ij + plm_EE_ij + plm_TE_ij + plm_TB_ij + plm_EB_ij
            plm_total_ji = plm_TT_ji + plm_EE_ji + plm_TE_ji + plm_TB_ji + plm_EB_ji

            # Response correct healqest
            plm_total_ij = hp.almxfl(plm_total_ij,inv_resp_original)
            plm_TT_ij = hp.almxfl(plm_TT_ij,inv_resps_original[:,0])
            plm_EE_ij = hp.almxfl(plm_EE_ij,inv_resps_original[:,1])
            plm_TE_ij = hp.almxfl(plm_TE_ij,inv_resps_original[:,2])
            plm_TB_ij = hp.almxfl(plm_TB_ij,inv_resps_original[:,3])
            plm_EB_ij = hp.almxfl(plm_EB_ij,inv_resps_original[:,4])

            plm_total_ji = hp.almxfl(plm_total_ji,inv_resp_original)
            plm_TT_ji = hp.almxfl(plm_TT_ji,inv_resps_original[:,0])
            plm_EE_ji = hp.almxfl(plm_EE_ji,inv_resps_original[:,1])
            plm_TE_ji = hp.almxfl(plm_TE_ji,inv_resps_original[:,2])
            plm_TB_ji = hp.almxfl(plm_TB_ji,inv_resps_original[:,3])
            plm_EB_ji = hp.almxfl(plm_EB_ji,inv_resps_original[:,4])

            # Get ij auto spectra <ijij>
            auto = hp.alm2cl(plm_total_ij, plm_total_ij, lmax=lmax)
            auto_TT = hp.alm2cl(plm_TT_ij, plm_TT_ij, lmax=lmax)
            auto_TE = hp.alm2cl(plm_TE_ij, plm_TE_ij, lmax=lmax)
            auto_EE = hp.alm2cl(plm_EE_ij, plm_EE_ij, lmax=lmax)
            auto_TB = hp.alm2cl(plm_TB_ij, plm_TB_ij, lmax=lmax)
            auto_EB = hp.alm2cl(plm_EB_ij, plm_EB_ij, lmax=lmax)

            # Get cross spectra <ijji>
            cross = hp.alm2cl(plm_total_ij, plm_total_ji, lmax=lmax)
            cross_TT = hp.alm2cl(plm_TT_ij, plm_TT_ji, lmax=lmax)
            cross_TE = hp.alm2cl(plm_TE_ij, plm_TE_ji, lmax=lmax)
            cross_EE = hp.alm2cl(plm_EE_ij, plm_EE_ji, lmax=lmax)
            cross_TB = hp.alm2cl(plm_TB_ij, plm_TB_ji, lmax=lmax)
            cross_EB = hp.alm2cl(plm_EB_ij, plm_EB_ji, lmax=lmax)

            n0['total'] += auto + cross
            n0['TT'] += auto_TT + cross_TT
            n0['EE'] += auto_TE + cross_TE
            n0['TE'] += auto_EE + cross_EE
            n0['TB'] += auto_TB + cross_TB
            n0['EB'] += auto_EB + cross_EB

        n0['total'] *= 1/num
        n0['TT'] *= 1/num
        n0['EE'] *= 1/num
        n0['TE'] *= 1/num
        n0['TB'] *= 1/num
        n0['EB'] *= 1/num

        with open(filename, 'wb') as f:
            pickle.dump(n0, f)
    else:
        print('Invalid argument qetype.')

    return n0

def get_analytic_response(est, config, gmv, cltype='len',
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
    lmax = config['Lmax']
    lmaxT = config['lmaxT']
    lmaxP = config['lmaxP']
    lmin = config['lmin']
    tdict = {'grad':'gcmb', 'len':'lcmb', 'unl':'ucmb'}
    sl = {ee:config['cls'][tdict[cltype]][ee] for ee in config['cls'][tdict[cltype]].keys()}
    ell = np.arange(lmax+1,dtype=np.float_)

    #TODO: doesn't recognize if u is different
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
        filename = f'/scratch/users/yukanaka/gmv/resp/an_resp{append}.npy'

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
        # Signal + noise spectra
        #TODO: no foregrounds
        #if gmv:
        # If lmaxT != lmaxP, we add artificial noise in TT for ell > lmaxT
        artificial_noise = np.zeros(lmax+1)
        if lmaxT < lmaxP:
            artificial_noise[lmaxT+2:] = 1.e10
        cltt = sl['tt'][:lmax+1] + nltt[:lmax+1] + artificial_noise
        clee = sl['ee'][:lmax+1] + nlee[:lmax+1]
        clbb = sl['bb'][:lmax+1] + nlbb[:lmax+1]
        clte = sl['te'][:lmax+1]
        #else:
        #    cltt = sl['tt'][:lmaxT+1] + nltt[:lmaxT+1]
        #    clee = sl['ee'][:lmaxP+1] + nlee[:lmaxP+1]
        #    clbb = sl['bb'][:lmaxP+1] + nlbb[:lmaxP+1]
        #    clte = sl['te'][:lmaxP+1]

        if not gmv:
            # Create 1/Nl filters
            #flt = np.zeros(lmaxT+1); flt[lmin:] = 1./cltt[lmin:]
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
                qeXY = weights.weights('TT',lmax,config,cltype,u=u)
                qeZA = weights.weights('TTprf',lmax,config,cltype,u=u)
            else:
                qeXY = weights.weights(est,lmax,config,cltype,u=u)
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
analyze()

