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

def plot_maps():
    config_file='test_yuka_lmaxT3500.yaml'
    config = utils.parse_yaml(config_file)
    lmax = config['lensrec']['Lmax']
    lmin = config['lensrec']['lminT']
    lmaxT = config['lensrec']['lmaxT']
    lmaxP = config['lensrec']['lmaxP']
    nside = config['lensrec']['nside']
    dir_out = config['dir_out']
    l = np.arange(0,lmax+1)
    sim = 1
    withT3 = False
    fg_model = 'agora'
    sims = np.arange(250)+1

    # Get response
    resp_standard = get_sim_response('TT',config,cinv=True,append='standard',sims=sims,gmv=True,withT3=withT3,fg_model=fg_model)
    inv_resp_standard = np.zeros_like(l,dtype=np.complex_); inv_resp_standard[3:] = 1/(resp_standard)[3:]
    resp_mh = get_sim_response('T1T2',config,cinv=True,append='mh',sims=sims,gmv=True,withT3=withT3,fg_model=fg_model)
    #resp_mh = 0.5*get_sim_response('T1T2',config,cinv=True,append='mh',sims=sims,gmv=True,withT3=withT3,fg_model=fg_model)
    #resp_mh += 0.5*get_sim_response('T2T1',config,cinv=True,append='mh',sims=sims,gmv=True,withT3=withT3,fg_model=fg_model)
    inv_resp_mh = np.zeros_like(l,dtype=np.complex_); inv_resp_mh[3:] = 1/(resp_mh)[3:]
    resp_xilc = get_sim_response('T1T2',config,cinv=True,append='crossilc_twoseds',sims=sims,gmv=True,withT3=withT3,fg_model=fg_model)
    #resp_xilc = 0.5*get_sim_response('T1T2',config,cinv=True,append='crossilc_twoseds',sims=sims,gmv=True,withT3=withT3,fg_model=fg_model)
    #resp_xilc += 0.5*get_sim_response('T2T1',config,cinv=True,append='crossilc_twoseds',sims=sims,gmv=True,withT3=withT3,fg_model=fg_model)
    inv_resp_xilc = np.zeros_like(l,dtype=np.complex_); inv_resp_xilc[3:] = 1/(resp_xilc)[3:]

    # Get plms
    #plm_standard = np.load(dir_out+f'/plm_TT_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_standard_cinv.npy')
    #plm_mh = 0.5*np.load(dir_out+f'/plm_T1T2_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_mh_cinv_noT3.npy')
    #plm_mh += 0.5*np.load(dir_out+f'/plm_T2T1_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_mh_cinv_noT3.npy')
    #plm_xilc = 0.5*np.load(dir_out+f'/plm_T1T2_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_crossilc_twoseds_cinv_noT3.npy')
    #plm_xilc += 0.5*np.load(dir_out+f'/plm_T2T1_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_crossilc_twoseds_cinv_noT3.npy')
    plm_standard = np.load(dir_out+f'/plm_TT_healqest_sqe_lmaxT3500_lmaxP4096_nside2048_agora_standard_NONOISE.npy')
    plm_mh = 0.5*np.load(dir_out+f'/plm_T1T2_healqest_sqe_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_mh_noT3_NONOISE.npy')
    plm_mh += 0.5*np.load(dir_out+f'/plm_T2T1_healqest_sqe_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_mh_noT3_NONOISE.npy')
    plm_xilc = 0.5*np.load(dir_out+f'/plm_T1T2_healqest_sqe_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_crossilc_twoseds_noT3_NONOISE.npy')
    plm_xilc += 0.5*np.load(dir_out+f'/plm_T2T1_healqest_sqe_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_crossilc_twoseds_noT3_NONOISE.npy')

    # Get maps
    plm_TT = plm_standard.copy()
    plm_TT = hp.almxfl(plm_TT,inv_resp_standard)
    plm_TT = alm_cutlmax(plm_TT,300)
    phi = hp.alm2map(hp.almxfl(plm_TT,(l*(l+1))/2),nside=nside)
    plm_TT_mh = plm_mh.copy()
    plm_TT_mh = hp.almxfl(plm_TT_mh,inv_resp_mh)
    plm_TT_mh = alm_cutlmax(plm_TT_mh,300)
    phi_mh = hp.alm2map(hp.almxfl(plm_TT_mh,(l*(l+1))/2),nside=nside)
    plm_TT_xilc = plm_xilc.copy()
    plm_TT_xilc = hp.almxfl(plm_TT_xilc,inv_resp_xilc)
    plm_TT_xilc = alm_cutlmax(plm_TT_xilc,300)
    phi_xilc = hp.alm2map(hp.almxfl(plm_TT_xilc,(l*(l+1))/2),nside=nside)

    # Input
    #input_plm = hp.read_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/phi/phi_lmax_{lmax}/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_phi1_seed{sim}_lmax{lmax}.alm')
    input_plm = hp.read_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_spt3g_input_plm_lmax{lmax}.fits')
    input_plm = alm_cutlmax(input_plm,300)
    input_phi = hp.alm2map(hp.almxfl(input_plm,(l*(l+1))/2),nside=nside)

    # Plot
    scale = 0.1*1.5
    plt.figure(0)
    plt.clf()
    hp.gnomview(phi,title='Reconstructed Standard Kappa Map',rot=[0,-60],notext=True,xsize=300,ysize=300,reso=3,min=-1*scale,max=scale,cmap='RdBu_r')#,unit="uK")
    plt.savefig(dir_out+f'/figs/TT_standard_reconstructed_kappa_map.png',bbox_inches='tight')

    plt.clf()
    hp.gnomview(phi_mh,title='Reconstructed Gradient Cleaned Kappa Map',rot=[0,-60],notext=True,xsize=300,ysize=300,reso=3,min=-1*scale,max=scale,cmap='RdBu_r')
    plt.savefig(dir_out+f'/figs/TT_mh_reconstructed_kappa_map.png',bbox_inches='tight')

    plt.clf()
    hp.gnomview(phi_xilc,title='Reconstructed Cross-ILC Kappa Map',rot=[0,-60],notext=True,xsize=300,ysize=300,reso=3,min=-1*scale,max=scale,cmap='RdBu_r')
    plt.savefig(dir_out+f'/figs/TT_xilc_reconstructed_kappa_map.png',bbox_inches='tight')

    plt.clf()
    hp.gnomview(input_phi,title='Input Kappa Map',rot=[0,-60],notext=True,xsize=300,ysize=300,reso=3,min=-1*scale,max=scale,cmap='RdBu_r')
    plt.savefig(dir_out+f'/figs/input_kappa_map.png',bbox_inches='tight')

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

def get_sim_response(est,config,cinv,append,sims,filename=None,gmv=True,withT3=False,fg_model='agora'):
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
        fn += f'_lmaxT{lmaxT}_lmaxP{lmaxP}_lmin{lmin}_cltype{cltype}_{fg_model}_{append}'
        if withT3 and append!='standard':
            fn += '_withT3'
        elif append!='standard':
            fn += '_noT3'
        if gmv and not cinv and append!='standard':
            fn += '_12ests'
        filename = dir_out+f'/resp/sim_resp_{num}sims{fn}.npy'
        print(filename)

    if os.path.isfile(filename):
        print('Loading from existing file!')
        sim_resp = np.load(filename)

    return sim_resp

