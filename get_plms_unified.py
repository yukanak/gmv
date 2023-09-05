#!/usr/bin/env python3
# This just tests all the estimators
# Run like python3 get_plms_unified.py TT 100 101
import sys
import numpy as np
import healpy as hp
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import utils
import qest
import camb
from pathlib import Path
import os
import matplotlib.pyplot as plt
import wignerd
import resp

####################################
#nside = 8192
nside = 2048
fluxlim = 0.200
#fluxlim = 0.010
cambini = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_params.ini'
dir_out = '/scratch/users/yukanaka/gmv/'
clfile = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat'
config_file = 'profhrd_yuka.yaml'
#noise_file='nl_cmbmv_20192020.dat'
noise_file = None
fsky_corr=25.308939726920805
append = f'tsrc_fluxlim{fluxlim:.3f}'
#append = 'cmbonly_phi1_tqu1tqu2'
#append = 'cmbonly_phi1_tqu2tqu1'
#append = 'cmbonly'
#append = 'noiseless_cmbonly'
#append = 'unl'
#append = 'noiseless_unl'
####################################
qe = str(sys.argv[1])
sim1 = int(sys.argv[2])
sim2 = int(sys.argv[3])

config = utils.parse_yaml(config_file)
lmax = config['lmax']
lmaxT = config['lmaxt']
lmaxP = config['lmaxp']
lmin = config['lmint']
u = np.ones(lmax+1, dtype=np.complex_)

tlm_with_sources_sim1 = f'/scratch/users/yukanaka/spt3g_planck2018alms_lowpass5000_withptsrc/cmb_Tsrc_fluxlim{fluxlim:.3f}_set1_rlz{sim1}.fits'
tlm_with_sources_sim2 = f'/scratch/users/yukanaka/spt3g_planck2018alms_lowpass5000_withptsrc/cmb_Tsrc_fluxlim{fluxlim:.3f}_set1_rlz{sim2}.fits'
alm_cmb_sim1 = f'/scratch/users/yukanaka/spt3g_planck2018alms_lowpass5000/lensedTQU1phi1_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_seed{sim1}_lmax9000_nside8192_interp1.0_method1_pol_1_lensed_alm_lowpass5000.fits'
alm_cmb_sim2 = f'/scratch/users/yukanaka/spt3g_planck2018alms_lowpass5000/lensedTQU1phi1_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_seed{sim2}_lmax9000_nside8192_interp1.0_method1_pol_1_lensed_alm_lowpass5000.fits'
alm_cmb_sim1_tqu2 = f'/scratch/users/yukanaka/spt3g_planck2018alms_lowpass5000/lensedTQU2phi1_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_seed{sim1}_lmax9000_nside8192_interp1.0_method1_pol_1_lensed_alm_lowpass5000.fits'
unl_map_sim1 = f'/scratch/users/yukanaka/full_res_maps/unl_from_lensed_cls/unl_from_lensed_cls_seed{sim1}_lmax{lmax}_nside{nside}_20230905.fits'
unl_map_sim2 = f'/scratch/users/yukanaka/full_res_maps/unl_from_lensed_cls/unl_from_lensed_cls_seed{sim2}_lmax{lmax}_nside{nside}_20230905.fits'
if qe == 'TTEETE' or qe == 'TBEB' or qe == 'all' or qe == 'TTEETEprf':
    gmv = True
elif qe == 'TT' or qe == 'TE' or qe == 'EE' or qe == 'TB' or qe == 'EB' or qe == 'TTprf':
    gmv = False
else:
    print('Invalid qe')
####################################

cltype = config['cltype']
cls = config['cls']
sl = {ee:config['cls'][cltype][ee] for ee in config['cls'][cltype].keys()}
filename_sqe = dir_out+f'/plm_{qe}_healqest_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy'
filename_gmv = dir_out+f'/plm_{qe}_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy'
    
if os.path.isfile(filename_sqe) or os.path.isfile(filename_gmv):
    print('File already exists!')
else:
    print(f'Doing reconstruction for sims {sim1} and {sim2}, qe {qe}')

    # Load inputs: full-sky noiseless alms
    if append == 'unl' or append == 'noiseless_unl':
        t1,q1,u1 = hp.read_map(unl_map_sim1,field=[0,1,2])
        tlm1,elm1,blm1 = hp.map2alm([t1,q1,u1],lmax=lmax)
        t2,q2,u2 = hp.read_map(unl_map_sim2,field=[0,1,2])
        tlm2,elm2,blm2 = hp.map2alm([t2,q2,u2],lmax=lmax)
    elif append == 'cmbonly_phi1_tqu1tqu2':
        # Sims that were lensed with the same phi but different CMB realizations, no foregrounds for N1
        tlm1,elm1,blm1 = hp.read_alm(alm_cmb_sim1,hdu=[1,2,3])
        tlm1 = utils.reduce_lmax(tlm1,lmax=lmax)
        elm1 = utils.reduce_lmax(elm1,lmax=lmax)
        blm1 = utils.reduce_lmax(blm1,lmax=lmax)
        tlm2,elm2,blm2 = hp.read_alm(alm_cmb_sim1_tqu2,hdu=[1,2,3])
        tlm2 = utils.reduce_lmax(tlm2,lmax=lmax)
        elm2 = utils.reduce_lmax(elm2,lmax=lmax)
        blm2 = utils.reduce_lmax(blm2,lmax=lmax)
    elif append == 'cmbonly_phi1_tqu2tqu1':
        # ji sims for N1
        tlm2,elm2,blm2 = hp.read_alm(alm_cmb_sim1,hdu=[1,2,3])
        tlm2 = utils.reduce_lmax(tlm2,lmax=lmax)
        elm2 = utils.reduce_lmax(elm2,lmax=lmax)
        blm2 = utils.reduce_lmax(blm2,lmax=lmax)
        tlm1,elm1,blm1 = hp.read_alm(alm_cmb_sim1_tqu2,hdu=[1,2,3])
        tlm1 = utils.reduce_lmax(tlm1,lmax=lmax)
        elm1 = utils.reduce_lmax(elm1,lmax=lmax)
        blm1 = utils.reduce_lmax(blm1,lmax=lmax)
    elif append == 'cmbonly' or append == 'noiseless_cmbonly':
        # No foregrounds, lensed CMB sims for N0 calculation used to subtract from N1
        tlm1,elm1,blm1 = hp.read_alm(alm_cmb_sim1,hdu=[1,2,3])
        tlm1 = utils.reduce_lmax(tlm1,lmax=lmax)
        elm1 = utils.reduce_lmax(elm1,lmax=lmax)
        blm1 = utils.reduce_lmax(blm1,lmax=lmax)
        tlm2,elm2,blm2 = hp.read_alm(alm_cmb_sim2,hdu=[1,2,3])
        tlm2 = utils.reduce_lmax(tlm2,lmax=lmax)
        elm2 = utils.reduce_lmax(elm2,lmax=lmax)
        blm2 = utils.reduce_lmax(blm2,lmax=lmax)
    else:
        # With foregrounds in T, used for N0 calculation and the actual reconstruction
        tlm1 = hp.read_alm(tlm_with_sources_sim1,hdu=[1])
        _,elm1,blm1 = hp.read_alm(alm_cmb_sim1,hdu=[1,2,3])
        tlm1 = utils.reduce_lmax(tlm1,lmax=lmax)
        elm1 = utils.reduce_lmax(elm1,lmax=lmax)
        blm1 = utils.reduce_lmax(blm1,lmax=lmax)
        tlm2 = hp.read_alm(tlm_with_sources_sim2,hdu=[1])
        _,elm2,blm2 = hp.read_alm(alm_cmb_sim2,hdu=[1,2,3])
        tlm2 = utils.reduce_lmax(tlm2,lmax=lmax)
        elm2 = utils.reduce_lmax(elm2,lmax=lmax)
        blm2 = utils.reduce_lmax(blm2,lmax=lmax)
    
    # Adding noise!
    if noise_file is not None:
        noise_curves = np.loadtxt(noise_file)
        nltt = fsky_corr * noise_curves[:,1]
        nlee = fsky_corr * noise_curves[:,2]
        nlbb = fsky_corr * noise_curves[:,2]
        if append == 'cmbonly_phi1_tqu1tqu2' or append == 'cmbonly_phi1_tqu2tqu1' or append == 'noiseless_cmbonly' or append == f'tsrc_fluxlim{fluxlim:.3f}' or append == 'noiseless_unl':
            # For the N1 calc, it’s fine to not add noise to the maps; but you’d filter the maps as if there were noise so that you suppress the modes exactly as in the signal map
            nlmt1 = 0; nlme1 = 0; nlmb1 = 0; nlmt2 = 0; nlme2 = 0; nlmb2 = 0
        else:
            nlm1_filename = f'/scratch/users/yukanaka/gmv/nlm/2019_2020_ilc_noise_nlm_lmax{lmax}_seed{sim1}.npy'
            nlm2_filename = f'/scratch/users/yukanaka/gmv/nlm/2019_2020_ilc_noise_nlm_lmax{lmax}_seed{sim2}.npy'
            if os.path.isfile(nlm1_filename):
                nlm1 = np.load(nlm1_filename,allow_pickle=True)
                nlmt1 = nlm1[:,0]
                nlme1 = nlm1[:,1]
                nlmb1 = nlm1[:,2]
            else:
                np.random.seed(hash('tora')%2**32+sim1)
                nlmt1,nlme1,nlmb1 = hp.synalm([nltt,nlee,nlbb,nltt*0],new=True,lmax=lmax)
                #nlmt1 = utils.reduce_lmax(nlmt1,lmax=lmaxT)
                nlm1 = np.zeros((len(nlmt1),3),dtype=np.complex_)
                nlm1[:,0] = nlmt1; nlm1[:,1] = nlme1; nlm1[:,2] = nlmb1
                np.save(nlm1_filename, nlm1)
            if os.path.isfile(nlm2_filename):
                nlm2 = np.load(nlm2_filename,allow_pickle=True)
                nlmt2 = nlm2[:,0]
                nlme2 = nlm2[:,1]
                nlmb2 = nlm2[:,2]
            else:
                np.random.seed(hash('tora')%2**32+sim2)
                nlmt2,nlme2,nlmb2 = hp.synalm([nltt,nlee,nlbb,nltt*0],new=True,lmax=lmax)
                #nlmt2 = utils.reduce_lmax(nlmt2,lmax=lmaxT)
                nlm2 = np.zeros((len(nlmt2),3),dtype=np.complex_)
                nlm2[:,0] = nlmt2; nlm2[:,1] = nlme2; nlm2[:,2] = nlmb2
                np.save(nlm2_filename, nlm2)
        tlm1 += nlmt1
        elm1 += nlme1
        blm1 += nlmb1
        tlm2 += nlmt2
        elm2 += nlme2
        blm2 += nlmb2
    else:
        nltt = np.zeros(lmax+1); nlee = np.zeros(lmax+1); nlbb = np.zeros(lmax+1)
    
    # Signal + Noise spectra
    # TODO: If you have foregrounds, without fgtt, the 1/R won't match the N0 bias
    # If lmaxT != lmaxP, we add artificial noise in TT for ell > lmaxT
    artificial_noise = np.zeros(lmax+1)
    artificial_noise[lmaxT+2:] = 1.e10
    cltt = sl['tt'][:lmax+1] + nltt[:lmax+1] + artificial_noise
    clee = sl['ee'][:lmax+1] + nlee[:lmax+1]
    clbb = sl['bb'][:lmax+1] + nlbb[:lmax+1]
    clte = sl['te'][:lmax+1]
    
    if not gmv:
        print('Creating filters...')
        # Create 1/Nl filters
        flt = np.zeros(lmax+1); flt[lmin:] = 1./cltt[lmin:]
        fle = np.zeros(lmax+1); fle[lmin:] = 1./clee[lmin:]
        flb = np.zeros(lmax+1); flb[lmin:] = 1./clbb[lmin:]
        
        if qe[0] == 'T': almbar1 = hp.almxfl(tlm1,flt); flm1 = flt
        if qe[0] == 'E': almbar1 = hp.almxfl(elm1,fle); flm1 = fle
        if qe[0] == 'B': almbar1 = hp.almxfl(blm1,flb); flm1 = flb
        
        if qe[1] == 'T': almbar2 = hp.almxfl(tlm2,flt); flm2 = flt
        if qe[1] == 'E': almbar2 = hp.almxfl(elm2,fle); flm2 = fle
        if qe[1] == 'B': almbar2 = hp.almxfl(blm2,flb); flm2 = flb
    else:
        print('Doing the 1/Dl for GMV...')
        invDl = np.zeros(lmax+1, dtype=np.complex_)
        invDl[lmin:] = 1./(cltt[lmin:]*clee[lmin:] - clte[lmin:]**2)
        flb = np.zeros(lmax+1); flb[lmin:] = 1./clbb[lmin:]
    
        # Order is TT, EE, TE, TB, EB
        alm1all = np.zeros((len(tlm1),5), dtype=np.complex_) 
        alm2all = np.zeros((len(tlm2),5), dtype=np.complex_)
        # TT
        alm1all[:,0] = hp.almxfl(tlm1,invDl)
        alm2all[:,0] = hp.almxfl(tlm2,invDl)
        # EE
        alm1all[:,1] = hp.almxfl(elm1,invDl)
        alm2all[:,1] = hp.almxfl(elm2,invDl)
        # TE
        alm1all[:,2] = hp.almxfl(tlm1,invDl)
        alm2all[:,2] = hp.almxfl(elm2,invDl)
        # TB
        alm1all[:,3] = hp.almxfl(tlm1,invDl)
        alm2all[:,3] = hp.almxfl(blm2,flb)
        # EB
        alm1all[:,4] = hp.almxfl(elm1,invDl)
        alm2all[:,4] = hp.almxfl(blm2,flb)
    
        totalcls = np.vstack((cltt,clee,clbb,clte)).T
    
    # Run healqest
    if not gmv:
        #q_original = qest.qest(config,qe,almbar1,almbar2,cltype=cltype)
        q_original = qest.qest(config,cls)
        glm,clm = q_original.eval(qe,almbar1,almbar2,u=u)
        # Save plm and clm
        Path(dir_out).mkdir(parents=True, exist_ok=True)
        np.save(filename_sqe,glm)
    else:
        #q_gmv = qest.qest_gmv(config,qe,alm1all,alm2all,totalcls,cltype=cltype)
        q_gmv = qest.qest_gmv(config,cls)
        glm,clm = q_gmv.eval(qe,alm1all,alm2all,totalcls,u=u)
        # Save plm and clm
        Path(dir_out).mkdir(parents=True, exist_ok=True)
        np.save(filename_gmv,glm)
