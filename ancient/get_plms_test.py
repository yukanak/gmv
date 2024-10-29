#!/usr/bin/env python3
# This just tests all the estimators
# Run like python3 get_plms_test.py TT 100 101
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
nside = 8192
dir_out = '/scratch/users/yukanaka/gmv/'
config_file = 'test_yuka.yaml'
cltype = 'len'
append = 'unl'
fwhm=1; nlev_t=5; nlev_p=5

config = utils.parse_yaml(config_file)
lmax = config['Lmax']
lmaxT = config['lmaxT']
lmaxP = config['lmaxP']
lmin = config['lmin']
####################################
qe = str(sys.argv[1])
sim1 = int(sys.argv[2])
sim2 = int(sys.argv[3])
alm_cmb_sim1 = f'/scratch/users/yukanaka/spt3g_planck2018alms_lowpass5000/lensedTQU1phi1_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_seed{sim1}_lmax9000_nside8192_interp1.0_method1_pol_1_lensed_alm_lowpass5000.fits'
alm_cmb_sim2 = f'/scratch/users/yukanaka/spt3g_planck2018alms_lowpass5000/lensedTQU1phi1_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_seed{sim2}_lmax9000_nside8192_interp1.0_method1_pol_1_lensed_alm_lowpass5000.fits'
unl_map_sim1 = f'/scratch/users/yukanaka/full_res_maps/unl_from_lensed_cls/unl_from_lensed_cls_seed{sim1}_lmax{lmax}_nside{nside}_20230623.fits'
unl_map_sim2 = f'/scratch/users/yukanaka/full_res_maps/unl_from_lensed_cls/unl_from_lensed_cls_seed{sim2}_lmax{lmax}_nside{nside}_20230623.fits'
if qe == 'TTEETE' or qe == 'TBEB' or qe == 'all' or qe == 'TTEETEprf':
    gmv = True
elif qe == 'TT' or qe == 'TE' or qe == 'EE' or qe == 'TB' or qe == 'EB' or qe == 'TTprf':
    gmv = False
else:
    print('Invalid qe')
####################################
tdict = {'grad':'gcmb', 'len':'lcmb', 'unl':'ucmb'}
sl = {ee:config['cls'][tdict[cltype]][ee] for ee in config['cls'][tdict[cltype]].keys()}
filename_sqe = dir_out+f'/plm_{qe}_healqest_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_fwhm{fwhm}_nlevt{nlev_t}_nlevp{nlev_p}.npy'
filename_gmv = dir_out+f'/plm_{qe}_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_fwhm{fwhm}_nlevt{nlev_t}_nlevp{nlev_p}.npy'
    
if os.path.isfile(filename_sqe) or os.path.isfile(filename_gmv):
    print('File already exists!')
else:
    print(f'Doing reconstruction for sims {sim1} and {sim2}, qe {qe}')

    # Load inputs: full-sky noiseless alms
    if append == 'unl':
        t1,q1,u1 = hp.read_map(unl_map_sim1,field=[0,1,2])
        t1 = hp.pixelfunc.ud_grade(t1,nside)
        q1 = hp.pixelfunc.ud_grade(q1,nside)
        u1 = hp.pixelfunc.ud_grade(u1,nside)
        tlm1,elm1,blm1 = hp.map2alm([t1,q1,u1],lmax=lmax)
        #tlm1 = utils.reduce_lmax(tlm1,lmax=lmaxT)
        t2,q2,u2 = hp.read_map(unl_map_sim2,field=[0,1,2])
        t2 = hp.pixelfunc.ud_grade(t2,nside)
        q2 = hp.pixelfunc.ud_grade(q2,nside)
        u2 = hp.pixelfunc.ud_grade(u2,nside)
        tlm2,elm2,blm2 = hp.map2alm([t2,q2,u2],lmax=lmax)
        #tlm2 = utils.reduce_lmax(tlm2,lmax=lmaxT)
    elif append == 'cmbonly':
        # No foregrounds, lensed CMB sims for N0 calculation used to subtract from N1
        tlm1,elm1,blm1 = hp.read_alm(alm_cmb_sim1,hdu=[1,2,3])
        tlm1 = utils.reduce_lmax(tlm1,lmax=lmax)
        elm1 = utils.reduce_lmax(elm1,lmax=lmax)
        blm1 = utils.reduce_lmax(blm1,lmax=lmax)
        tlm2,elm2,blm2 = hp.read_alm(alm_cmb_sim2,hdu=[1,2,3])
        tlm2 = utils.reduce_lmax(tlm2,lmax=lmax)
        elm2 = utils.reduce_lmax(elm2,lmax=lmax)
        blm2 = utils.reduce_lmax(blm2,lmax=lmax)
    
    # Adding noise!
    bl = hp.gauss_beam(fwhm=fwhm*0.00029088,lmax=lmax)
    nltt = (np.pi/180./60.*nlev_t)**2 / bl**2
    nlee=nlbb = (np.pi/180./60.*nlev_p)**2 / bl**2
    np.random.seed(hash('tora')%2**32+sim1)
    nlmt,nlme,nlmb = hp.synalm([nltt,nlee,nlbb,nltt*0],new=True,lmax=lmax)
    tlm1 += nlmt
    elm1 += nlme
    blm1 += nlmb
    tlm2 += nlmt
    elm2 += nlme
    blm2 += nlmb
    
    # Signal + Noise spectra
    cltt = sl['tt'][:lmax+1] + nltt[:lmax+1]
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
        q_original = qest.qest(config,qe,almbar1,almbar2,cltype=cltype)
        glm,clm = q_original.eval()
        # Save plm and clm
        Path(dir_out).mkdir(parents=True, exist_ok=True)
        np.save(filename_sqe,glm)
    else:
        q_gmv = qest.qest_gmv(config,qe,alm1all,alm2all,totalcls,cltype=cltype)
        glm,clm = q_gmv.eval()
        # Save plm and clm
        Path(dir_out).mkdir(parents=True, exist_ok=True)
        np.save(filename_gmv,glm)
