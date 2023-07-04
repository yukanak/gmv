#!/usr/bin/env python3
# This just tests all the estimators
# Run like python3 get_plms_prf.py TT 100 101
import sys
import numpy as np
import healpy as hp
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import utils
import qest_combined_qestobj
import camb
from pathlib import Path
import os
import matplotlib.pyplot as plt
import wignerd
import resp

####################################
lmax = 4096
nside = 8192
fluxlim = 0.200
cambini = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_params.ini'
dir_out = '/scratch/users/yukanaka/gmv/'
clfile = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat'
config_file = 'profhrd_yuka.yaml'
cltype = 'len'
u = np.ones(lmax+1, dtype=np.complex_)
append = f'tsrc_fluxlim{fluxlim:.3f}'
#append = f'tsrc_fluxlim{fluxlim:.3f}_TEl1l2flip'
#append = 'unl'
####################################
qe = str(sys.argv[1])
sim1 = int(sys.argv[2])
sim2 = int(sys.argv[3])
tlm_with_sources_sim1 = f'/scratch/users/yukanaka/spt3g_planck2018alms_lowpass5000_withptsrc/cmb_Tsrc_fluxlim{fluxlim:.3f}_set1_rlz{sim1}.fits'
tlm_with_sources_sim2 = f'/scratch/users/yukanaka/spt3g_planck2018alms_lowpass5000_withptsrc/cmb_Tsrc_fluxlim{fluxlim:.3f}_set1_rlz{sim2}.fits'
alm_cmb_sim1 = f'/scratch/users/yukanaka/spt3g_planck2018alms_lowpass5000/lensedTQU2phi1_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_seed{sim1}_lmax9000_nside8192_interp1.0_method1_pol_1_lensed_alm_lowpass5000.fits'
alm_cmb_sim2 = f'/scratch/users/yukanaka/spt3g_planck2018alms_lowpass5000/lensedTQU2phi1_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_seed{sim2}_lmax9000_nside8192_interp1.0_method1_pol_1_lensed_alm_lowpass5000.fits'
unl_map_sim1 = f'/scratch/users/yukanaka/full_res_maps/unl_from_lensed_cls/unl_from_lensed_cls_seed{sim1}_lmax{lmax}_nside{nside}_20230623.fits'
unl_map_sim2 = f'/scratch/users/yukanaka/full_res_maps/unl_from_lensed_cls/unl_from_lensed_cls_seed{sim2}_lmax{lmax}_nside{nside}_20230623.fits'
if qe == 'TTEETE' or qe == 'TBEB' or qe == 'all':
    gmv = True
elif qe == 'TT' or qe == 'TE' or qe == 'EE' or qe == 'TB' or qe == 'EB' or qe == 'TTprf':
    gmv = False
else:
    print('Invalid qe')
####################################

config = utils.parse_yaml(config_file)

print(f'Doing reconstruction for sims {sim1} and {sim2}, qe {qe}')

# Run CAMB to get theory Cls
ell,sltt,slee,slbb,slte = utils.get_lensedcls(clfile,lmax=lmax)

# Load inputs: full-sky noiseless alms
if append == 'unl':
    t1,q1,u1 = hp.read_map(unl_map_sim1,field=[0,1,2])
    t1 = hp.pixelfunc.ud_grade(t1,nside)
    q1 = hp.pixelfunc.ud_grade(q1,nside)
    u1 = hp.pixelfunc.ud_grade(u1,nside)
    tlm1,elm1,blm1 = hp.map2alm([t1,q1,u1],lmax=lmax)
    t2,q2,u2 = hp.read_map(unl_map_sim2,field=[0,1,2])
    t2 = hp.pixelfunc.ud_grade(t2,nside)
    q2 = hp.pixelfunc.ud_grade(q2,nside)
    u2 = hp.pixelfunc.ud_grade(u2,nside)
    tlm2,elm2,blm2 = hp.map2alm([t2,q2,u2],lmax=lmax)
else:
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

# Signal + Noise spectra (but not adding noise here)
# TODO: without fgtt, the 1/R won't match the N0 bias
cltt = sltt
clee = slee
clbb = slbb
clte = slte

if not gmv:
    print('Creating filters...')
    # Create 1/Nl filters
    flt = np.zeros(lmax+1); flt[100:] = 1./cltt[100:]
    fle = np.zeros(lmax+1); fle[100:] = 1./clee[100:]
    flb = np.zeros(lmax+1); flb[100:] = 1./clbb[100:]
    
    if qe[0] == 'T': almbar1 = hp.almxfl(tlm1,flt); flm1 = flt
    if qe[0] == 'E': almbar1 = hp.almxfl(elm1,fle); flm1 = fle
    if qe[0] == 'B': almbar1 = hp.almxfl(blm1,flb); flm1 = flb
    
    if qe[1] == 'T': almbar2 = hp.almxfl(tlm2,flt); flm2 = flt
    if qe[1] == 'E': almbar2 = hp.almxfl(elm2,fle); flm2 = fle
    if qe[1] == 'B': almbar2 = hp.almxfl(blm2,flb); flm2 = flb
else:
    print('Doing the 1/Dl for GMV...')
    invDl = np.zeros(lmax+1, dtype=np.complex_)
    invDl[100:] = 1./(cltt[100:]*clee[100:] - clte[100:]**2)
    flb = np.zeros(lmax+1); flb[100:] = 1./clbb[100:]

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
    q_original = qest_combined_qestobj.qest(config,qe,almbar1,almbar2,cltype='len',u=u)
    glm,clm = q_original.eval()
    # Save plm and clm
    Path(dir_out).mkdir(parents=True, exist_ok=True)
    np.save(dir_out+f'/plm_{qe}_healqest_seed1{sim1}_seed2{sim2}_lmax{lmax}_nside{nside}_{append}.npy',glm)
    #np.save(dir_out+f'/clm_{qe}_healqest_seed1{sim1}_seed2{sim2}_lmax{lmax}_nside{nside}_{append}.npy',clm)
else:
    q_gmv = qest_combined_qestobj.qest_gmv(config,qe,alm1all,alm2all,totalcls,cltype='len',u=u)
    glm,clm = q_gmv.eval()
    # Save plm and clm
    Path(dir_out).mkdir(parents=True, exist_ok=True)
    np.save(dir_out+f'/plm_{qe}_healqest_gmv_seed1{sim1}_seed2{sim2}_lmax{lmax}_nside{nside}_{append}.npy',glm)
    #np.save(dir_out+f'/clm_{qe}_healqest_gmv_seed1{sim1}_seed2{sim2}_lmax{lmax}_nside{nside}_{append}.npy',clm)
    if qe == 'TBEB':
        glm_prf,_ = q_gmv.get_source_estimator()
        np.save(dir_out+f'/plm_TTEETEprf_healqest_gmv_seed1{sim1}_seed2{sim2}_lmax{lmax}_nside{nside}_{append}.npy',glm_prf)
