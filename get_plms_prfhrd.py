#!/usr/bin/env python3
# This just tests all the estimators
# Run like python3 get_plms_prf.py TT 100
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
unl = False
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
####################################
qe = str(sys.argv[1])
sim = int(sys.argv[2])
#append = str(sys.argv[3])
tlm_with_sources = f'/scratch/users/yukanaka/spt3g_planck2018alms_lowpass5000_withptsrc/cmb_Tsrc_fluxlim{fluxlim:.3f}_set1_rlz{sim}.fits'
alm_cmb = f'/scratch/users/yukanaka/spt3g_planck2018alms_lowpass5000/lensedTQU2phi1_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_seed{sim}_lmax9000_nside8192_interp1.0_method1_pol_1_lensed_alm_lowpass5000.fits'
if qe == 'TTEETE' or qe == 'TBEB' or qe == 'all':
    gmv = True
elif qe == 'TT' or qe == 'TE' or qe == 'EE' or qe == 'TB' or qe == 'EB' or qe == 'TTprf':
    gmv = False
else:
    print('Invalid qe')
####################################

config = utils.parse_yaml(config_file)

print(f'Doing sim {sim}, qe {qe}')

# Run CAMB to get theory Cls
ell,sltt,slee,slbb,slte = utils.get_lensedcls(clfile,lmax=lmax)

# Load inputs: full-sky noiseless alms
tlm = hp.read_alm(tlm_with_sources,hdu=[1])
_,elm,blm = hp.read_alm(alm_cmb,hdu=[1,2,3])
tlm = utils.reduce_lmax(tlm,lmax=lmax)
elm = utils.reduce_lmax(elm,lmax=lmax)
blm = utils.reduce_lmax(blm,lmax=lmax)

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
    
    if qe[0] == 'T': almbar1 = hp.almxfl(tlm,flt); flm1 = flt
    if qe[0] == 'E': almbar1 = hp.almxfl(elm,fle); flm1 = fle
    if qe[0] == 'B': almbar1 = hp.almxfl(blm,flb); flm1 = flb
    
    if qe[1] == 'T': almbar2 = hp.almxfl(tlm,flt); flm2 = flt
    if qe[1] == 'E': almbar2 = hp.almxfl(elm,fle); flm2 = fle
    if qe[1] == 'B': almbar2 = hp.almxfl(blm,flb); flm2 = flb
else:
    print('Doing the 1/Dl for GMV...')
    invDl = np.zeros(lmax+1, dtype=np.complex_)
    invDl[100:] = 1./(cltt[100:]*clee[100:] - clte[100:]**2)
    flb = np.zeros(lmax+1); flb[100:] = 1./clbb[100:]

    # Order is TT, EE, TE, TB, EB
    alm1all = np.zeros((len(tlm),5), dtype=np.complex_) 
    alm2all = np.zeros((len(tlm),5), dtype=np.complex_)
    # TT
    alm1all[:,0] = hp.almxfl(tlm,invDl)
    alm2all[:,0] = hp.almxfl(tlm,invDl)
    # EE
    alm1all[:,1] = hp.almxfl(elm,invDl)
    alm2all[:,1] = hp.almxfl(elm,invDl)
    # TE
    alm1all[:,2] = hp.almxfl(tlm,invDl)
    alm2all[:,2] = hp.almxfl(elm,invDl)
    # TB
    alm1all[:,3] = hp.almxfl(tlm,invDl)
    alm2all[:,3] = hp.almxfl(blm,flb)
    # EB
    alm1all[:,4] = hp.almxfl(elm,invDl)
    alm2all[:,4] = hp.almxfl(blm,flb)

    totalcls = np.vstack((cltt,clee,clbb,clte)).T

# Run healqest
if not gmv:
    q_original = qest_combined_qestobj.qest(config,qe,almbar1,almbar2,cltype='len',u=u)
    glm,clm = q_original.eval()
    # Save plm and clm
    append = f'tsrc_fluxlim{fluxlim:.3f}'
    Path(dir_out).mkdir(parents=True, exist_ok=True)
    np.save(dir_out+f'/plm_{qe}_healqest_seed{sim}_lmax{lmax}_nside{nside}_{append}.npy',glm)
    #np.save(dir_out+f'/clm_{qe}_healqest_seed{sim}_lmax{lmax}_nside{nside}_{append}.npy',clm)
else:
    q_gmv = qest_combined_qestobj.qest_gmv(config,qe,alm1all,alm2all,totalcls,cltype='len',u=u)
    glm,clm = q_gmv.eval()
    # Save plm and clm
    append = f'tsrc_fluxlim{fluxlim:.3f}'
    Path(dir_out).mkdir(parents=True, exist_ok=True)
    np.save(dir_out+f'/plm_{qe}_healqest_gmv_seed{sim}_lmax{lmax}_nside{nside}_{append}.npy',glm)
    #np.save(dir_out+f'/clm_{qe}_healqest_gmv_seed{sim}_lmax{lmax}_nside{nside}_{append}.npy',clm)
    if qe == 'TTEETE':
        glm_prf,_ = q_gmv.get_source_estimator()
        np.save(dir_out+f'/plm_TTEETEprf_healqest_gmv_seed{sim}_lmax{lmax}_nside{nside}_{append}.npy',glm_prf)
