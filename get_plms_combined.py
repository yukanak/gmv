#!/usr/bin/env python3
# This just tests all the estimators
# Run like python3 get_plms_gmv.py all 1 qest_gmv_all
import sys
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import numpy as np
import healpy as hp
import utils
import weights
import camb
from pathlib import Path
import os
import matplotlib.pyplot as plt
import wignerd
import resp
import qest

####################################
gmv = True
map_inputs = True
lmax = 4096
nside = 8192
fwhm = 1
nlev_t = 5
nlev_p = 5
cambini = 'healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_params.ini'
dir_out = '/scratch/users/yukanaka/gmv/'
clfile    = 'camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat'
####################################
est = str(sys.argv[1]) # TT/EE/TE/TB/EB/all/A/B
sim = int(sys.argv[2])
append = str(sys.argv[3])
# Get input map or alm
if map_inputs:
    file_map = f'/scratch/users/yukanaka/full_res_maps/len/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb1_seed{sim}_lmax17000_nside8192_interp1.6_method1_pol_1_lensedmap.fits'
else:
    file_alm = f'/scratch/users/yukanaka/alms_llcdm/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb1_seed{sim}_lmax17000_nside8192_interp1.6_method1_pol_1_lensedmap_lmax2048.alm'
####################################
print(f'Doing sim {sim}, est {est}')

# Run CAMB to get theory Cls
ell,sltt,slee,slbb,slte = utils.get_lensedcls(clfile,lmax=lmax)

# Create noise spectra
bl = hp.gauss_beam(fwhm=fwhm*0.00029088,lmax=lmax)
nltt = (np.pi/180./60.*nlev_t)**2 / bl**2
nlee = (np.pi/180./60.*nlev_p)**2 / bl**2
nlbb = (np.pi/180./60.*nlev_p)**2 / bl**2
nlte = 0

# Load inputs: full-sky noiseless alms
if map_inputs:
    t,q,u = hp.read_map(file_map,field=[0,1,2])
    t = hp.pixelfunc.ud_grade(t,nside)
    q = hp.pixelfunc.ud_grade(q,nside)
    u = hp.pixelfunc.ud_grade(u,nside)
    tlm,elm,blm = hp.map2alm([t,q,u],lmax=lmax)
else:
    tlm,elm,blm = hp.read_alm(file_alm,hdu=[1,2,3])
    tlm = utils.reduce_lmax(tlm,lmax=lmax)
    elm = utils.reduce_lmax(elm,lmax=lmax)
    blm = utils.reduce_lmax(blm,lmax=lmax)

# Get noise alms
np.random.seed(hash('tora')%2**32+sim)
nlmt,nlme,nlmb = hp.synalm([nltt,nlee,nlbb,nltt*0],new=True,lmax=lmax)

tlm += nlmt
elm += nlme
blm += nlmb

# Signal+Noise spectra
cltt = sltt + nltt
clee = slee + nlee
clbb = slbb + nlbb
clte = slte + nlte

if not gmv:
    print('Creating filters...')
    # Create 1/Nl filters
    flt = np.zeros(lmax+1); flt[100:] = 1./cltt[100:]
    fle = np.zeros(lmax+1); fle[100:] = 1./clee[100:]
    flb = np.zeros(lmax+1); flb[100:] = 1./clbb[100:]
    
    if est[0] == 'T': almbar1 = hp.almxfl(tlm,flt); flm1 = flt
    if est[0] == 'E': almbar1 = hp.almxfl(elm,fle); flm1 = fle
    if est[0] == 'B': almbar1 = hp.almxfl(blm,flb); flm1 = flb
    
    if est[1] == 'T': almbar2 = hp.almxfl(tlm,flt); flm2 = flt
    if est[1] == 'E': almbar2 = hp.almxfl(elm,fle); flm2 = fle
    if est[1] == 'B': almbar2 = hp.almxfl(blm,flb); flm2 = flb

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
    glm,clm = qest.qest(est,lmax,clfile,almbar1,almbar2,gmv=False)
    # Save plm and clm
    Path(dir_out).mkdir(parents=True, exist_ok=True)
    np.save(dir_out+f'/plm_{est}_healqest_seed{sim}_lmax{lmax}_nside{nside}_{append}.npy',glm)
    np.save(dir_out+f'/clm_{est}_healqest_seed{sim}_lmax{lmax}_nside{nside}_{append}.npy',clm)
else:
    glm,clm = qest.qest(est,lmax,clfile,alm1all,alm2all,gmv=True,totalcls=totalcls)
    # Save plm and clm
    Path(dir_out).mkdir(parents=True, exist_ok=True)
    np.save(dir_out+f'/plm_healqest_seed{sim}_lmax{lmax}_nside{nside}_{append}.npy',glm)
    np.save(dir_out+f'/clm_healqest_seed{sim}_lmax{lmax}_nside{nside}_{append}.npy',clm)
