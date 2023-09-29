#!/usr/bin/env python3
# Run like python3 get_plms_example.py TT 100 101
import os, sys
import numpy as np
import healpy as hp
from pathlib import Path
from time import time
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import utils
import qest

####################################
# EDIT THIS PART!
# Current setting for getting plms for sims with 200 mJy Poisson distributed point sources in T, no noise
config_file = 'profhrd_yuka.yaml'
config = utils.parse_yaml(config_file)
lmax = config['lmax']
# Even if your sims here don't have added noise (or foregrounds), you can keep noise_file (or u) defined to include the nl (or fgtt) in the filtering
# (e.g. when you are computing noiseless sims for the N1 calculation but you want the filter to still include nl to suppress modes exactly as in the signal map)
#noise_file='nl_cmbmv_20192020.dat'
noise_file = None
fsky_corr=25.308939726920805
#u = None
u = np.ones(lmax+1, dtype=np.complex_)

# Sims used for actual reconstruction and N0 calculation, lensed CMB + sources in T + noise added if noise_file is not None
append = f'tsrc_fluxlim0.200'
# Sims used for actual reconstruction and N0 calculation, lensed CMB + no foregrounds + noise added if noise_file is not None
#append = 'cmbonly'
# Sims used for N1 calculation, these are lensed with the same phi but different CMB realizations, no foregrounds or noise
#append = 'cmbonly_phi1_tqu1tqu2'
#append = 'cmbonly_phi1_tqu2tqu1'
# Sims used for N0 calculation for subtracting from N1, lensed CMB + no foregrounds + no noise
#append = 'noiseless_cmbonly'
# Unlensed sims + no foregrounds + noise added if noise_file is not None
#append = 'unl'
# Unlensed sims + foregrounds + noise added if noise_file is not None
#append = 'unl_with_fg'

# Full sky
# 200 mJy Poisson distributed point sources in T
flm_sim1 = f'/scratch/users/yukanaka/rand_ptsrc_rlz/src_fluxlim0.200_alm_set1_rlz{sim1}.fits'
flm_sim2 = f'/scratch/users/yukanaka/rand_ptsrc_rlz/src_fluxlim0.200_alm_set1_rlz{sim2}.fits'
# From amscott:/sptlocal/analysis/eete+lensing_19-20/resources/sims/planck2018/inputcmb/
alm_cmb_sim1 = f'/scratch/users/yukanaka/lensing19-20/inputcmb/tqu1/len/alms/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb1_seed{sim1}_alm_lmax{lmax}.fits'
alm_cmb_sim2 = f'/scratch/users/yukanaka/lensing19-20/inputcmb/tqu1/len/alms/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb1_seed{sim2}_alm_lmax{lmax}.fits'
alm_cmb_sim1_tqu2 = f'/scratch/users/yukanaka/lensing19-20/inputcmb/tqu2/len/alms/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb2_seed{sim1}_alm_lmax{lmax}.fits'
# Unlensed alms sampled from lensed theory spectra
unl_map_sim1 = f'/scratch/users/yukanaka/full_res_maps/unl_from_lensed_cls/unl_from_lensed_cls_seed{sim1}_lmax{lmax}_nside{nside}_20230905.fits'
unl_map_sim2 = f'/scratch/users/yukanaka/full_res_maps/unl_from_lensed_cls/unl_from_lensed_cls_seed{sim2}_lmax{lmax}_nside{nside}_20230905.fits'
####################################
time0 = time()

qe = str(sys.argv[1])
sim1 = int(sys.argv[2])
sim2 = int(sys.argv[3])
if qe == 'TTEETE' or qe == 'TBEB' or qe == 'all' or qe == 'TTEETEprf':
    gmv = True
elif qe == 'TT' or qe == 'TE' or qe == 'EE' or qe == 'TB' or qe == 'EB' or qe == 'TTprf':
    gmv = False
else:
    print('Invalid qe!')

config = utils.parse_yaml(config_file)
cambini = config['cambini']
dir_out = config['dir_out']
lmax = config['lmax']
lmaxT = config['lmaxt']
lmaxP = config['lmaxp']
lmin = config['lmint']
nside = config['nside']
cltype = config['cltype']
cls = config['cls']
sl = {ee:config['cls'][cltype][ee] for ee in config['cls'][cltype].keys()}
filename_sqe = dir_out+f'/plm_{qe}_healqest_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy'
filename_gmv = dir_out+f'/plm_{qe}_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy'

if os.path.isfile(filename_sqe) or os.path.isfile(filename_gmv):
    print('File already exists!')
else:
    print(f'Doing reconstruction for sims {sim1} and {sim2}, qe {qe}')

    # Get full sky CMB alms
    print('Getting alms...')
    if append == f'tsrc_fluxlim0.200' or append == 'cmbonly' or append == 'noiseless_cmbonly':
        tlm1,elm1,blm1 = hp.read_alm(alm_cmb_sim1,hdu=[1,2,3])
        tlm2,elm2,blm2 = hp.read_alm(alm_cmb_sim2,hdu=[1,2,3])
    elif append == 'cmbonly_phi1_tqu1tqu2':
        tlm1,elm1,blm1 = hp.read_alm(alm_cmb_sim1,hdu=[1,2,3])
        tlm2,elm2,blm2 = hp.read_alm(alm_cmb_sim1_tqu2,hdu=[1,2,3])
    elif append == 'cmbonly_phi1_tqu2tqu1':
        tlm1,elm1,blm1 = hp.read_alm(alm_cmb_sim1_tqu2,hdu=[1,2,3])
        tlm2,elm2,blm2 = hp.read_alm(alm_cmb_sim1,hdu=[1,2,3])
    elif append == 'unl' or append == 'unl_with_fg':
        t1,q1,u1 = hp.read_map(unl_map_sim1,field=[0,1,2])
        tlm1,elm1,blm1 = hp.map2alm([t1,q1,u1],lmax=lmax)
        t2,q2,u2 = hp.read_map(unl_map_sim2,field=[0,1,2])
        tlm2,elm2,blm2 = hp.map2alm([t2,q2,u2],lmax=lmax)
    tlm1 = utils.reduce_lmax(tlm1,lmax=lmax)
    elm1 = utils.reduce_lmax(elm1,lmax=lmax)
    blm1 = utils.reduce_lmax(blm1,lmax=lmax)
    tlm2 = utils.reduce_lmax(tlm2,lmax=lmax)
    elm2 = utils.reduce_lmax(elm2,lmax=lmax)
    blm2 = utils.reduce_lmax(blm2,lmax=lmax)

    # Adding foregrounds!
    if append == f'tsrc_fluxlim0.200' or append == 'unl_with_fg':
        flm1 = hp.read_alm(flm_sim1,hdu=[1])
        flm1 = utils.reduce_lmax(flm1,lmax=lmax)
        tlm1 += flm1
        flm2 = hp.read_alm(flm_sim2,hdu=[1])
        flm2 = utils.reduce_lmax(flm2,lmax=lmax)
        tlm2 += flm2
   
    # Adding noise!
    if append == f'tsrc_fluxlim0.200' or append == 'cmbonly' or append == 'unl' or append == 'unl_with_fg':
        if noise_file is not None:
            noise_curves = np.loadtxt(noise_file)
            nltt = fsky_corr * noise_curves[:,1]; nlee = fsky_corr * noise_curves[:,2]; nlbb = fsky_corr * noise_curves[:,2]
            nlm1_filename = f'/scratch/users/yukanaka/gmv/nlm/2019_2020_ilc_noise_nlm_lmax{lmax}_seed{sim1}.alm'
            nlm2_filename = f'/scratch/users/yukanaka/gmv/nlm/2019_2020_ilc_noise_nlm_lmax{lmax}_seed{sim2}.alm'
            if os.path.isfile(nlm1_filename):
                nlmt1,nlme1,nlmb1 = hp.read_alm(nlm1_filename,hdu=[1,2,3])
            else:
                np.random.seed(hash('tora')%2**32+sim1)
                nlmt1,nlme1,nlmb1 = hp.synalm([nltt,nlee,nlbb,nltt*0],new=True,lmax=lmax)
                hp.write_alm(nlm1_filename,[nlmt1,nlme1,nlmb1])
            if os.path.isfile(nlm2_filename):
                nlmt2,nlme2,nlmb2 = hp.read_alm(nlm2_filename,hdu=[1,2,3])
            else:
                np.random.seed(hash('tora')%2**32+sim2)
                nlmt2,nlme2,nlmb2 = hp.synalm([nltt,nlee,nlbb,nltt*0],new=True,lmax=lmax)
                hp.write_alm(nlm2_filename,[nlmt2,nlme2,nlmb2])
            tlm1 += nlmt1; elm1 += nlme1; blm1 += nlmb1
            tlm2 += nlmt2; elm2 += nlme2; blm2 += nlmb2
        else:
            pass

    # Get signal + noise spectra for constructing fl filters
    # Noise
    if noise_file is not None:
        noise_curves = np.loadtxt(noise_file)
        nltt = fsky_corr * noise_curves[:,1]; nlee = fsky_corr * noise_curves[:,2]; nlbb = fsky_corr * noise_curves[:,2]
    else:
        nltt = np.zeros(lmax+1); nlee = np.zeros(lmax+1); nlbb = np.zeros(lmax+1)
    # Foregrounds
    if u is not None:
        # Point source maps have a flat Cl power spectrum at 2.18e-05 uK^2
        fgtt =  np.ones(lmax+1) * 2.18e-5
    else:
        fgtt = np.zeros(lmax+1)
    # If lmaxT != lmaxP, we add artificial noise in TT for ell > lmaxT
    artificial_noise = np.zeros(lmax+1)
    artificial_noise[lmaxT+2:] = 1.e10
    # Resulting spectra
    cltt = sl['tt'][:lmax+1] + nltt[:lmax+1] + fgtt + artificial_noise
    clee = sl['ee'][:lmax+1] + nlee[:lmax+1]
    clbb = sl['bb'][:lmax+1] + nlbb[:lmax+1]
    clte = sl['te'][:lmax+1]
    
    if not gmv:
        print('Creating filters...')
        # Create 1/cl filters
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
    print('Running healqest...')
    if not gmv:
        q_original = qest.qest(config,cls)
        glm,clm = q_original.eval(qe,almbar1,almbar2,u=u)
        # Save plm and clm
        Path(dir_out).mkdir(parents=True, exist_ok=True)
        np.save(filename_sqe,glm)
    else:
        q_gmv = qest.qest_gmv(config,cls)
        glm,clm = q_gmv.eval(qe,alm1all,alm2all,totalcls,u=u)
        # Save plm and clm
        Path(dir_out).mkdir(parents=True, exist_ok=True)
        np.save(filename_gmv,glm)

elapsed = time() - time0
elapsed /= 60
print('Time taken (minutes): ', elapsed)
