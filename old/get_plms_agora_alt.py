#!/usr/bin/env python3
# Run like python3 get_plms.py TT append test_yuka.yaml
# Note: argument append should be 'agora_standard_rotatedcmb', 'agora_standard_gaussianfg'
import os, sys
import numpy as np
import healpy as hp
import pickle
from pathlib import Path
from time import time
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import healqest_utils as utils
import qest

def main():

    qe = str(sys.argv[1])
    append = str(sys.argv[2])
    config_file = str(sys.argv[3])

    time0 = time()

    config = utils.parse_yaml(config_file)
    lmax = config['lensrec']['lmax']
    nside = config['lensrec']['nside']
    dir_out = config['dir_out']
    lmaxT = config['lensrec']['lmaxT']
    lmaxP = config['lensrec']['lmaxP']
    lmin = config['lensrec']['lminT']
    cltype = config['lensrec']['cltype']
    cls = config['cls']
    sl = {ee:config['cls'][cltype][ee] for ee in config['cls'][cltype].keys()}
    l = np.arange(0,lmax+1)
    filename_sqe = dir_out+f'/plm_{qe}_healqest_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy'
    filename_gmv = dir_out+f'/plm_{qe}_healqest_gmv_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy'

    if os.path.isfile(filename_sqe) or os.path.isfile(filename_gmv):
        print('File already exists!')
    else:
        do_reconstruction(qe,append,config_file)

    elapsed = time() - time0
    elapsed /= 60
    print('Time taken (minutes): ', elapsed)

def do_reconstruction(qe,append,config_file):
    '''
    Function to do the actual reconstruction.
    '''
    print(f'Doing reconstruction for qe {qe}, append {append}')

    config = utils.parse_yaml(config_file)
    lmax = config['lensrec']['lmax']
    nside = config['lensrec']['nside']
    dir_out = config['dir_out']
    lmaxT = config['lensrec']['lmaxT']
    lmaxP = config['lensrec']['lmaxP']
    lmin = config['lensrec']['lminT']
    cltype = config['lensrec']['cltype']
    cls = config['cls']
    sl = {ee:config['cls'][cltype][ee] for ee in config['cls'][cltype].keys()}
    l = np.arange(0,lmax+1)
    u = None
    filename_sqe = dir_out+f'/plm_{qe}_healqest_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy'
    filename_gmv = dir_out+f'/plm_{qe}_healqest_gmv_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy'

    # Agora sims (TOTAL, CMB + foregrounds)
    agora_095 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_95ghz_alm_lmax4096.fits'
    agora_150 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_150ghz_alm_lmax4096.fits'
    agora_220 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_220ghz_alm_lmax4096.fits'

    # Lensed CMB-only Agora sims
    lcmb_095 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lcmb_spt3g_95ghz_alm_lmax4096.fits'
    lcmb_150 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lcmb_spt3g_150ghz_alm_lmax4096.fits'
    lcmb_220 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lcmb_spt3g_220ghz_alm_lmax4096.fits'

    # Full sky single frequency foreground sims
    flm_95ghz_sim1 = f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_95ghz_seed1_alm_lmax{lmax}.fits'
    flm_150ghz_sim1 = f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_150ghz_seed1_alm_lmax{lmax}.fits'
    flm_220ghz_sim1 = f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_220ghz_seed1_alm_lmax{lmax}.fits'

    # Noise curves
    fsky_corr=1
    noise_curves_090_090 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_090_090.txt'))
    noise_curves_150_150 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_150_150.txt'))
    noise_curves_220_220 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_220_220.txt'))
    noise_curves_090_150 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_090_150.txt'))
    noise_curves_090_220 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_090_220.txt'))
    noise_curves_150_220 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_150_220.txt'))

    # ILC weights
    # Dimension (3, 6001) for 90, 150, 220 GHz respectively
    w_tsz_null = np.loadtxt('ilc_weights/weights1d_TT_spt3g_cmbynull.dat')
    w_cib_null = np.load('ilc_weights/weights_cmb_mv_cmbfree_cibfreetwoSEDs_spt3g1920.npy',allow_pickle=True)
    w_cib_null_95 = w_cib_null.item()['cmbcibfree'][95][1]
    w_cib_null_150 = w_cib_null.item()['cmbcibfree'][150][1]
    w_cib_null_220 = w_cib_null.item()['cmbcibfree'][220][1]
    w_Tmv = np.loadtxt('ilc_weights/weights1d_TT_spt3g_cmbmv.dat')
    w_Emv = np.loadtxt('ilc_weights/weights1d_EE_spt3g_cmbmv.dat')
    w_Bmv = np.loadtxt('ilc_weights/weights1d_BB_spt3g_cmbmv.dat')

    if qe == 'TTEETE' or qe == 'TBEB' or qe == 'all' or qe == 'TTEETEprf':
        gmv = True
    elif qe == 'TT' or qe == 'TE' or  qe == 'ET' or qe == 'EE' or qe == 'TB' or  qe == 'BT' or qe == 'EB' or  qe == 'BE' or qe == 'TTprf' or qe == 'T1T2' or qe == 'T2T1':
        gmv = False
    else:
        print('Invalid qe!')

    # Get Agora sim (signal + foregrounds)
    print('Getting alms...')
    tlm_95, elm_95, blm_95 = hp.read_alm(agora_095,hdu=[1,2,3])
    tlm_150, elm_150, blm_150 = hp.read_alm(agora_150,hdu=[1,2,3])
    tlm_220, elm_220, blm_220 = hp.read_alm(agora_220,hdu=[1,2,3])

    # Get Agora lensed CMB-only
    tlm_lcmb_95, elm_lcmb_95, blm_lcmb_95 = hp.read_alm(lcmb_095,hdu=[1,2,3])
    tlm_lcmb_150, elm_lcmb_150, blm_lcmb_150 = hp.read_alm(lcmb_150,hdu=[1,2,3])
    tlm_lcmb_220, elm_lcmb_220, blm_lcmb_220 = hp.read_alm(lcmb_220,hdu=[1,2,3])

    # Get Agora foreground-only
    tlm_fg_95 = tlm_95 - tlm_lcmb_95; elm_fg_95 = elm_95 - elm_lcmb_95; blm_fg_95 = blm_95 - blm_lcmb_95
    tlm_fg_150 = tlm_150 - tlm_lcmb_150; elm_fg_150 = elm_150 - elm_lcmb_150; blm_fg_150 = blm_150 - blm_lcmb_150
    tlm_fg_220 = tlm_220 - tlm_lcmb_220; elm_fg_220 = elm_220 - elm_lcmb_220; blm_fg_220 = blm_220 - blm_lcmb_220

    if append == 'agora_standard_rotatedcmb' or append == 'agora_mh_rotatedcmb' or append == 'agora_crossilc_twoseds_rotatedcmb':
        # Rotate lensed CMB map
        r = hp.Rotator(np.array([np.pi/2,np.pi/2,0]))
        tlm_lcmb_95, elm_lcmb_95, blm_lcmb_95 = r.rotate_alm([tlm_lcmb_95, elm_lcmb_95, blm_lcmb_95])
        tlm_lcmb_150, elm_lcmb_150, blm_lcmb_150 = r.rotate_alm([tlm_lcmb_150, elm_lcmb_150, blm_lcmb_150])
        tlm_lcmb_220, elm_lcmb_220, blm_lcmb_220 = r.rotate_alm([tlm_lcmb_220, elm_lcmb_220, blm_lcmb_220])

        # Add rotated CMB-only map to unrotated total foreground-only map
        tlm_95 = tlm_lcmb_95 + tlm_fg_95; tlm_150 = tlm_lcmb_150 + tlm_fg_150; tlm_220 = tlm_lcmb_220 + tlm_fg_220
        elm_95 = elm_lcmb_95 + elm_fg_95; elm_150 = elm_lcmb_150 + elm_fg_150; elm_220 = elm_lcmb_220 + elm_fg_220
        blm_95 = blm_lcmb_95 + blm_fg_95; blm_150 = blm_lcmb_150 + blm_fg_150; blm_220 = blm_lcmb_220 + blm_fg_220

    elif append == 'agora_standard_separated':
        tlm_95 = tlm_lcmb_95 + tlm_fg_95; tlm_150 = tlm_lcmb_150 + tlm_fg_150; tlm_220 = tlm_lcmb_220 + tlm_fg_220
        elm_95 = elm_lcmb_95 + elm_fg_95; elm_150 = elm_lcmb_150 + elm_fg_150; elm_220 = elm_lcmb_220 + elm_fg_220
        blm_95 = blm_lcmb_95 + blm_fg_95; blm_150 = blm_lcmb_150 + blm_fg_150; blm_220 = blm_lcmb_220 + blm_fg_220

    elif append == 'agora_standard_gaussianfg' or append == 'agora_mh_gaussianfg' or append == 'agora_crossilc_twoseds_gaussianfg':
        # Get Gaussian foregrounds sim
        tflm1_95, eflm1_95, bflm1_95 = hp.read_alm(flm_95ghz_sim1,hdu=[1,2,3])
        tflm1_150, eflm1_150, bflm1_150 = hp.read_alm(flm_150ghz_sim1,hdu=[1,2,3])
        tflm1_220, eflm1_220, bflm1_220 = hp.read_alm(flm_220ghz_sim1,hdu=[1,2,3])

        # Add Gaussian foregrounds to CMB-only map
        tlm_95 = tlm_lcmb_95 + tflm1_95; tlm_150 = tlm_lcmb_150 + tflm1_150; tlm_220 = tlm_lcmb_220 + tflm1_220
        elm_95 = elm_lcmb_95 + eflm1_95; elm_150 = elm_lcmb_150 + eflm1_150; elm_220 = elm_lcmb_220 + eflm1_220
        blm_95 = blm_lcmb_95 + bflm1_95; blm_150 = blm_lcmb_150 + bflm1_150; blm_220 = blm_lcmb_220 + bflm1_220

    elif append == 'agora_standard_gaussiancmb':
        # Load Gaussian CMB maps
        alm_cmb_sim1 = f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/tqu1/len/alms/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb1_seed1_alm_lmax{lmax}.fits'
        tlm1,elm1,blm1 = hp.read_alm(alm_cmb_sim1,hdu=[1,2,3])
        tlm_150 = tlm1.copy(); tlm_220 = tlm1.copy(); tlm_95 = tlm1.copy()
        elm_150 = elm1.copy(); elm_220 = elm1.copy(); elm_95 = elm1.copy()
        blm_150 = blm1.copy(); blm_220 = blm1.copy(); blm_95 = blm1.copy()

        # Add Gaussian CMB-only map to total foreground-only map
        tlm_95 = tlm_95 + tlm_fg_95; tlm_150 = tlm_150 + tlm_fg_150; tlm_220 = tlm_220 + tlm_fg_220
        elm_95 = elm_95 + elm_fg_95; elm_150 = elm_150 + elm_fg_150; elm_220 = elm_220 + elm_fg_220
        blm_95 = blm_95 + blm_fg_95; blm_150 = blm_150 + blm_fg_150; blm_220 = blm_220 + blm_fg_220

    if append == 'agora_standard_rotatedcmb_gaussianfg':
        # Rotate lensed CMB map
        r = hp.Rotator(np.array([np.pi/2,np.pi/2,0]))
        tlm_lcmb_95, elm_lcmb_95, blm_lcmb_95 = r.rotate_alm([tlm_lcmb_95, elm_lcmb_95, blm_lcmb_95])
        tlm_lcmb_150, elm_lcmb_150, blm_lcmb_150 = r.rotate_alm([tlm_lcmb_150, elm_lcmb_150, blm_lcmb_150])
        tlm_lcmb_220, elm_lcmb_220, blm_lcmb_220 = r.rotate_alm([tlm_lcmb_220, elm_lcmb_220, blm_lcmb_220])

        # Get Gaussian foregrounds sim
        tflm1_95, eflm1_95, bflm1_95 = hp.read_alm(flm_95ghz_sim1,hdu=[1,2,3])
        tflm1_150, eflm1_150, bflm1_150 = hp.read_alm(flm_150ghz_sim1,hdu=[1,2,3])
        tflm1_220, eflm1_220, bflm1_220 = hp.read_alm(flm_220ghz_sim1,hdu=[1,2,3])

        # Add Gaussian foregrounds to CMB-only map
        tlm_95 = tlm_lcmb_95 + tflm1_95; tlm_150 = tlm_lcmb_150 + tflm1_150; tlm_220 = tlm_lcmb_220 + tflm1_220
        elm_95 = elm_lcmb_95 + eflm1_95; elm_150 = elm_lcmb_150 + eflm1_150; elm_220 = elm_lcmb_220 + eflm1_220
        blm_95 = blm_lcmb_95 + bflm1_95; blm_150 = blm_lcmb_150 + bflm1_150; blm_220 = blm_lcmb_220 + bflm1_220

    if append == 'agora_standard_rotatedgaussiancmb':
        # NOT LENSED
        # Rotate lensed CMB map
        r = hp.Rotator(np.array([np.pi/2,np.pi/2,0]))
        tlm_lcmb_95, elm_lcmb_95, blm_lcmb_95 = r.rotate_alm([tlm_lcmb_95, elm_lcmb_95, blm_lcmb_95])
        tlm_lcmb_150, elm_lcmb_150, blm_lcmb_150 = r.rotate_alm([tlm_lcmb_150, elm_lcmb_150, blm_lcmb_150])
        tlm_lcmb_220, elm_lcmb_220, blm_lcmb_220 = r.rotate_alm([tlm_lcmb_220, elm_lcmb_220, blm_lcmb_220])

        # Gaussian CMB maps with the same power spectra
        [cltt_090_090,cltt_150_150,cltt_220_220,cltt_090_150,cltt_150_220,cltt_090_220] = hp.alm2cl([tlm_lcmb_95,tlm_lcmb_150,tlm_lcmb_220])
        [clee_090_090,clee_150_150,clee_220_220,clee_090_150,clee_150_220,clee_090_220] = hp.alm2cl([elm_lcmb_95,elm_lcmb_150,elm_lcmb_220])
        [clbb_090_090,clbb_150_150,clbb_220_220,clbb_090_150,clbb_150_220,clbb_090_220] = hp.alm2cl([blm_lcmb_95,blm_lcmb_150,blm_lcmb_220])

        # Seed "A"
        np.random.seed(4190002645)
        tlm_lcmb_95,elm_lcmb_95,blm_lcmb_95 = hp.synalm([cltt_090_090,clee_090_090,clbb_090_090,cltt_090_090*0],new=True,lmax=lmax)

        # Seed "A"
        np.random.seed(4190002645)
        cltt_T2a = np.nan_to_num((cltt_090_150)**2 / cltt_090_090); clee_T2a = np.nan_to_num((clee_090_150)**2 / clee_090_090); clbb_T2a = np.nan_to_num((clbb_090_150)**2 / clbb_090_090)
        tlm_T2a,elm_T2a,blm_T2a = hp.synalm([cltt_T2a,clee_T2a,clbb_T2a,cltt_T2a*0],new=True,lmax=lmax)
        # Seed "B"
        np.random.seed(89052206)
        cltt_T2b = cltt_150_150 - cltt_T2a; clee_T2b = clee_150_150 - clee_T2a; clbb_T2b = clbb_150_150 - clbb_T2a
        tlm_T2b,elm_T2b,blm_T2b = hp.synalm([cltt_T2b,clee_T2b,clbb_T2b,cltt_T2b*0],new=True,lmax=lmax)
        tlm_lcmb_150 = tlm_T2a + tlm_T2b; elm_lcmb_150 = elm_T2a + elm_T2b; blm_lcmb_150 = blm_T2a + blm_T2b

        # Seed "A"
        np.random.seed(4190002645)
        cltt_T3a = np.nan_to_num((cltt_090_220)**2 / cltt_090_090); clee_T3a = np.nan_to_num((clee_090_220)**2 / clee_090_090); clbb_T3a = np.nan_to_num((clbb_090_220)**2 / clbb_090_090)
        tlm_T3a,elm_T3a,blm_T3a = hp.synalm([cltt_T3a,clee_T3a,clbb_T3a,cltt_T3a*0],new=True,lmax=lmax)
        # Seed "B"
        np.random.seed(89052206)
        cltt_T3b = np.nan_to_num((cltt_150_220 - cltt_090_150*cltt_090_220/cltt_090_090)**2 / cltt_T2b)
        clee_T3b = np.nan_to_num((clee_150_220 - clee_090_150*clee_090_220/clee_090_090)**2 / clee_T2b)
        clbb_T3b = np.nan_to_num((clbb_150_220 - clbb_090_150*clbb_090_220/clbb_090_090)**2 / clbb_T2b)
        tlm_T3b,elm_T3b,blm_T3b = hp.synalm([cltt_T3b,clee_T3b,clbb_T3b,cltt_T3b*0],new=True,lmax=lmax)
        # Seed "C"
        np.random.seed(978540195)
        cltt_T3c = cltt_220_220 - cltt_T3a - cltt_T3b; clee_T3c = clee_220_220 - clee_T3a - clee_T3b; clbb_T3c = clbb_220_220 - clbb_T3a - clbb_T3b
        tlm_T3c,elm_T3c,blm_T3c = hp.synalm([cltt_T3c,clee_T3c,clbb_T3c,cltt_T3c*0],new=True,lmax=lmax)
        tlm_lcmb_220 = tlm_T3a + tlm_T3b + tlm_T3c; elm_lcmb_220 = elm_T3a + elm_T3b + elm_T3c; blm_lcmb_220 = blm_T3a + blm_T3b + blm_T3c

        # Add rotated CMB-only map to unrotated total foreground-only map
        tlm_95 = tlm_lcmb_95 + tlm_fg_95; tlm_150 = tlm_lcmb_150 + tlm_fg_150; tlm_220 = tlm_lcmb_220 + tlm_fg_220
        elm_95 = elm_lcmb_95 + elm_fg_95; elm_150 = elm_lcmb_150 + elm_fg_150; elm_220 = elm_lcmb_220 + elm_fg_220
        blm_95 = blm_lcmb_95 + blm_fg_95; blm_150 = blm_lcmb_150 + blm_fg_150; blm_220 = blm_lcmb_220 + blm_fg_220

    # Adding noise! (Using sim 1)
    nlm1_090_filename = dir_out + f'nlm/nlm_090_lmax{lmax}_seed1.alm'
    nlm1_150_filename = dir_out + f'nlm/nlm_150_lmax{lmax}_seed1.alm'
    nlm1_220_filename = dir_out + f'nlm/nlm_220_lmax{lmax}_seed1.alm'
    nlmt1_090,nlme1_090,nlmb1_090 = hp.read_alm(nlm1_090_filename,hdu=[1,2,3])
    nlmt1_150,nlme1_150,nlmb1_150 = hp.read_alm(nlm1_150_filename,hdu=[1,2,3])
    nlmt1_220,nlme1_220,nlmb1_220 = hp.read_alm(nlm1_220_filename,hdu=[1,2,3])
    tlm_95 += nlmt1_090; tlm_150 += nlmt1_150; tlm_220 += nlmt1_220
    elm_95 += nlme1_090; elm_150 += nlme1_150; elm_220 += nlme1_220
    blm_95 += nlmb1_090; blm_150 += nlmb1_150; blm_220 += nlmb1_220

    tlm = hp.almxfl(tlm_95,w_Tmv[0][:lmax+1]) + hp.almxfl(tlm_150,w_Tmv[1][:lmax+1]) + hp.almxfl(tlm_220,w_Tmv[2][:lmax+1])
    tlm_tszn = hp.almxfl(tlm_95,w_tsz_null[0][:lmax+1]) + hp.almxfl(tlm_150,w_tsz_null[1][:lmax+1]) + hp.almxfl(tlm_220,w_tsz_null[2][:lmax+1])
    tlm_cibn = hp.almxfl(tlm_95,w_cib_null_95[:lmax+1]) + hp.almxfl(tlm_150,w_cib_null_150[:lmax+1]) + hp.almxfl(tlm_220,w_cib_null_220[:lmax+1])
    elm = hp.almxfl(elm_95,w_Emv[0][:lmax+1]) + hp.almxfl(elm_150,w_Emv[1][:lmax+1]) + hp.almxfl(elm_220,w_Emv[2][:lmax+1])
    blm = hp.almxfl(blm_95,w_Bmv[0][:lmax+1]) + hp.almxfl(blm_150,w_Bmv[1][:lmax+1]) + hp.almxfl(blm_220,w_Bmv[2][:lmax+1])

    # Get signal + noise residuals spectra for constructing fl filters
    print('Getting signal + noise residuals spectra for filtering')
    # If lmaxT != lmaxP, we add artificial noise in TT for ell > lmaxT
    artificial_noise = np.zeros(lmax+1)
    artificial_noise[lmaxT+2:] = 1.e10
    totalcls_filename = dir_out+f'totalcls/totalcls_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy'
    if os.path.isfile(totalcls_filename):
        totalcls = np.load(totalcls_filename)
        if append[6:14] == 'standard':
            cltt = totalcls[:,0]; clee = totalcls[:,1]; clbb = totalcls[:,2]; clte = totalcls[:,3]
        elif append[6:8] == 'mh' or append[6:22] == 'crossilc_twoseds':
            cltt1 = totalcls[:,4]; cltt2 = totalcls[:,5]; clttx = totalcls[:,6]; cltt3 = totalcls[:,0]; clee = totalcls[:,1]; clbb = totalcls[:,2]; clte = totalcls[:,3]
    elif append[6:14] == 'standard':
        print(f"totalcls file doesn't exist yet, getting the totalcls")
        cltt = hp.alm2cl(tlm,tlm) + artificial_noise
        clee = hp.alm2cl(elm,elm)
        clbb = hp.alm2cl(blm,blm)
        clte = hp.alm2cl(tlm,elm)
        totalcls = np.vstack((cltt,clee,clbb,clte)).T
        np.save(totalcls_filename,totalcls)
    elif append[6:8] == 'mh':
        cltt_mv = hp.alm2cl(tlm,tlm) + artificial_noise
        clee = hp.alm2cl(elm,elm)
        clbb = hp.alm2cl(blm,blm)
        clte = hp.alm2cl(tlm,elm)
        cltt_tszn = hp.alm2cl(tlm_tszn,tlm_tszn) + artificial_noise
        clt1t2 = hp.alm2cl(tlm,tlm_tszn) + artificial_noise
        clt2e = hp.alm2cl(elm,tlm_tszn)
        totalcls = np.vstack((cltt_mv,clee,clbb,clte,cltt_mv,cltt_tszn,clt1t2,cltt_mv,clt1t2,clte,clt2e)).T
        np.save(totalcls_filename,totalcls)
        cltt1 = totalcls[:,4]; cltt2 = totalcls[:,5]; clttx = totalcls[:,6]; cltt3 = totalcls[:,0]; clee = totalcls[:,1]; clbb = totalcls[:,2]; clte = totalcls[:,3]
    elif append[6:22] == 'crossilc_twoseds':
        cltt_mv = hp.alm2cl(tlm,tlm) + artificial_noise
        clee = hp.alm2cl(elm,elm)
        clbb = hp.alm2cl(blm,blm)
        clte = hp.alm2cl(tlm,elm)
        cltt_cibn = hp.alm2cl(tlm_cibn,tlm_cibn) + artificial_noise
        cltt_tszn = hp.alm2cl(tlm_tszn,tlm_tszn) + artificial_noise
        clt1t2 = hp.alm2cl(tlm_cibn,tlm_tszn) + artificial_noise
        clt1t3 = hp.alm2cl(tlm_cibn,tlm) + artificial_noise
        clt2t3 = hp.alm2cl(tlm_tszn,tlm) + artificial_noise
        clt1e = hp.alm2cl(elm,tlm_cibn)
        clt2e = hp.alm2cl(elm,tlm_tszn)
        totalcls = np.vstack((cltt_mv,clee,clbb,clte,cltt_cibn,cltt_tszn,clt1t2,clt1t3,clt2t3,clt1e,clt2e)).T
        np.save(totalcls_filename,totalcls)
        cltt1 = totalcls[:,4]; cltt2 = totalcls[:,5]; clttx = totalcls[:,6]; cltt3 = totalcls[:,0]; clee = totalcls[:,1]; clbb = totalcls[:,2]; clte = totalcls[:,3]

    if not gmv:
        print('Creating filters...')
        if append[6:14] == 'standard':
            flt = np.zeros(lmax+1); flt[lmin:] = 1./cltt[lmin:] # MV
            fle = np.zeros(lmax+1); fle[lmin:] = 1./clee[lmin:]
            flb = np.zeros(lmax+1); flb[lmin:] = 1./clbb[lmin:]

            if qe[0] == 'T': almbar1 = hp.almxfl(tlm,flt); flm1 = flt
            if qe[0] == 'E': almbar1 = hp.almxfl(elm,fle); flm1 = fle
            if qe[0] == 'B': almbar1 = hp.almxfl(blm,flb); flm1 = flb

            if qe[1] == 'T': almbar2 = hp.almxfl(tlm,flt); flm2 = flt
            if qe[1] == 'E': almbar2 = hp.almxfl(elm,fle); flm2 = fle
            if qe[1] == 'B': almbar2 = hp.almxfl(blm,flb); flm2 = flb
        elif append[6:8] == 'mh':
            # Create 1/cl filters
            flt1 = np.zeros(lmax+1); flt1[lmin:] = 1./cltt1[lmin:] # MV
            flt2 = np.zeros(lmax+1); flt2[lmin:] = 1./cltt2[lmin:] # tSZ-null
            fle = np.zeros(lmax+1); fle[lmin:] = 1./clee[lmin:]
            flb = np.zeros(lmax+1); flb[lmin:] = 1./clbb[lmin:]

            if qe[:2] == 'T1': almbar1 = hp.almxfl(tlm,flt1); flm1 = flt1
            elif qe[:2] == 'T2': almbar1 = hp.almxfl(tlm_tszn,flt2); flm1 = flt2
            elif qe[0] == 'T': almbar1 = hp.almxfl(tlm,flt1); flm1 = flt1
            elif qe[0] == 'E': almbar1 = hp.almxfl(elm,fle); flm1 = fle
            elif qe[0] == 'B': almbar1 = hp.almxfl(blm,flb); flm1 = flb

            if qe[2:4] == 'T1': almbar2 = hp.almxfl(tlm,flt1); flm2 = flt1
            elif qe[2:4] == 'T2': almbar2 = hp.almxfl(tlm_tszn,flt2); flm2 = flt2
            elif qe[1] == 'T': almbar2 = hp.almxfl(tlm,flt1); flm2 = flt1
            elif qe[1] == 'E': almbar2 = hp.almxfl(elm,fle); flm2 = fle
            elif qe[1] == 'B': almbar2 = hp.almxfl(blm,flb); flm2 = flb
        elif append[6:22] == 'crossilc_twoseds':
            # Create 1/cl filters
            flt1 = np.zeros(lmax+1); flt1[lmin:] = 1./cltt1[lmin:] # CIB-null
            flt2 = np.zeros(lmax+1); flt2[lmin:] = 1./cltt2[lmin:] # tSZ-null
            flt3 = np.zeros(lmax+1); flt3[lmin:] = 1./cltt3[lmin:] # MV
            fle = np.zeros(lmax+1); fle[lmin:] = 1./clee[lmin:]
            flb = np.zeros(lmax+1); flb[lmin:] = 1./clbb[lmin:]

            if qe[:2] == 'T1': almbar1 = hp.almxfl(tlm_cibn,flt1); flm1 = flt1
            elif qe[:2] == 'T2': almbar1 = hp.almxfl(tlm_tszn,flt2); flm1 = flt2
            elif qe[0] == 'T': almbar1 = hp.almxfl(tlm,flt3); flm1 = flt3
            elif qe[0] == 'E': almbar1 = hp.almxfl(elm,fle); flm1 = fle
            elif qe[0] == 'B': almbar1 = hp.almxfl(blm,flb); flm1 = flb

            if qe[2:4] == 'T1': almbar2 = hp.almxfl(tlm_cibn,flt1); flm2 = flt1
            elif qe[2:4] == 'T2': almbar2 = hp.almxfl(tlm_tszn,flt2); flm2 = flt2
            elif qe[1] == 'T': almbar2 = hp.almxfl(tlm,flt3); flm2 = flt3
            elif qe[1] == 'E': almbar2 = hp.almxfl(elm,fle); flm2 = fle
            elif qe[1] == 'B': almbar2 = hp.almxfl(blm,flb); flm2 = flb
    else:
        if append[6:14] == 'standard':
            print('Doing the 1/Dl for GMV...')
            invDl = np.zeros(lmax+1, dtype=np.complex_)
            invDl[lmin:] = 1./(cltt[lmin:]*clee[lmin:] - clte[lmin:]**2)
            flb = np.zeros(lmax+1); flb[lmin:] = 1./clbb[lmin:]

            # Order is TT, EE, TE, ET, TB, BT, EB, BE
            alm1all = np.zeros((len(tlm),8), dtype=np.complex_)
            alm2all = np.zeros((len(tlm),8), dtype=np.complex_)
            # TT
            alm1all[:,0] = hp.almxfl(tlm,invDl)
            alm2all[:,0] = hp.almxfl(tlm,invDl)
            # EE
            alm1all[:,1] = hp.almxfl(elm,invDl)
            alm2all[:,1] = hp.almxfl(elm,invDl)
            # TE
            alm1all[:,2] = hp.almxfl(tlm,invDl)
            alm2all[:,2] = hp.almxfl(elm,invDl)
            # ET
            alm1all[:,3] = hp.almxfl(elm,invDl)
            alm2all[:,3] = hp.almxfl(tlm,invDl)
            # TB
            alm1all[:,4] = hp.almxfl(tlm,invDl)
            alm2all[:,4] = hp.almxfl(blm,flb)
            # BT
            alm1all[:,5] = hp.almxfl(blm,flb)
            alm2all[:,5] = hp.almxfl(tlm,invDl)
            # EB
            alm1all[:,6] = hp.almxfl(elm,invDl)
            alm2all[:,6] = hp.almxfl(blm,flb)
            # BE
            alm1all[:,7] = hp.almxfl(blm,flb)
            alm2all[:,7] = hp.almxfl(elm,invDl)
        elif append[6:8] == 'mh':
            print('Doing the 1/Dl for GMV...')
            invDl1 = np.zeros(lmax+1, dtype=np.complex_)
            invDl2 = np.zeros(lmax+1, dtype=np.complex_)
            invDl3 = np.zeros(lmax+1, dtype=np.complex_)
            invDl1[lmin:] = 1./(cltt1[lmin:]*clee[lmin:] - clte[lmin:]**2)
            invDl2[lmin:] = 1./(cltt2[lmin:]*clee[lmin:] - clte[lmin:]**2)
            invDl3[lmin:] = 1./(cltt3[lmin:]*clee[lmin:] - clte[lmin:]**2)
            flb = np.zeros(lmax+1); flb[lmin:] = 1./clbb[lmin:]

            # Order is T1T2, T2T1, EE, TE, ET, TB, BT, EB, BE
            alm1all = np.zeros((len(tlm),9), dtype=np.complex_)
            alm2all = np.zeros((len(tlm_tszn),9), dtype=np.complex_)
            # T1T2
            alm1all[:,0] = hp.almxfl(tlm,invDl1)
            alm2all[:,0] = hp.almxfl(tlm_tszn,invDl2)
            # T2T1
            alm1all[:,1] = hp.almxfl(tlm_tszn,invDl2)
            alm2all[:,1] = hp.almxfl(tlm,invDl1)
            # EE
            alm1all[:,2] = hp.almxfl(elm,invDl3)
            alm2all[:,2] = hp.almxfl(elm,invDl3)
            # TE
            alm1all[:,3] = hp.almxfl(tlm,invDl3)
            alm2all[:,3] = hp.almxfl(elm,invDl3)
            # ET
            alm1all[:,4] = hp.almxfl(elm,invDl3)
            alm2all[:,4] = hp.almxfl(tlm,invDl3)
            # TB
            alm1all[:,5] = hp.almxfl(tlm,invDl3)
            alm2all[:,5] = hp.almxfl(blm,flb)
            # BT
            alm1all[:,6] = hp.almxfl(blm,flb)
            alm2all[:,6] = hp.almxfl(tlm,invDl3)
            # EB
            alm1all[:,7] = hp.almxfl(elm,invDl3)
            alm2all[:,7] = hp.almxfl(blm,flb)
            # BE
            alm1all[:,8] = hp.almxfl(blm,flb)
            alm2all[:,8] = hp.almxfl(elm,invDl3)
        elif append[6:22] == 'crossilc_twoseds':
            print('Doing the 1/Dl for GMV...')
            invDl1 = np.zeros(lmax+1, dtype=np.complex_)
            invDl2 = np.zeros(lmax+1, dtype=np.complex_)
            invDl3 = np.zeros(lmax+1, dtype=np.complex_)
            invDl1[lmin:] = 1./(cltt1[lmin:]*clee[lmin:] - clte[lmin:]**2)
            invDl2[lmin:] = 1./(cltt2[lmin:]*clee[lmin:] - clte[lmin:]**2)
            invDl3[lmin:] = 1./(cltt3[lmin:]*clee[lmin:] - clte[lmin:]**2)
            flb = np.zeros(lmax+1); flb[lmin:] = 1./clbb[lmin:]

            # Order is T1T2, T2T1, EE, TE, ET, TB, BT, EB, BE
            alm1all = np.zeros((len(tlm_cibn),9), dtype=np.complex_)
            alm2all = np.zeros((len(tlm_tszn),9), dtype=np.complex_)
            # T1T2
            alm1all[:,0] = hp.almxfl(tlm_cibn,invDl1)
            alm2all[:,0] = hp.almxfl(tlm_tszn,invDl2)
            # T2T1
            alm1all[:,1] = hp.almxfl(tlm_tszn,invDl2)
            alm2all[:,1] = hp.almxfl(tlm_cibn,invDl1)
            # EE
            alm1all[:,2] = hp.almxfl(elm,invDl3)
            alm2all[:,2] = hp.almxfl(elm,invDl3)
            # TE
            alm1all[:,3] = hp.almxfl(tlm,invDl3)
            alm2all[:,3] = hp.almxfl(elm,invDl3)
            # ET
            alm1all[:,4] = hp.almxfl(elm,invDl3)
            alm2all[:,4] = hp.almxfl(tlm,invDl3)
            # TB
            alm1all[:,5] = hp.almxfl(tlm,invDl3)
            alm2all[:,5] = hp.almxfl(blm,flb)
            # BT
            alm1all[:,6] = hp.almxfl(blm,flb)
            alm2all[:,6] = hp.almxfl(tlm,invDl3)
            # EB
            alm1all[:,7] = hp.almxfl(elm,invDl3)
            alm2all[:,7] = hp.almxfl(blm,flb)
            # BE
            alm1all[:,8] = hp.almxfl(blm,flb)
            alm2all[:,8] = hp.almxfl(elm,invDl3)

    # Run healqest
    print('Running healqest...')
    if not gmv:
        q_original = qest.qest(config,cls)
        if qe == 'T1T2' or qe == 'T2T1': qe='TT'
        glm,clm = q_original.eval(qe,almbar1,almbar2,u=u)
        # Save plm and clm
        Path(dir_out).mkdir(parents=True, exist_ok=True)
        np.save(filename_sqe,glm)
        return
    else:
        q_gmv = qest.qest_gmv(config,cls)
        if append[6:14] == 'standard':
            glm,clm = q_gmv.eval(qe,alm1all,alm2all,totalcls,u=u,crossilc=False)
        elif append[6:8] == 'mh' or append[6:22] == 'crossilc_twoseds':
            glm,clm = q_gmv.eval(qe,alm1all,alm2all,totalcls,u=u,crossilc=True)
        # Save plm and clm
        Path(dir_out).mkdir(parents=True, exist_ok=True)
        np.save(filename_gmv,glm)
        return

if __name__ == '__main__':

    main()
