#!/usr/bin/env python3
# Run like python3 get_plms.py T1T2 100 101 append test_yuka.yaml noT3
# Note: argument append should be either 'mh' (used for actual reconstruction and N0 calculation, lensed CMB + Yuuki's foreground sims + noise),
# 'mh_cmbonly_phi1_tqu1tqu2', 'mh_cmbonly_phi1_tqu2tqu1' (used for N1 calculation, these are lensed with the same phi but different CMB realizations, no foregrounds or noise),
# 'mh_cmbonly' (used for N0 calculation for subtracting from N1, lensed CMB + no foregrounds + no noise),
# 'mh_unl_cmbonly' (unlensed sims + no foregrounds + no noise), or 'mh_unl' (unlensed sims + foregrounds + noise)
# For cinv-style, append should be 'mh_cinv', 'mh_cmbonly_phi1_tqu1tqu2_cinv', etc.
# For cross-ILC, it's 'crossilc_twoseds' or 'crossilc_onesed' instead of 'mh'
# If MH or cross-ILC, have another argument 'noT3' or 'withT3'
# See submit script slurm_get_plms.sh
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
    if str(sys.argv[2]) == 'r':
        sim1 = str(sys.argv[2])
        sim2 = int(sys.argv[3])
    elif str(sys.argv[3]) =='r':
        sim1 = int(sys.argv[2])
        sim2 = str(sys.argv[3])
    else:
        sim1 = int(sys.argv[2])
        sim2 = int(sys.argv[3])
    append = str(sys.argv[4])
    config_file = str(sys.argv[5])
    T3_opt = str(sys.argv[6])

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

    if append[-4:] != 'cinv' and (qe == 'TT' or qe == 'TE' or  qe == 'ET' or qe == 'EE' or qe == 'TB' or  qe == 'BT' or qe == 'EB' or  qe == 'BE' or qe == 'TTprf' or qe == 'T1T2' or qe == 'T2T1' or qe == 'T2E1' or qe == 'E2T1' or qe == 'E2E1'):
        # SQE
        gmv = False
        filename = dir_out+f'/plm_{qe}_healqest_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_{T3_opt}.npy'
    else:
        # GMV
        gmv = True
        filename = dir_out+f'/plm_{qe}_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_{T3_opt}.npy'

    if append[-4:] == 'cinv' and gmv:
        cinv = True
    else:
        cinv = False

    print(f'cinv is {cinv}, gmv is {gmv}, filename {filename}')

    if os.path.isfile(filename):
        print('File already exists!')
    else:
        do_reconstruction(qe,sim1,sim2,append,config_file,filename,gmv,cinv,T3_opt)

    elapsed = time() - time0
    elapsed /= 60
    print('Time taken (minutes): ', elapsed)

def do_reconstruction(qe,sim1,sim2,append,config_file,filename,gmv,cinv,T3_opt):
    '''
    Function to do the actual reconstruction.
    '''
    print(f'Doing reconstruction for sims {sim1} and {sim2}, qe {qe}, append {append}')

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
    if cinv:
        append_shortened = append[:-5]
    else:
        append_shortened = append

    # Agora sims
    agora_095 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_95ghz_alm_lmax4096.fits'
    agora_150 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_150ghz_alm_lmax4096.fits'
    agora_220 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_220ghz_alm_lmax4096.fits'

    # Noise curves
    fsky_corr=1
    noise_curves_090_090 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_090_090.txt'))
    noise_curves_150_150 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_150_150.txt'))
    noise_curves_220_220 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_220_220.txt'))
    noise_curves_090_150 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_090_150.txt'))
    noise_curves_090_220 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_090_220.txt'))
    noise_curves_150_220 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_150_220.txt'))

    # Full sky single frequency foreground sims
    if sim1 == 'r':
        flm_150ghz_sim2 = f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_150ghz_seed{sim2}_alm_lmax{lmax}.fits'
        flm_220ghz_sim2 = f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_220ghz_seed{sim2}_alm_lmax{lmax}.fits'
        flm_95ghz_sim2 = f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_95ghz_seed{sim2}_alm_lmax{lmax}.fits'
    elif sim2 == 'r':
        flm_150ghz_sim1 = f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_150ghz_seed{sim1}_alm_lmax{lmax}.fits'
        flm_220ghz_sim1 = f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_220ghz_seed{sim1}_alm_lmax{lmax}.fits'
        flm_95ghz_sim1 = f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_95ghz_seed{sim1}_alm_lmax{lmax}.fits'
    else:
        flm_150ghz_sim1 = f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_150ghz_seed{sim1}_alm_lmax{lmax}.fits'
        flm_150ghz_sim2 = f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_150ghz_seed{sim2}_alm_lmax{lmax}.fits'
        flm_220ghz_sim1 = f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_220ghz_seed{sim1}_alm_lmax{lmax}.fits'
        flm_220ghz_sim2 = f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_220ghz_seed{sim2}_alm_lmax{lmax}.fits'
        flm_95ghz_sim1 = f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_95ghz_seed{sim1}_alm_lmax{lmax}.fits'
        flm_95ghz_sim2 = f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_95ghz_seed{sim2}_alm_lmax{lmax}.fits'

    # CMB is same at all frequencies; also full sky here
    # From amscott:/sptlocal/analysis/eete+lensing_19-20/resources/sims/planck2018/inputcmb/
    if sim1 == 'r':
        alm_cmb_sim2 = f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/tqu1/len/alms/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb1_seed{sim2}_alm_lmax{lmax}.fits'
    elif sim2 == 'r':
        alm_cmb_sim1 = f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/tqu1/len/alms/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb1_seed{sim1}_alm_lmax{lmax}.fits'
    else:
        alm_cmb_sim1 = f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/tqu1/len/alms/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb1_seed{sim1}_alm_lmax{lmax}.fits'
        alm_cmb_sim2 = f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/tqu1/len/alms/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb1_seed{sim2}_alm_lmax{lmax}.fits'
        alm_cmb_sim1_tqu2 = f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/tqu2/len/alms/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb2_seed{sim1}_alm_lmax{lmax}.fits'

    # Unlensed CMB alms sampled from lensed theory spectra
    if sim1 == 'r':
        unl_map_sim2 = f'/oak/stanford/orgs/kipac/users/yukanaka/unl_from_lensed_cls/unl_from_lensed_cls_seed{sim2}_lmax{lmax}_nside{nside}_20230905.fits'
    elif sim2 == 'r':
        unl_map_sim1 = f'/oak/stanford/orgs/kipac/users/yukanaka/unl_from_lensed_cls/unl_from_lensed_cls_seed{sim1}_lmax{lmax}_nside{nside}_20230905.fits'
    else:
        unl_map_sim1 = f'/oak/stanford/orgs/kipac/users/yukanaka/unl_from_lensed_cls/unl_from_lensed_cls_seed{sim1}_lmax{lmax}_nside{nside}_20230905.fits'
        unl_map_sim2 = f'/oak/stanford/orgs/kipac/users/yukanaka/unl_from_lensed_cls/unl_from_lensed_cls_seed{sim2}_lmax{lmax}_nside{nside}_20230905.fits'

    # ILC weights
    # Dimension (3, 6001) for 90, 150, 220 GHz respectively
    w_tsz_null = np.loadtxt('ilc_weights/weights1d_TT_spt3g_cmbynull.dat')
    w_Tmv = np.loadtxt('ilc_weights/weights1d_TT_spt3g_cmbmv.dat')
    w_Emv = np.loadtxt('ilc_weights/weights1d_EE_spt3g_cmbmv.dat')
    w_Bmv = np.loadtxt('ilc_weights/weights1d_BB_spt3g_cmbmv.dat')
    # These are from Srini... Weird format. Assumes either one or two spectral energy distributions for CIB
    w_srini = np.load('ilc_weights/weights_cmb_mv_cmbfree_cibfree_spt3g1920.npy',allow_pickle=True)
    w_Tmv_srini_95 = w_srini.item()['cmbmv'][95][1]
    w_Tmv_srini_150 = w_srini.item()['cmbmv'][150][1]
    w_Tmv_srini_220 = w_srini.item()['cmbmv'][220][1]
    w_Tmv_srini = np.vstack((w_Tmv_srini_95,w_Tmv_srini_150,w_Tmv_srini_220))
    if append == 'crossilc_onesed' or append == 'crossilc_onesed_cinv':
        w_cib_null = np.load('ilc_weights/weights_cmb_mv_cmbfree_cibfree_spt3g1920.npy',allow_pickle=True)
        w_cib_null_95 = w_cib_null.item()['cmbcibfree'][95][1]
        w_cib_null_150 = w_cib_null.item()['cmbcibfree'][150][1]
        w_cib_null_220 = w_cib_null.item()['cmbcibfree'][220][1]
    elif append == 'crossilc_twoseds' or append == 'crossilc_twoseds_cinv':
        w_cib_null = np.load('ilc_weights/weights_cmb_mv_cmbfree_cibfreetwoSEDs_spt3g1920.npy',allow_pickle=True)
        w_cib_null_95 = w_cib_null.item()['cmbcibfree'][95][1]
        w_cib_null_150 = w_cib_null.item()['cmbcibfree'][150][1]
        w_cib_null_220 = w_cib_null.item()['cmbcibfree'][220][1]

    # Get Agora sim (signal + foregrounds)
    print('Getting alms...')
    tlm_95_agora, elm_95_agora, blm_95_agora = hp.read_alm(agora_095,hdu=[1,2,3])
    tlm_150_agora, elm_150_agora, blm_150_agora = hp.read_alm(agora_150,hdu=[1,2,3])
    tlm_220_agora, elm_220_agora, blm_220_agora = hp.read_alm(agora_220,hdu=[1,2,3])

    # Get full sky CMB alms
    print('Getting alms...')
    if append_shortened == 'mh' or append_shortened == 'mh_cmbonly' or append_shortened == 'crossilc_twoseds' or append_shortened == 'crossilc_twoseds_cmbonly' or append_shortened == 'crossilc_onesed' or append_shortened == 'crossilc_onesed_cmbonly':
        if sim1 == 'r':
            tlm2,elm2,blm2 = hp.read_alm(alm_cmb_sim2,hdu=[1,2,3])
        elif sim2 == 'r':
            tlm1,elm1,blm1 = hp.read_alm(alm_cmb_sim1,hdu=[1,2,3])
        else:
            tlm1,elm1,blm1 = hp.read_alm(alm_cmb_sim1,hdu=[1,2,3])
            tlm2,elm2,blm2 = hp.read_alm(alm_cmb_sim2,hdu=[1,2,3])
    elif append_shortened == 'mh_cmbonly_phi1_tqu1tqu2' or append_shortened == 'crossilc_twoseds_cmbonly_phi1_tqu1tqu2' or append_shortened == 'crossilc_onesed_cmbonly_phi1_tqu1tqu2':
        tlm1,elm1,blm1 = hp.read_alm(alm_cmb_sim1,hdu=[1,2,3])
        tlm2,elm2,blm2 = hp.read_alm(alm_cmb_sim1_tqu2,hdu=[1,2,3])
    elif append_shortened == 'mh_cmbonly_phi1_tqu2tqu1' or  append_shortened == 'crossilc_twoseds_cmbonly_phi1_tqu2tqu1' or append_shortened == 'crossilc_onesed_cmbonly_phi1_tqu2tqu1':
        tlm1,elm1,blm1 = hp.read_alm(alm_cmb_sim1_tqu2,hdu=[1,2,3])
        tlm2,elm2,blm2 = hp.read_alm(alm_cmb_sim1,hdu=[1,2,3])
    elif append_shortened == 'mh_unl_cmbonly' or append_shortened == 'mh_unl' or append_shortened == 'crossilc_twoseds_unl_cmbonly' or append_shortened == 'crossilc_twoseds_unl' or append_shortened == 'crossilc_onesed_unl_cmbonly' or append_shortened == 'crossilc_onesed_unl':
        if sim1 == 'r':
            t2,q2,u2 = hp.read_map(unl_map_sim2,field=[0,1,2])
            tlm2,elm2,blm2 = hp.map2alm([t2,q2,u2],lmax=lmax)
        elif sim2 == 'r':
            t1,q1,u1 = hp.read_map(unl_map_sim1,field=[0,1,2])
            tlm1,elm1,blm1 = hp.map2alm([t1,q1,u1],lmax=lmax)
        else:
            t1,q1,u1 = hp.read_map(unl_map_sim1,field=[0,1,2])
            tlm1,elm1,blm1 = hp.map2alm([t1,q1,u1],lmax=lmax)
            t2,q2,u2 = hp.read_map(unl_map_sim2,field=[0,1,2])
            tlm2,elm2,blm2 = hp.map2alm([t2,q2,u2],lmax=lmax)
    #tlm1 = utils.reduce_lmax(tlm1,lmax=lmax)
    #elm1 = utils.reduce_lmax(elm1,lmax=lmax)
    #blm1 = utils.reduce_lmax(blm1,lmax=lmax)
    #tlm2 = utils.reduce_lmax(tlm2,lmax=lmax)
    #elm2 = utils.reduce_lmax(elm2,lmax=lmax)
    #blm2 = utils.reduce_lmax(blm2,lmax=lmax)
    if append_shortened == 'mh' or append_shortened == 'mh_unl' or append_shortened == 'crossilc_twoseds' or append_shortened == 'crossilc_twoseds_unl' or append_shortened == 'crossilc_onesed' or append_shortened == 'crossilc_onesed_unl':
        if sim1 == 'r':
            tlm2_150 = tlm2.copy(); tlm2_220 = tlm2.copy(); tlm2_95 = tlm2.copy()
            elm2_150 = elm2.copy(); elm2_220 = elm2.copy(); elm2_95 = elm2.copy()
            blm2_150 = blm2.copy(); blm2_220 = blm2.copy(); blm2_95 = blm2.copy()
        elif sim2 == 'r':
            tlm1_150 = tlm1.copy(); tlm1_220 = tlm1.copy(); tlm1_95 = tlm1.copy()
            elm1_150 = elm1.copy(); elm1_220 = elm1.copy(); elm1_95 = elm1.copy()
            blm1_150 = blm1.copy(); blm1_220 = blm1.copy(); blm1_95 = blm1.copy()
        else:
            tlm1_150 = tlm1.copy(); tlm1_220 = tlm1.copy(); tlm1_95 = tlm1.copy()
            elm1_150 = elm1.copy(); elm1_220 = elm1.copy(); elm1_95 = elm1.copy()
            blm1_150 = blm1.copy(); blm1_220 = blm1.copy(); blm1_95 = blm1.copy()
            tlm2_150 = tlm2.copy(); tlm2_220 = tlm2.copy(); tlm2_95 = tlm2.copy()
            elm2_150 = elm2.copy(); elm2_220 = elm2.copy(); elm2_95 = elm2.copy()
            blm2_150 = blm2.copy(); blm2_220 = blm2.copy(); blm2_95 = blm2.copy()

    # Adding foregrounds!
    if append_shortened == 'mh' or append_shortened == 'mh_unl' or append_shortened == 'crossilc_twoseds' or append_shortened == 'crossilc_twoseds_unl' or append_shortened == 'crossilc_onesed' or append_shortened == 'crossilc_onesed_unl':
        if sim1 == 'r':
            tlm1_150 = tlm_150_agora; tlm1_220 = tlm_220_agora; tlm1_95 = tlm_95_agora
            elm1_150 = elm_150_agora; elm1_220 = elm_220_agora; elm1_95 = elm_95_agora
            blm1_150 = blm_150_agora; blm1_220 = blm_220_agora; blm1_95 = blm_95_agora

            tflm2_150, eflm2_150, bflm2_150 = hp.read_alm(flm_150ghz_sim2,hdu=[1,2,3])
            tflm2_150 = utils.reduce_lmax(tflm2_150,lmax=lmax); eflm2_150 = utils.reduce_lmax(eflm2_150,lmax=lmax); bflm2_150 = utils.reduce_lmax(bflm2_150,lmax=lmax)
            tflm2_220, eflm2_220, bflm2_220 = hp.read_alm(flm_220ghz_sim2,hdu=[1,2,3])
            tflm2_220 = utils.reduce_lmax(tflm2_220,lmax=lmax); eflm2_220 = utils.reduce_lmax(eflm2_220,lmax=lmax); bflm2_220 = utils.reduce_lmax(bflm2_220,lmax=lmax)
            tflm2_95, eflm2_95, bflm2_95 = hp.read_alm(flm_95ghz_sim2,hdu=[1,2,3])
            tflm2_95 = utils.reduce_lmax(tflm2_95,lmax=lmax); eflm2_95 = utils.reduce_lmax(eflm2_95,lmax=lmax); bflm2_95 = utils.reduce_lmax(bflm2_95,lmax=lmax)
            tlm2_150 += tflm2_150; tlm2_220 += tflm2_220; tlm2_95 += tflm2_95
            elm2_150 += eflm2_150; elm2_220 += eflm2_220; elm2_95 += eflm2_95
            blm2_150 += bflm2_150; blm2_220 += bflm2_220; blm2_95 += bflm2_95

            sim1 = 999
        elif sim2 == 'r':
            tflm1_150, eflm1_150, bflm1_150 = hp.read_alm(flm_150ghz_sim1,hdu=[1,2,3])
            tflm1_150 = utils.reduce_lmax(tflm1_150,lmax=lmax); eflm1_150 = utils.reduce_lmax(eflm1_150,lmax=lmax); bflm1_150 = utils.reduce_lmax(bflm1_150,lmax=lmax)
            tflm1_220, eflm1_220, bflm1_220 = hp.read_alm(flm_220ghz_sim1,hdu=[1,2,3])
            tflm1_220 = utils.reduce_lmax(tflm1_220,lmax=lmax); eflm1_220 = utils.reduce_lmax(eflm1_220,lmax=lmax); bflm1_220 = utils.reduce_lmax(bflm1_220,lmax=lmax)
            tflm1_95, eflm1_95, bflm1_95 = hp.read_alm(flm_95ghz_sim1,hdu=[1,2,3])
            tflm1_95 = utils.reduce_lmax(tflm1_95,lmax=lmax); eflm1_95 = utils.reduce_lmax(eflm1_95,lmax=lmax); bflm1_95 = utils.reduce_lmax(bflm1_95,lmax=lmax)
            tlm1_150 += tflm1_150; tlm1_220 += tflm1_220; tlm1_95 += tflm1_95
            elm1_150 += eflm1_150; elm1_220 += eflm1_220; elm1_95 += eflm1_95
            blm1_150 += bflm1_150; blm1_220 += bflm1_220; blm1_95 += bflm1_95

            tlm2_150 = tlm_150_agora; tlm2_220 = tlm_220_agora; tlm2_95 = tlm_95_agora
            elm2_150 = elm_150_agora; elm2_220 = elm_220_agora; elm2_95 = elm_95_agora
            blm2_150 = blm_150_agora; blm2_220 = blm_220_agora; blm2_95 = blm_95_agora

            sim2 = 999
        else:
            tflm1_150, eflm1_150, bflm1_150 = hp.read_alm(flm_150ghz_sim1,hdu=[1,2,3])
            tflm1_150 = utils.reduce_lmax(tflm1_150,lmax=lmax); eflm1_150 = utils.reduce_lmax(eflm1_150,lmax=lmax); bflm1_150 = utils.reduce_lmax(bflm1_150,lmax=lmax)
            tflm1_220, eflm1_220, bflm1_220 = hp.read_alm(flm_220ghz_sim1,hdu=[1,2,3])
            tflm1_220 = utils.reduce_lmax(tflm1_220,lmax=lmax); eflm1_220 = utils.reduce_lmax(eflm1_220,lmax=lmax); bflm1_220 = utils.reduce_lmax(bflm1_220,lmax=lmax)
            tflm1_95, eflm1_95, bflm1_95 = hp.read_alm(flm_95ghz_sim1,hdu=[1,2,3])
            tflm1_95 = utils.reduce_lmax(tflm1_95,lmax=lmax); eflm1_95 = utils.reduce_lmax(eflm1_95,lmax=lmax); bflm1_95 = utils.reduce_lmax(bflm1_95,lmax=lmax)
            tlm1_150 += tflm1_150; tlm1_220 += tflm1_220; tlm1_95 += tflm1_95
            elm1_150 += eflm1_150; elm1_220 += eflm1_220; elm1_95 += eflm1_95
            blm1_150 += bflm1_150; blm1_220 += bflm1_220; blm1_95 += bflm1_95

            tflm2_150, eflm2_150, bflm2_150 = hp.read_alm(flm_150ghz_sim2,hdu=[1,2,3])
            tflm2_150 = utils.reduce_lmax(tflm2_150,lmax=lmax); eflm2_150 = utils.reduce_lmax(eflm2_150,lmax=lmax); bflm2_150 = utils.reduce_lmax(bflm2_150,lmax=lmax)
            tflm2_220, eflm2_220, bflm2_220 = hp.read_alm(flm_220ghz_sim2,hdu=[1,2,3])
            tflm2_220 = utils.reduce_lmax(tflm2_220,lmax=lmax); eflm2_220 = utils.reduce_lmax(eflm2_220,lmax=lmax); bflm2_220 = utils.reduce_lmax(bflm2_220,lmax=lmax)
            tflm2_95, eflm2_95, bflm2_95 = hp.read_alm(flm_95ghz_sim2,hdu=[1,2,3])
            tflm2_95 = utils.reduce_lmax(tflm2_95,lmax=lmax); eflm2_95 = utils.reduce_lmax(eflm2_95,lmax=lmax); bflm2_95 = utils.reduce_lmax(bflm2_95,lmax=lmax)
            tlm2_150 += tflm2_150; tlm2_220 += tflm2_220; tlm2_95 += tflm2_95
            elm2_150 += eflm2_150; elm2_220 += eflm2_220; elm2_95 += eflm2_95
            blm2_150 += bflm2_150; blm2_220 += bflm2_220; blm2_95 += bflm2_95

    # Adding noise!
    if append_shortened == 'mh' or append_shortened == 'mh_unl' or append_shortened == 'crossilc_twoseds' or append_shortened == 'crossilc_twoseds_unl' or append_shortened == 'crossilc_onesed' or append_shortened == 'crossilc_onesed_unl':
        nltt_090_090 = fsky_corr * noise_curves_090_090[:,1]; nlee_090_090 = fsky_corr * noise_curves_090_090[:,2]; nlbb_090_090 = fsky_corr * noise_curves_090_090[:,3]
        nltt_150_150 = fsky_corr * noise_curves_150_150[:,1]; nlee_150_150 = fsky_corr * noise_curves_150_150[:,2]; nlbb_150_150 = fsky_corr * noise_curves_150_150[:,3]
        nltt_220_220 = fsky_corr * noise_curves_220_220[:,1]; nlee_220_220 = fsky_corr * noise_curves_220_220[:,2]; nlbb_220_220 = fsky_corr * noise_curves_220_220[:,3]
        nltt_090_150 = fsky_corr * noise_curves_090_150[:,1]; nlee_090_150 = fsky_corr * noise_curves_090_150[:,2]; nlbb_090_150 = fsky_corr * noise_curves_090_150[:,3]
        nltt_090_220 = fsky_corr * noise_curves_090_220[:,1]; nlee_090_220 = fsky_corr * noise_curves_090_220[:,2]; nlbb_090_220 = fsky_corr * noise_curves_090_220[:,3]
        nltt_150_220 = fsky_corr * noise_curves_150_220[:,1]; nlee_150_220 = fsky_corr * noise_curves_150_220[:,2]; nlbb_150_220 = fsky_corr * noise_curves_150_220[:,3]
        nlm1_090_filename = dir_out + f'nlm/nlm_090_lmax{lmax}_seed{sim1}.alm'
        nlm1_150_filename = dir_out + f'nlm/nlm_150_lmax{lmax}_seed{sim1}.alm'
        nlm1_220_filename = dir_out + f'nlm/nlm_220_lmax{lmax}_seed{sim1}.alm'
        nlm2_090_filename = dir_out + f'nlm/nlm_090_lmax{lmax}_seed{sim2}.alm'
        nlm2_150_filename = dir_out + f'nlm/nlm_150_lmax{lmax}_seed{sim2}.alm'
        nlm2_220_filename = dir_out + f'nlm/nlm_220_lmax{lmax}_seed{sim2}.alm'
        if os.path.isfile(nlm1_090_filename):
            nlmt1_090,nlme1_090,nlmb1_090 = hp.read_alm(nlm1_090_filename,hdu=[1,2,3])
            nlmt1_150,nlme1_150,nlmb1_150 = hp.read_alm(nlm1_150_filename,hdu=[1,2,3])
            nlmt1_220,nlme1_220,nlmb1_220 = hp.read_alm(nlm1_220_filename,hdu=[1,2,3])
        else:
            # See appendix of https://arxiv.org/pdf/0801.4380.pdf
            # Need to generate frequency correlated noise realizations
            # Don't have to worry about anti-correlation between frequencies/switching signs as mentioned in the paper, because our cross spectra are positive above ell of 300
            # Seed "A"
            np.random.seed(4190002645+sim1)
            nlmt1_090,nlme1_090,nlmb1_090 = hp.synalm([nltt_090_090,nlee_090_090,nlbb_090_090,nltt_090_090*0],new=True,lmax=lmax)

            # Seed "A"
            # Quick note, the hash part returns a different value for different python processes
            np.random.seed(4190002645+sim1)
            nltt_T2a = np.nan_to_num((nltt_090_150)**2 / nltt_090_090); nlee_T2a = np.nan_to_num((nlee_090_150)**2 / nlee_090_090); nlbb_T2a = np.nan_to_num((nlbb_090_150)**2 / nlbb_090_090)
            nlmt1_T2a,nlme1_T2a,nlmb1_T2a = hp.synalm([nltt_T2a,nlee_T2a,nlbb_T2a,nltt_T2a*0],new=True,lmax=lmax)
            # Seed "B"
            np.random.seed(89052206+sim1)
            nltt_T2b = nltt_150_150 - nltt_T2a; nlee_T2b = nlee_150_150 - nlee_T2a; nlbb_T2b = nlbb_150_150 - nlbb_T2a
            nlmt1_T2b,nlme1_T2b,nlmb1_T2b = hp.synalm([nltt_T2b,nlee_T2b,nlbb_T2b,nltt_T2b*0],new=True,lmax=lmax)
            nlmt1_150 = nlmt1_T2a + nlmt1_T2b; nlme1_150 = nlme1_T2a + nlme1_T2b; nlmb1_150 = nlmb1_T2a + nlmb1_T2b

            # Seed "A"
            np.random.seed(4190002645+sim1)
            nltt_T3a = np.nan_to_num((nltt_090_220)**2 / nltt_090_090); nlee_T3a = np.nan_to_num((nlee_090_220)**2 / nlee_090_090); nlbb_T3a = np.nan_to_num((nlbb_090_220)**2 / nlbb_090_090)
            nlmt1_T3a,nlme1_T3a,nlmb1_T3a = hp.synalm([nltt_T3a,nlee_T3a,nlbb_T3a,nltt_T3a*0],new=True,lmax=lmax)
            # Seed "B"
            np.random.seed(89052206+sim1)
            nltt_T3b = np.nan_to_num((nltt_150_220 - nltt_090_150*nltt_090_220/nltt_090_090)**2 / nltt_T2b)
            nlee_T3b = np.nan_to_num((nlee_150_220 - nlee_090_150*nlee_090_220/nlee_090_090)**2 / nlee_T2b)
            nlbb_T3b = np.nan_to_num((nlbb_150_220 - nlbb_090_150*nlbb_090_220/nlbb_090_090)**2 / nlbb_T2b)
            nlmt1_T3b,nlme1_T3b,nlmb1_T3b = hp.synalm([nltt_T3b,nlee_T3b,nlbb_T3b,nltt_T3b*0],new=True,lmax=lmax)
            # Seed "C"
            np.random.seed(978540195+sim1)
            nltt_T3c = nltt_220_220 - nltt_T3a - nltt_T3b; nlee_T3c = nlee_220_220 - nlee_T3a - nlee_T3b; nlbb_T3c = nlbb_220_220 - nlbb_T3a - nlbb_T3b
            nlmt1_T3c,nlme1_T3c,nlmb1_T3c = hp.synalm([nltt_T3c,nlee_T3c,nlbb_T3c,nltt_T3c*0],new=True,lmax=lmax)
            nlmt1_220 = nlmt1_T3a + nlmt1_T3b + nlmt1_T3c; nlme1_220 = nlme1_T3a + nlme1_T3b + nlme1_T3c; nlmb1_220 = nlmb1_T3a + nlmb1_T3b + nlmb1_T3c

            Path(dir_out+f'/nlm/').mkdir(parents=True, exist_ok=True)
            hp.write_alm(nlm1_090_filename,[nlmt1_090,nlme1_090,nlmb1_090])
            hp.write_alm(nlm1_150_filename,[nlmt1_150,nlme1_150,nlmb1_150])
            hp.write_alm(nlm1_220_filename,[nlmt1_220,nlme1_220,nlmb1_220])
        if os.path.isfile(nlm2_090_filename):
            nlmt2_090,nlme2_090,nlmb2_090 = hp.read_alm(nlm2_090_filename,hdu=[1,2,3])
            nlmt2_150,nlme2_150,nlmb2_150 = hp.read_alm(nlm2_150_filename,hdu=[1,2,3])
            nlmt2_220,nlme2_220,nlmb2_220 = hp.read_alm(nlm2_220_filename,hdu=[1,2,3])
        else:
            # Seed "A"
            np.random.seed(4190002645+sim2)
            nlmt2_090,nlme2_090,nlmb2_090 = hp.synalm([nltt_090_090,nlee_090_090,nlbb_090_090,nltt_090_090*0],new=True,lmax=lmax)

            # Seed "A"
            np.random.seed(4190002645+sim2)
            nltt_T2a = (nltt_090_150)**2 / nltt_090_090; nlee_T2a = (nlee_090_150)**2 / nlee_090_090; nlbb_T2a = (nlbb_090_150)**2 / nlbb_090_090
            nlmt2_T2a,nlme2_T2a,nlmb2_T2a = hp.synalm([nltt_T2a,nlee_T2a,nlbb_T2a,nltt_T2a*0],new=True,lmax=lmax)
            # Seed "B"
            np.random.seed(89052206+sim2)
            nltt_T2b = nltt_150_150 - nltt_T2a; nlee_T2b = nlee_150_150 - nlee_T2a; nlbb_T2b = nlbb_150_150 - nlbb_T2a
            nlmt2_T2b,nlme2_T2b,nlmb2_T2b = hp.synalm([nltt_T2b,nlee_T2b,nlbb_T2b,nltt_T2b*0],new=True,lmax=lmax)
            nlmt2_150 = nlmt2_T2a + nlmt2_T2b; nlme2_150 = nlme2_T2a + nlme2_T2b; nlmb2_150 = nlmb2_T2a + nlmb2_T2b

            # Seed "A"
            np.random.seed(4190002645+sim2)
            nltt_T3a = (nltt_090_220)**2 / nltt_090_090; nlee_T3a = (nlee_090_220)**2 / nlee_090_090; nlbb_T3a = (nlbb_090_220)**2 / nlbb_090_090
            nlmt2_T3a,nlme2_T3a,nlmb2_T3a = hp.synalm([nltt_T3a,nlee_T3a,nlbb_T3a,nltt_T3a*0],new=True,lmax=lmax)
            # Seed "B"
            np.random.seed(89052206+sim2)
            nltt_T3b = (nltt_150_220 - nltt_090_150*nltt_090_220/nltt_090_090)**2 / nltt_T2b
            nlee_T3b = (nlee_150_220 - nlee_090_150*nlee_090_220/nlee_090_090)**2 / nlee_T2b
            nlbb_T3b = (nlbb_150_220 - nlbb_090_150*nlbb_090_220/nlbb_090_090)**2 / nlbb_T2b
            nlmt2_T3b,nlme2_T3b,nlmb2_T3b = hp.synalm([nltt_T3b,nlee_T3b,nlbb_T3b,nltt_T3b*0],new=True,lmax=lmax)
            # Seed "C"
            np.random.seed(978540195+sim2)
            nltt_T3c = nltt_220_220 - nltt_T3a - nltt_T3b; nlee_T3c = nlee_220_220 - nlee_T3a - nlee_T3b; nlbb_T3c = nlbb_220_220 - nlbb_T3a - nlbb_T3b
            nlmt2_T3c,nlme2_T3c,nlmb2_T3c = hp.synalm([nltt_T3c,nlee_T3c,nlbb_T3c,nltt_T3c*0],new=True,lmax=lmax)
            nlmt2_220 = nlmt2_T3a + nlmt2_T3b + nlmt2_T3c; nlme2_220 = nlme2_T3a + nlme2_T3b + nlme2_T3c; nlmb2_220 = nlmb2_T3a + nlmb2_T3b + nlmb2_T3c

            Path(dir_out+f'/nlm/').mkdir(parents=True, exist_ok=True)
            hp.write_alm(nlm2_090_filename,[nlmt2_090,nlme2_090,nlmb2_090])
            hp.write_alm(nlm2_150_filename,[nlmt2_150,nlme2_150,nlmb2_150])
            hp.write_alm(nlm2_220_filename,[nlmt2_220,nlme2_220,nlmb2_220])
        tlm1_150 += nlmt1_150; tlm1_220 += nlmt1_220; tlm1_95 += nlmt1_090
        elm1_150 += nlme1_150; elm1_220 += nlme1_220; elm1_95 += nlme1_090
        blm1_150 += nlmb1_150; blm1_220 += nlmb1_220; blm1_95 += nlmb1_090
        tlm2_150 += nlmt2_150; tlm2_220 += nlmt2_220; tlm2_95 += nlmt2_090
        elm2_150 += nlme2_150; elm2_220 += nlme2_220; elm2_95 += nlme2_090
        blm2_150 += nlmb2_150; blm2_220 += nlmb2_220; blm2_95 += nlmb2_090

    if append_shortened == 'mh' or append_shortened == 'mh_unl' or append_shortened == 'crossilc_twoseds' or append_shortened == 'crossilc_twoseds_unl' or append_shortened == 'crossilc_onesed' or append_shortened == 'crossilc_onesed_unl':
        tlm1_mv = hp.almxfl(tlm1_95,w_Tmv[0][:lmax+1]) + hp.almxfl(tlm1_150,w_Tmv[1][:lmax+1]) + hp.almxfl(tlm1_220,w_Tmv[2][:lmax+1])
        if append_shortened == 'crossilc_twoseds' or append_shortened == 'crossilc_twoseds_unl' or append_shortened == 'crossilc_onesed' or append_shortened == 'crossilc_onesed_unl':
            tlm1_cibn = hp.almxfl(tlm1_95,w_cib_null_95[:lmax+1]) + hp.almxfl(tlm1_150,w_cib_null_150[:lmax+1]) + hp.almxfl(tlm1_220,w_cib_null_220[:lmax+1])
        tlm1_tszn = hp.almxfl(tlm1_95,w_tsz_null[0][:lmax+1]) + hp.almxfl(tlm1_150,w_tsz_null[1][:lmax+1]) + hp.almxfl(tlm1_220,w_tsz_null[2][:lmax+1])
        elm1 = hp.almxfl(elm1_95,w_Emv[0][:lmax+1]) + hp.almxfl(elm1_150,w_Emv[1][:lmax+1]) + hp.almxfl(elm1_220,w_Emv[2][:lmax+1])
        blm1 = hp.almxfl(blm1_95,w_Bmv[0][:lmax+1]) + hp.almxfl(blm1_150,w_Bmv[1][:lmax+1]) + hp.almxfl(blm1_220,w_Bmv[2][:lmax+1])
        tlm2_mv = hp.almxfl(tlm2_95,w_Tmv[0][:lmax+1]) + hp.almxfl(tlm2_150,w_Tmv[1][:lmax+1]) + hp.almxfl(tlm2_220,w_Tmv[2][:lmax+1])
        if append_shortened == 'crossilc_twoseds' or append_shortened == 'crossilc_twoseds_unl' or append_shortened == 'crossilc_onesed' or append_shortened == 'crossilc_onesed_unl':
            tlm2_cibn = hp.almxfl(tlm2_95,w_cib_null_95[:lmax+1]) + hp.almxfl(tlm2_150,w_cib_null_150[:lmax+1]) + hp.almxfl(tlm2_220,w_cib_null_220[:lmax+1])
        tlm2_tszn = hp.almxfl(tlm2_95,w_tsz_null[0][:lmax+1]) + hp.almxfl(tlm2_150,w_tsz_null[1][:lmax+1]) + hp.almxfl(tlm2_220,w_tsz_null[2][:lmax+1])
        elm2 = hp.almxfl(elm2_95,w_Emv[0][:lmax+1]) + hp.almxfl(elm2_150,w_Emv[1][:lmax+1]) + hp.almxfl(elm2_220,w_Emv[2][:lmax+1])
        blm2 = hp.almxfl(blm2_95,w_Bmv[0][:lmax+1]) + hp.almxfl(blm2_150,w_Bmv[1][:lmax+1]) + hp.almxfl(blm2_220,w_Bmv[2][:lmax+1])

    # Get signal + noise residuals spectra for constructing fl filters
    print('Getting signal + noise residuals spectra for filtering')
    # If lmaxT != lmaxP, we add artificial noise in TT for ell > lmaxT
    artificial_noise = np.zeros(lmax+1)
    artificial_noise[lmaxT+2:] = 1.e10
    if append[:2] == 'mh':
        totalcls_filename = dir_out+f'totalcls/totalcls_average_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_mh.npy'
    elif append[9:15] == 'onesed':
        totalcls_filename = dir_out+f'totalcls/totalcls_average_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_crossilc_onesed.npy'
    elif append[9:16] == 'twoseds':
        totalcls_filename = dir_out+f'totalcls/totalcls_average_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_crossilc_twoseds.npy'
    if os.path.isfile(totalcls_filename):
        totalcls = np.load(totalcls_filename)
        cltt1 = totalcls[:,4]; cltt2 = totalcls[:,5]; clttx = totalcls[:,6]; cltt3 = totalcls[:,0]; clee = totalcls[:,1]; clbb = totalcls[:,2]; clte = totalcls[:,3]
    elif append_shortened == 'mh' or append_shortened == 'crossilc_twoseds' or append_shortened == 'crossilc_onesed':
        print(f"Averaged totalcls file doesn't exist yet, getting the totalcls for sim {sim1}, need to average later")
        cltt_mv = hp.alm2cl(tlm1_mv,tlm1_mv) + artificial_noise
        cltt_tszn = hp.alm2cl(tlm1_tszn,tlm1_tszn) + artificial_noise
        clte = hp.alm2cl(tlm1_mv,elm1)
        clee = hp.alm2cl(elm1,elm1)
        clbb = hp.alm2cl(blm1,blm1)
        if append_shortened == 'crossilc_twoseds' or append_shortened == 'crossilc_onesed':
            cltt_cibn = hp.alm2cl(tlm1_cibn,tlm1_cibn) + artificial_noise
            clt1t2 = hp.alm2cl(tlm1_cibn,tlm1_tszn) + artificial_noise
            clt1t3 = hp.alm2cl(tlm1_cibn,tlm1_mv) + artificial_noise
            clt2t3 = hp.alm2cl(tlm1_tszn,tlm1_mv) + artificial_noise
            clt1e = hp.alm2cl(elm1,tlm1_cibn)
            clt2e = hp.alm2cl(elm1,tlm1_tszn)
            totalcls = np.vstack((cltt_mv,clee,clbb,clte,cltt_cibn,cltt_tszn,clt1t2,clt1t3,clt2t3,clt1e,clt2e)).T
        else:
            cltt_cross = hp.alm2cl(tlm1_mv,tlm2_tszn) + artificial_noise
            clte_tszn = hp.alm2cl(elm1,tlm2_tszn)
            totalcls = np.vstack((cltt_mv,clee,clbb,clte,cltt_mv,cltt_tszn,cltt_cross,cltt_mv,cltt_cross,clte,clte_tszn)).T
        np.save(dir_out+f'totalcls/totalcls_seed1_{sim1}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append_shortened}.npy',totalcls)
        return
    else:
        print('WARNING: even for CMB-only sims, we want the filters to have the noise residuals if being used for N1 calculation!')
        print("Averaged totalcls file doesn't exist, run this script with append == 'mh', append == 'crossilc_onesed' or append == 'crossilc_twoseds'")
        return

    if append_shortened[-7:] == 'cmbonly' or append_shortened[-8:] == 'tqu1tqu2' or append_shortened[-8:] == 'tqu2tqu1':
        tlm1_T2 = tlm1; tlm2_T1 = tlm2; tlm1_T3 = tlm1; tlm2_T3 = tlm2
    else:
        if append[:2] == 'mh':
            tlm1 = tlm1_mv; tlm2 = tlm2_tszn; tlm1_T2 = tlm1_tszn; tlm2_T1 = tlm2_mv; tlm1_T3 = tlm1_mv; tlm2_T3 = tlm2_mv
        elif append[9:15] == 'onesed' or append[9:16] == 'twoseds':
            tlm1 = tlm1_cibn; tlm2 = tlm2_tszn; tlm1_T2 = tlm1_tszn; tlm2_T1 = tlm2_cibn; tlm1_T3 = tlm1_mv; tlm2_T3 = tlm2_mv

    if cinv:
        print('Doing the 1/Dl for GMV...')
        invDl1 = np.zeros(lmax+1, dtype=np.complex_)
        invDl2 = np.zeros(lmax+1, dtype=np.complex_)
        invDl3 = np.zeros(lmax+1, dtype=np.complex_)
        invDl1[lmin:] = 1./(cltt1[lmin:]*clee[lmin:] - clte[lmin:]**2)
        invDl2[lmin:] = 1./(cltt2[lmin:]*clee[lmin:] - clte[lmin:]**2)
        invDl3[lmin:] = 1./(cltt3[lmin:]*clee[lmin:] - clte[lmin:]**2)
        flb = np.zeros(lmax+1); flb[lmin:] = 1./clbb[lmin:]

        if qe == 'T1T2':
            almbar1 = hp.almxfl((hp.almxfl(tlm1,clee)-hp.almxfl(elm1,clte)),invDl1)
            almbar2 = hp.almxfl((hp.almxfl(tlm2,clee)-hp.almxfl(elm2,clte)),invDl2)
        elif qe == 'T2T1':
            almbar1 = hp.almxfl((hp.almxfl(tlm1_T2,clee)-hp.almxfl(elm1,clte)),invDl2)
            almbar2 = hp.almxfl((hp.almxfl(tlm2_T1,clee)-hp.almxfl(elm2,clte)),invDl1)
        elif qe == 'T2E1':
            if T3_opt == 'noT3':
                almbar1 = hp.almxfl((hp.almxfl(tlm1_T2,clee)-hp.almxfl(elm1,clte)),invDl2)
                almbar2 = hp.almxfl((hp.almxfl(elm2,cltt1)-hp.almxfl(tlm2_T1,clte)),invDl1)
            elif T3_opt == 'withT3':
                almbar1 = hp.almxfl((hp.almxfl(tlm1_T3,clee)-hp.almxfl(elm1,clte)),invDl3)
                almbar2 = hp.almxfl((hp.almxfl(elm2,cltt3)-hp.almxfl(tlm2_T3,clte)),invDl3)
        elif qe == 'E2T1':
            if T3_opt == 'noT3':
                almbar1 = hp.almxfl((hp.almxfl(elm1,cltt2)-hp.almxfl(tlm1_T2,clte)),invDl2)
                almbar2 = hp.almxfl((hp.almxfl(tlm2_T1,clee)-hp.almxfl(elm2,clte)),invDl1)
            elif T3_opt == 'withT3':
                almbar1 = hp.almxfl((hp.almxfl(elm1,cltt3)-hp.almxfl(tlm1_T3,clte)),invDl3)
                almbar2 = hp.almxfl((hp.almxfl(tlm2_T3,clee)-hp.almxfl(elm2,clte)),invDl3)
        elif qe == 'E2E1':
            if T3_opt == 'noT3':
                almbar1 = hp.almxfl((hp.almxfl(elm1,cltt2)-hp.almxfl(tlm1_T2,clte)),invDl2)
                almbar2 = hp.almxfl((hp.almxfl(elm2,cltt1)-hp.almxfl(tlm2_T1,clte)),invDl1)
            elif T3_opt == 'withT3':
                almbar1 = hp.almxfl((hp.almxfl(elm1,cltt3)-hp.almxfl(tlm1_T3,clte)),invDl3)
                almbar2 = hp.almxfl((hp.almxfl(elm2,cltt3)-hp.almxfl(tlm2_T3,clte)),invDl3)
        else:
            if T3_opt == 'noT3':
                if qe[0] == 'T': almbar1 = hp.almxfl((hp.almxfl(tlm1,clee)-hp.almxfl(elm1,clte)),invDl1)
                elif qe[0] == 'E': almbar1 = hp.almxfl((hp.almxfl(elm1,cltt1)-hp.almxfl(tlm1,clte)),invDl1)
                elif qe[0] == 'B': almbar1 = hp.almxfl(blm1,flb)

                if qe[1] == 'T': almbar2 = hp.almxfl((hp.almxfl(tlm2,clee)-hp.almxfl(elm2,clte)),invDl2)
                elif qe[1] == 'E': almbar2 = hp.almxfl((hp.almxfl(elm2,cltt2)-hp.almxfl(tlm2,clte)),invDl2)
                elif qe[1] == 'B': almbar2 = hp.almxfl(blm2,flb)
            elif T3_opt == 'withT3':
                if qe[0] == 'T': almbar1 = hp.almxfl((hp.almxfl(tlm1_T3,clee)-hp.almxfl(elm1,clte)),invDl3)
                elif qe[0] == 'E': almbar1 = hp.almxfl((hp.almxfl(elm1,cltt3)-hp.almxfl(tlm1_T3,clte)),invDl3)
                elif qe[0] == 'B': almbar1 = hp.almxfl(blm1,flb)

                if qe[1] == 'T': almbar2 = hp.almxfl((hp.almxfl(tlm2_T3,clee)-hp.almxfl(elm2,clte)),invDl3)
                elif qe[1] == 'E': almbar2 = hp.almxfl((hp.almxfl(elm2,cltt3)-hp.almxfl(tlm2_T3,clte)),invDl3)
                elif qe[1] == 'B': almbar2 = hp.almxfl(blm2,flb)

    elif gmv:
        print('Doing the 1/Dl for GMV...')
        invDl1 = np.zeros(lmax+1, dtype=np.complex_)
        invDl2 = np.zeros(lmax+1, dtype=np.complex_)
        invDl3 = np.zeros(lmax+1, dtype=np.complex_)
        invDl1[lmin:] = 1./(cltt1[lmin:]*clee[lmin:] - clte[lmin:]**2)
        invDl2[lmin:] = 1./(cltt2[lmin:]*clee[lmin:] - clte[lmin:]**2)
        invDl3[lmin:] = 1./(cltt3[lmin:]*clee[lmin:] - clte[lmin:]**2)
        flb = np.zeros(lmax+1); flb[lmin:] = 1./clbb[lmin:]

        # Order is T1T2, T2T1, EE, TE, ET, TB, BT, EB, BE
        alm1all = np.zeros((len(tlm1_T3),9), dtype=np.complex_)
        alm2all = np.zeros((len(tlm2_T3),9), dtype=np.complex_)
        # T1T2
        alm1all[:,0] = hp.almxfl(tlm1,invDl1)
        alm2all[:,0] = hp.almxfl(tlm2,invDl2)
        # T2T1
        alm1all[:,1] = hp.almxfl(tlm1_T2,invDl2)
        alm2all[:,1] = hp.almxfl(tlm2_T1,invDl1)
        if T3_opt == 'noT3':
            # EE
            alm1all[:,2] = hp.almxfl(elm1,invDl1)
            alm2all[:,2] = hp.almxfl(elm2,invDl2)
            # TE
            alm1all[:,3] = hp.almxfl(tlm1,invDl1)
            alm2all[:,3] = hp.almxfl(elm2,invDl2)
            # ET
            alm1all[:,4] = hp.almxfl(elm1,invDl1)
            alm2all[:,4] = hp.almxfl(tlm2,invDl2)
            # TB
            alm1all[:,5] = hp.almxfl(tlm1,invDl1)
            alm2all[:,5] = hp.almxfl(blm2,flb)
            # BT
            alm1all[:,6] = hp.almxfl(blm1,flb)
            alm2all[:,6] = hp.almxfl(tlm2,invDl2)
            # EB
            alm1all[:,7] = hp.almxfl(elm1,invDl1)
            alm2all[:,7] = hp.almxfl(blm2,flb)
            # BE
            alm1all[:,8] = hp.almxfl(blm1,flb)
            alm2all[:,8] = hp.almxfl(elm2,invDl2)
        elif T3_opt == 'withT3':
            # EE
            alm1all[:,2] = hp.almxfl(elm1,invDl3)
            alm2all[:,2] = hp.almxfl(elm2,invDl3)
            # TE
            alm1all[:,3] = hp.almxfl(tlm1_T3,invDl3)
            alm2all[:,3] = hp.almxfl(elm2,invDl3)
            # ET
            alm1all[:,4] = hp.almxfl(elm1,invDl3)
            alm2all[:,4] = hp.almxfl(tlm2_T3,invDl3)
            # TB
            alm1all[:,5] = hp.almxfl(tlm1_T3,invDl3)
            alm2all[:,5] = hp.almxfl(blm2,flb)
            # BT
            alm1all[:,6] = hp.almxfl(blm1,flb)
            alm2all[:,6] = hp.almxfl(tlm2_T3,invDl3)
            # EB
            alm1all[:,7] = hp.almxfl(elm1,invDl3)
            alm2all[:,7] = hp.almxfl(blm2,flb)
            # BE
            alm1all[:,8] = hp.almxfl(blm1,flb)
            alm2all[:,8] = hp.almxfl(elm2,invDl3)

    else:
        # SQE
        # Create 1/cl filters
        flt1 = np.zeros(lmax+1); flt1[lmin:] = 1./cltt1[lmin:] # MV or CIB-null
        flt2 = np.zeros(lmax+1); flt2[lmin:] = 1./cltt2[lmin:] # tSZ-null
        flt3 = np.zeros(lmax+1); flt3[lmin:] = 1./cltt3[lmin:] # MV
        fle = np.zeros(lmax+1); fle[lmin:] = 1./clee[lmin:]
        flb = np.zeros(lmax+1); flb[lmin:] = 1./clbb[lmin:]

        if T3_opt == 'noT3':
            if qe == 'T1T2':
                almbar1 = hp.almxfl(tlm1,flt1); flm1 = flt1
                almbar2 = hp.almxfl(tlm2,flt2); flm2 = flt2
            elif qe == 'T2T1':
                almbar1 = hp.almxfl(tlm1_T2,flt2); flm1 = flt2
                almbar2 = hp.almxfl(tlm2_T1,flt1); flm2 = flt1
            elif qe == 'T2E1':
                almbar1 = hp.almxfl(tlm1_T2,flt2); flm1 = flt2
                almbar2 = hp.almxfl(elm2,fle); flm2 = fle
            elif qe == 'E2T1':
                almbar1 = hp.almxfl(elm1,fle); flm1 = fle
                almbar2 = hp.almxfl(tlm2_T1,flt1); flm2 = flt1
            elif qe == 'E2E1':
                almbar1 = hp.almxfl(elm1,fle); flm1 = fle
                almbar2 = hp.almxfl(elm2,fle); flm2 = fle
            else:
                if qe[0] == 'T': almbar1 = hp.almxfl(tlm1,flt1); flm1 = flt1
                elif qe[0] == 'E': almbar1 = hp.almxfl(elm1,fle); flm1 = fle
                elif qe[0] == 'B': almbar1 = hp.almxfl(blm1,flb); flm1 = flb

                if qe[1] == 'T': almbar2 = hp.almxfl(tlm2,flt2); flm2 = flt2
                elif qe[1] == 'E': almbar2 = hp.almxfl(elm2,fle); flm2 = fle
                elif qe[1] == 'B': almbar2 = hp.almxfl(blm2,flb); flm2 = flb
        elif T3_opt == 'withT3':
            if qe == 'T1T2':
                almbar1 = hp.almxfl(tlm1,flt1); flm1 = flt1
                almbar2 = hp.almxfl(tlm2,flt2); flm2 = flt2
            elif qe == 'T2T1':
                almbar1 = hp.almxfl(tlm1_T2,flt2); flm1 = flt2
                almbar2 = hp.almxfl(tlm2_T1,flt1); flm2 = flt1
            elif qe == 'T2E1':
                almbar1 = hp.almxfl(tlm1_T3,flt3); flm1 = flt3
                almbar2 = hp.almxfl(elm2,fle); flm2 = fle
            elif qe == 'E2T1':
                almbar1 = hp.almxfl(elm1,fle); flm1 = fle
                almbar2 = hp.almxfl(tlm2_T3,flt3); flm2 = flt3
            elif qe == 'E2E1':
                almbar1 = hp.almxfl(elm1,fle); flm1 = fle
                almbar2 = hp.almxfl(elm2,fle); flm2 = fle
            else:
                if qe[0] == 'T': almbar1 = hp.almxfl(tlm1_T3,flt3); flm1 = flt3
                elif qe[0] == 'E': almbar1 = hp.almxfl(elm1,fle); flm1 = fle
                elif qe[0] == 'B': almbar1 = hp.almxfl(blm1,flb); flm1 = flb

                if qe[1] == 'T': almbar2 = hp.almxfl(tlm2_T3,flt3); flm2 = flt3
                elif qe[1] == 'E': almbar2 = hp.almxfl(elm2,fle); flm2 = fle
                elif qe[1] == 'B': almbar2 = hp.almxfl(blm2,flb); flm2 = flb

    # Run healqest
    print('Running healqest...')
    if qe == 'T1T2' or qe == 'T2T1': qe='TT'
    if qe == 'T2E1': qe='TE'
    if qe == 'E2T1': qe='ET'
    if qe == 'E2E1': qe='EE'
    if T3_opt == 'withT3':
        withT3 = True
    else:
        withT3 = False
    if gmv and not cinv:
        q_gmv = qest.qest_gmv(config,cls)
        #TODO: withT3 not properly implemented!
        #glm,clm = q_gmv.eval(qe,alm1all,alm2all,totalcls,crossilc=True,withT3=withT3)
        glm,clm = q_gmv.eval(qe,alm1all,alm2all,totalcls,crossilc=True)
    else:
        q = qest.qest(config,cls)
        #glm,clm = q.eval(qe,almbar1,almbar2,withT3=withT3)
        glm,clm = q.eval(qe,almbar1,almbar2)
    # Save plm
    Path(dir_out).mkdir(parents=True, exist_ok=True)
    np.save(filename,glm)
    return

if __name__ == '__main__':

    main()
