#!/usr/bin/env python3
# Run like python3 get_plms.py TT r 1 append test_yuka.yaml
# Note: argument append should be either 'standard' (used for actual reconstruction and N0 calculation, lensed CMB + Yuuki's foreground sims + noise),
# 'standard_cmbonly_phi1_tqu1tqu2', 'standard_cmbonly_phi1_tqu2tqu1' (used for N1 calculation, these are lensed with the same phi but different CMB realizations, no foregrounds or noise),
# 'standard_cmbonly' (used for N0 calculation for subtracting from N1, lensed CMB + no foregrounds + no noise),
# 'standard_unl_cmbonly' (unlensed sims + no foregrounds + no noise), or 'standard_unl' (unlensed sims + foregrounds + noise)
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
    append = str(sys.argv[4])
    config_file = str(sys.argv[5])

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
    filename_sqe = dir_out+f'/plm_{qe}_healqest_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy'
    filename_gmv = dir_out+f'/plm_{qe}_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy'

    if os.path.isfile(filename_sqe) or os.path.isfile(filename_gmv):
        print('File already exists!')
    else:
        do_reconstruction(qe,sim1,sim2,append,config_file)

    elapsed = time() - time0
    elapsed /= 60
    print('Time taken (minutes): ', elapsed)

def do_reconstruction(qe,sim1,sim2,append,config_file):
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
    filename_sqe = dir_out+f'/plm_{qe}_healqest_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy'
    filename_gmv = dir_out+f'/plm_{qe}_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy'
    if sim1 == 'r':
        sim = sim2
    elif sim2 == 'r':
        sim = sim1

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

    # Full sky single frequency Gaussian foreground sims
    flm_150ghz = f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_150ghz_seed{sim}_alm_lmax{lmax}.fits'
    flm_220ghz = f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_220ghz_seed{sim}_alm_lmax{lmax}.fits'
    flm_95ghz = f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_95ghz_seed{sim}_alm_lmax{lmax}.fits'

    # CMB is same at all frequencies; also full sky here
    # From amscott:/sptlocal/analysis/eete+lensing_19-20/resources/sims/planck2018/inputcmb/
    alm_cmb = f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/tqu1/len/alms/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb1_seed{sim}_alm_lmax{lmax}.fits'

    # Unlensed CMB alms sampled from lensed theory spectra
    unl_map = f'/oak/stanford/orgs/kipac/users/yukanaka/unl_from_lensed_cls/unl_from_lensed_cls_seed{sim}_lmax{lmax}_nside{nside}_20230905.fits'

    # ILC weights
    # Dimension (3, 6001) for 90, 150, 220 GHz respectively
    w_tsz_null = np.loadtxt('ilc_weights/weights1d_TT_spt3g_cmbynull.dat')
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
    tlm_95_agora, elm_95_agora, blm_95_agora = hp.read_alm(agora_095,hdu=[1,2,3])
    tlm_150_agora, elm_150_agora, blm_150_agora = hp.read_alm(agora_150,hdu=[1,2,3])
    tlm_220_agora, elm_220_agora, blm_220_agora = hp.read_alm(agora_220,hdu=[1,2,3])

    # Get full sky Gaussian lensed CMB alms
    print('Getting alms...')
    if append == 'standard' or append == 'standard_cmbonly':
        tlm,elm,blm = hp.read_alm(alm_cmb,hdu=[1,2,3])
    elif append == 'standard_unl_cmbonly' or append == 'standard_unl':
        t,q,u = hp.read_map(unl_map,field=[0,1,2])
        tlm,elm,blm = hp.map2alm([t,q,u],lmax=lmax)
    if append == 'standard' or append == 'standard_unl':
        tlm_150 = tlm.copy(); tlm_220 = tlm.copy(); tlm_95 = tlm.copy()
        elm_150 = elm.copy(); elm_220 = elm.copy(); elm_95 = elm.copy()
        blm_150 = blm.copy(); blm_220 = blm.copy(); blm_95 = blm.copy()

    # Adding Gaussian foregrounds to Gaussian CMB alms!
    if append == 'standard' or append == 'standard_unl':
        tflm_150, eflm_150, bflm_150 = hp.read_alm(flm_150ghz,hdu=[1,2,3])
        tflm_220, eflm_220, bflm_220 = hp.read_alm(flm_220ghz,hdu=[1,2,3])
        tflm_95, eflm_95, bflm_95 = hp.read_alm(flm_95ghz,hdu=[1,2,3])
        tlm_150 += tflm_150; tlm_220 += tflm_220; tlm_95 += tflm_95
        elm_150 += eflm_150; elm_220 += eflm_220; elm_95 += eflm_95
        blm_150 += bflm_150; blm_220 += bflm_220; blm_95 += bflm_95

    if sim1 == 'r':
        tlm1_150 = tlm_150_agora; tlm1_220 = tlm_220_agora; tlm1_95 = tlm_95_agora
        elm1_150 = elm_150_agora; elm1_220 = elm_220_agora; elm1_95 = elm_95_agora
        blm1_150 = blm_150_agora; blm1_220 = blm_220_agora; blm1_95 = blm_95_agora
        tlm2_150 = tlm_150; tlm2_220 = tlm_220; tlm2_95 = tlm_95
        elm2_150 = elm_150; elm2_220 = elm_220; elm2_95 = elm_95
        blm2_150 = blm_150; blm2_220 = blm_220; blm2_95 = blm_95
        sim1 = 999
    elif sim2 == 'r':
        tlm2_150 = tlm_150_agora; tlm2_220 = tlm_220_agora; tlm2_95 = tlm_95_agora
        elm2_150 = elm_150_agora; elm2_220 = elm_220_agora; elm2_95 = elm_95_agora
        blm2_150 = blm_150_agora; blm2_220 = blm_220_agora; blm2_95 = blm_95_agora
        tlm1_150 = tlm_150; tlm1_220 = tlm_220; tlm1_95 = tlm_95
        elm1_150 = elm_150; elm1_220 = elm_220; elm1_95 = elm_95
        blm1_150 = blm_150; blm1_220 = blm_220; blm1_95 = blm_95
        sim2 = 999

    # Adding noise!
    if append == 'standard' or append == 'standard_unl':
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

    if append == 'standard' or append == 'standard_unl':
        tlm1 = hp.almxfl(tlm1_95,w_Tmv[0][:lmax+1]) + hp.almxfl(tlm1_150,w_Tmv[1][:lmax+1]) + hp.almxfl(tlm1_220,w_Tmv[2][:lmax+1])
        elm1 = hp.almxfl(elm1_95,w_Emv[0][:lmax+1]) + hp.almxfl(elm1_150,w_Emv[1][:lmax+1]) + hp.almxfl(elm1_220,w_Emv[2][:lmax+1])
        blm1 = hp.almxfl(blm1_95,w_Bmv[0][:lmax+1]) + hp.almxfl(blm1_150,w_Bmv[1][:lmax+1]) + hp.almxfl(blm1_220,w_Bmv[2][:lmax+1])
        tlm2 = hp.almxfl(tlm2_95,w_Tmv[0][:lmax+1]) + hp.almxfl(tlm2_150,w_Tmv[1][:lmax+1]) + hp.almxfl(tlm2_220,w_Tmv[2][:lmax+1])
        elm2 = hp.almxfl(elm2_95,w_Emv[0][:lmax+1]) + hp.almxfl(elm2_150,w_Emv[1][:lmax+1]) + hp.almxfl(elm2_220,w_Emv[2][:lmax+1])
        blm2 = hp.almxfl(blm2_95,w_Bmv[0][:lmax+1]) + hp.almxfl(blm2_150,w_Bmv[1][:lmax+1]) + hp.almxfl(blm2_220,w_Bmv[2][:lmax+1])

    # Get signal + noise residuals spectra for constructing fl filters
    # Filter both with the spectra for the real data
    print('Getting signal + noise residuals spectra for filtering')
    # If lmaxT != lmaxP, we add artificial noise in TT for ell > lmaxT
    artificial_noise = np.zeros(lmax+1)
    artificial_noise[lmaxT+2:] = 1.e10
    # This file is the same as just the totalcls for agora_standard, but that one has noise sim 1 and this has noise sim 999... Shouldn't matter though?
    totalcls_filename = dir_out+f'totalcls/totalcls_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_agora_standard_rdn0.npy'
    if os.path.isfile(totalcls_filename):
        totalcls = np.load(totalcls_filename)
        cltt = totalcls[:,0]; clee = totalcls[:,1]; clbb = totalcls[:,2]; clte = totalcls[:,3]
    elif append == 'standard':
        print(f"totalcls file doesn't exist yet, getting the totalcls for the Agora sim with Gaussian noise")
        if sim1 == 999:
            cltt = hp.alm2cl(tlm1,tlm1) + artificial_noise
            clee = hp.alm2cl(elm1,elm1)
            clbb = hp.alm2cl(blm1,blm1)
            clte = hp.alm2cl(tlm1,elm1)
            totalcls = np.vstack((cltt,clee,clbb,clte)).T
            np.save(totalcls_filename,totalcls)
        elif sim2 == 999:
            cltt = hp.alm2cl(tlm2,tlm2) + artificial_noise
            clee = hp.alm2cl(elm2,elm2)
            clbb = hp.alm2cl(blm2,blm2)
            clte = hp.alm2cl(tlm2,elm2)
            totalcls = np.vstack((cltt,clee,clbb,clte)).T
            np.save(totalcls_filename,totalcls)

    if not gmv:
        print('Creating filters...')
        # Create 1/cl filters
        flt = np.zeros(lmax+1); flt[lmin:] = 1./cltt[lmin:] # MV
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

        # Order is TT, EE, TE, ET, TB, BT, EB, BE
        alm1all = np.zeros((len(tlm1),8), dtype=np.complex_)
        alm2all = np.zeros((len(tlm2),8), dtype=np.complex_)
        # TT
        alm1all[:,0] = hp.almxfl(tlm1,invDl)
        alm2all[:,0] = hp.almxfl(tlm2,invDl)
        # EE
        alm1all[:,1] = hp.almxfl(elm1,invDl)
        alm2all[:,1] = hp.almxfl(elm2,invDl)
        # TE
        alm1all[:,2] = hp.almxfl(tlm1,invDl)
        alm2all[:,2] = hp.almxfl(elm2,invDl)
        # ET
        alm1all[:,3] = hp.almxfl(elm1,invDl)
        alm2all[:,3] = hp.almxfl(tlm2,invDl)
        # TB
        alm1all[:,4] = hp.almxfl(tlm1,invDl)
        alm2all[:,4] = hp.almxfl(blm2,flb)
        # BT
        alm1all[:,5] = hp.almxfl(blm1,flb)
        alm2all[:,5] = hp.almxfl(tlm2,invDl)
        # EB
        alm1all[:,6] = hp.almxfl(elm1,invDl)
        alm2all[:,6] = hp.almxfl(blm2,flb)
        # BE
        alm1all[:,7] = hp.almxfl(blm1,flb)
        alm2all[:,7] = hp.almxfl(elm2,invDl)

    # Run healqest
    print('Running healqest...')
    if not gmv:
        q_original = qest.qest(config,cls)
        glm,clm = q_original.eval(qe,almbar1,almbar2)
        # Save plm and clm
        Path(dir_out).mkdir(parents=True, exist_ok=True)
        np.save(filename_sqe,glm)
        return
    else:
        q_gmv = qest.qest_gmv(config,cls)
        glm,clm = q_gmv.eval(qe,alm1all,alm2all,totalcls,crossilc=False)
        # Save plm and clm
        Path(dir_out).mkdir(parents=True, exist_ok=True)
        np.save(filename_gmv,glm)
        return

if __name__ == '__main__':

    main()
