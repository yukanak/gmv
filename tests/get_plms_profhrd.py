#!/usr/bin/env python3
# Run like python3 get_plms_example.py TT 100 101 append
import os, sys
import numpy as np
import healpy as hp
import pickle
from pathlib import Path
from time import time
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import healqest_utils as utils
import qest

# Note: argument append should be either 'profhrd' (used for actual reconstruction and N0 calculation, lensed CMB + tSZ in T + noise),
# 'profhrd_cmbonly_phi1_tqu1tqu2', 'profhrd_cmbonly_phi1_tqu2tqu1' (used for N1 calculation, these are lensed with the same phi but different CMB realizations, no foregrounds or noise),
# 'profhrd_cmbonly' (used for N0 calculation for subtracting from N1, lensed CMB + no foregrounds + no noise),
# 'profhrd_unl_cmbonly' (unlensed sims + no foregrounds + no noise), or 'profhrd_unl' (unlensed sims + foregrounds + noise)

qe = str(sys.argv[1])
sim1 = int(sys.argv[2])
sim2 = int(sys.argv[3])
append = str(sys.argv[4])

####################################
config_file = 'test_yuka.yaml'
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

# Noise curves
fsky_corr = 1
noise_curves_090_090 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_090_090.txt'))
noise_curves_150_150 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_150_150.txt'))
noise_curves_220_220 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_220_220.txt'))
noise_curves_090_150 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_090_150.txt'))
noise_curves_090_220 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_090_220.txt'))
noise_curves_150_220 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_150_220.txt'))

# Foreground curves
fg_curves = pickle.load(open('agora_tsz_spec.pk','rb'))
tsz_curve_095_095 = fg_curves['masked']['95x95']
tsz_curve_150_150 = fg_curves['masked']['150x150']
tsz_curve_220_220 = fg_curves['masked']['220x220']
tsz_curve_095_150 = fg_curves['masked']['95x150']
tsz_curve_095_220 = fg_curves['masked']['95x220']
tsz_curve_150_220 = fg_curves['masked']['150x220']

# Full sky CMB signal maps; same at all frequencies
# From amscott:/sptlocal/analysis/eete+lensing_19-20/resources/sims/planck2018/inputcmb/
alm_cmb_sim1 = f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/tqu1/len/alms/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb1_seed{sim1}_alm_lmax{lmax}.fits'
alm_cmb_sim2 = f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/tqu1/len/alms/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb1_seed{sim2}_alm_lmax{lmax}.fits'
alm_cmb_sim1_tqu2 = f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/tqu2/len/alms/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb2_seed{sim1}_alm_lmax{lmax}.fits'

# Unlensed CMB alms sampled from lensed theory spectra
unl_map_sim1 = f'/oak/stanford/orgs/kipac/users/yukanaka/unl_from_lensed_cls/unl_from_lensed_cls_seed{sim1}_lmax{lmax}_nside{nside}_20230905.fits'
unl_map_sim2 = f'/oak/stanford/orgs/kipac/users/yukanaka/unl_from_lensed_cls/unl_from_lensed_cls_seed{sim2}_lmax{lmax}_nside{nside}_20230905.fits'

# ILC weights
# Dimension (3, 6001) for 90, 150, 220 GHz respectively
w_tsz_null = np.loadtxt('ilc_weights/weights1d_TT_spt3g_cmbynull.dat')
w_Tmv = np.loadtxt('ilc_weights/weights1d_TT_spt3g_cmbmv.dat')
w_Emv = np.loadtxt('ilc_weights/weights1d_EE_spt3g_cmbmv.dat')
w_Bmv = np.loadtxt('ilc_weights/weights1d_BB_spt3g_cmbmv.dat')

# Profile
profile_filename = 'TT_mvilc_foreground_residuals.pkl'
if os.path.isfile(profile_filename):
    u = pickle.load(open(profile_filename,'rb'))
else:
    # Combine Agora TT cross frequency tSZ spectra with MV ILC weights to get ILC-ed foreground residuals
    ret = np.zeros((lmax+1))
    b='tt'; c=1; w1=w_Tmv; w2=w_Tmv
    for ll in l:
        # At each ell, get 3x3 matrix with each block containing Cl for different freq combinations
        clmat = np.zeros((3,3))
        clmat[0,0] = tsz_curve_095_095[ll]
        clmat[1,1] = tsz_curve_150_150[ll]
        clmat[2,2] = tsz_curve_220_220[ll]
        clmat[0,1] = clmat[1,0] = tsz_curve_095_150[ll]
        clmat[0,2] = clmat[2,0] = tsz_curve_095_220[ll]
        clmat[1,2] = clmat[2,1] = tsz_curve_150_220[ll]
        ret[ll] = np.dot(w1[:,ll], np.dot(clmat, w2[:,ll].T))
    # Use the TT ILC-ed foreground residuals as the profile
    u = ret
    with open(profile_filename,'wb') as f:
        pickle.dump(u,f)
####################################

time0 = time()

if qe == 'TTEETE' or qe == 'TBEB' or qe == 'all' or qe == 'TTEETEprf':
    gmv = True
elif qe == 'TT' or qe == 'TE' or  qe == 'ET' or qe == 'EE' or qe == 'TB' or  qe == 'BT' or qe == 'EB' or  qe == 'BE' or qe == 'TTprf' or qe == 'T1T2' or qe == 'T2T1':
    gmv = False
else:
    print('Invalid qe!')

if os.path.isfile(filename_sqe) or os.path.isfile(filename_gmv):
    print('File already exists!')
else:
    print(f'Doing reconstruction for sims {sim1} and {sim2}, qe {qe}, append {append}')

    # Get full sky CMB alms
    print('Getting alms...')
    if append == f'profhrd' or append == 'profhrd_cmbonly':
        tlm1,elm1,blm1 = hp.read_alm(alm_cmb_sim1,hdu=[1,2,3])
        tlm2,elm2,blm2 = hp.read_alm(alm_cmb_sim2,hdu=[1,2,3])
    elif append == 'profhrd_cmbonly_phi1_tqu1tqu2':
        tlm1,elm1,blm1 = hp.read_alm(alm_cmb_sim1,hdu=[1,2,3])
        tlm2,elm2,blm2 = hp.read_alm(alm_cmb_sim1_tqu2,hdu=[1,2,3])
    elif append == 'profhrd_cmbonly_phi1_tqu2tqu1':
        tlm1,elm1,blm1 = hp.read_alm(alm_cmb_sim1_tqu2,hdu=[1,2,3])
        tlm2,elm2,blm2 = hp.read_alm(alm_cmb_sim1,hdu=[1,2,3])
    elif append == 'profhrd_unl' or append == 'profhrd_unl_cmbonly':
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
    if append == 'profhrd' or append == 'profhrd_unl':
        tlm1_150 = tlm1.copy(); tlm1_220 = tlm1.copy(); tlm1_95 = tlm1.copy()
        elm1_150 = elm1.copy(); elm1_220 = elm1.copy(); elm1_95 = elm1.copy()
        blm1_150 = blm1.copy(); blm1_220 = blm1.copy(); blm1_95 = blm1.copy()
        tlm2_150 = tlm2.copy(); tlm2_220 = tlm2.copy(); tlm2_95 = tlm2.copy()
        elm2_150 = elm2.copy(); elm2_220 = elm2.copy(); elm2_95 = elm2.copy()
        blm2_150 = blm2.copy(); blm2_220 = blm2.copy(); blm2_95 = blm2.copy()

    # Adding foregrounds!
    if append == f'profhrd' or append == 'profhrd_unl':
        flmt1_095_filename = dir_out + f'flm/flmt_095_lmax{lmax}_seed{sim1}.alm'
        flmt1_150_filename = dir_out + f'flm/flmt_150_lmax{lmax}_seed{sim1}.alm'
        flmt1_220_filename = dir_out + f'flm/flmt_220_lmax{lmax}_seed{sim1}.alm'
        flmt2_095_filename = dir_out + f'flm/flmt_095_lmax{lmax}_seed{sim2}.alm'
        flmt2_150_filename = dir_out + f'flm/flmt_150_lmax{lmax}_seed{sim2}.alm'
        flmt2_220_filename = dir_out + f'flm/flmt_220_lmax{lmax}_seed{sim2}.alm'

        if os.path.isfile(flmt1_095_filename):
            flmt1_095 = hp.read_alm(flmt1_095_filename,hdu=[1,2,3])
            flmt1_150 = hp.read_alm(flmt1_150_filename,hdu=[1,2,3])
            flmt1_220 = hp.read_alm(flmt1_220_filename,hdu=[1,2,3])
        else:
            # See appendix of https://arxiv.org/pdf/0801.4380.pdf
            # Need to generate frequency correlated realizations
            # Seed "A"
            np.random.seed(3241998+sim1)
            flmt1_095 = hp.synalm(tsz_curve_095_095,lmax=lmax)

            # Seed "A"
            # Quick note, the hash part returns a different value for different python processes
            np.random.seed(3241998+sim1)
            fltt_T2a = np.nan_to_num((tsz_curve_095_150)**2 / tsz_curve_095_095)
            flmt1_T2a = hp.synalm(fltt_T2a,lmax=lmax)
            # Seed "B"
            np.random.seed(4102002+sim1)
            fltt_T2b = tsz_curve_150_150 - fltt_T2a
            flmt1_T2b = hp.synalm(fltt_T2b,lmax=lmax)
            flmt1_150 = flmt1_T2a + flmt1_T2b

            # Seed "A"
            np.random.seed(3241998+sim1)
            fltt_T3a = np.nan_to_num((tsz_curve_095_220)**2 / tsz_curve_095_095)
            flmt1_T3a = hp.synalm(fltt_T3a,lmax=lmax)
            # Seed "B"
            np.random.seed(4102002+sim1)
            fltt_T3b = np.nan_to_num((tsz_curve_150_220 - tsz_curve_095_150*tsz_curve_095_220/tsz_curve_095_095)**2 / fltt_T2b)
            flmt1_T3b = hp.synalm(fltt_T3b,lmax=lmax)
            # Seed "C"
            np.random.seed(9011958+sim1)
            fltt_T3c = tsz_curve_220_220 - fltt_T3a - fltt_T3b
            flmt1_T3c = hp.synalm(fltt_T3c,lmax=lmax)
            flmt1_220 = flmt1_T3a + flmt1_T3b + flmt1_T3c

            Path(dir_out+f'/flm/').mkdir(parents=True, exist_ok=True)
            hp.write_alm(flmt1_095_filename,flmt1_095)
            hp.write_alm(flmt1_150_filename,flmt1_150)
            hp.write_alm(flmt1_220_filename,flmt1_220)

        if os.path.isfile(flmt2_095_filename):
            flmt2_095 = hp.read_alm(flmt2_095_filename,hdu=[1,2,3])
            flmt2_150 = hp.read_alm(flmt2_150_filename,hdu=[1,2,3])
            flmt2_220 = hp.read_alm(flmt2_220_filename,hdu=[1,2,3])
        else:
            # Seed "A"
            np.random.seed(3241998+sim2)
            flmt2_095 = hp.synalm(tsz_curve_095_095,lmax=lmax)

            # Seed "A"
            np.random.seed(3241998+sim2)
            fltt_T2a = np.nan_to_num((tsz_curve_095_150)**2 / tsz_curve_095_095)
            flmt2_T2a = hp.synalm(fltt_T2a,lmax=lmax)
            # Seed "B"
            np.random.seed(4102002+sim2)
            fltt_T2b = tsz_curve_150_150 - fltt_T2a
            flmt2_T2b = hp.synalm(fltt_T2b,lmax=lmax)
            flmt2_150 = flmt2_T2a + flmt2_T2b

            # Seed "A"
            np.random.seed(3241998+sim2)
            fltt_T3a = np.nan_to_num((tsz_curve_095_220)**2 / tsz_curve_095_095)
            flmt2_T3a = hp.synalm(fltt_T3a,lmax=lmax)
            # Seed "B"
            np.random.seed(4102002+sim2)
            fltt_T3b = np.nan_to_num((tsz_curve_150_220 - tsz_curve_095_150*tsz_curve_095_220/tsz_curve_095_095)**2 / fltt_T2b)
            flmt2_T3b = hp.synalm(fltt_T3b,lmax=lmax)
            # Seed "C"
            np.random.seed(9011958+sim2)
            fltt_T3c = tsz_curve_220_220 - fltt_T3a - fltt_T3b
            flmt2_T3c = hp.synalm(fltt_T3c,lmax=lmax)
            flmt2_220 = flmt2_T3a + flmt2_T3b + flmt2_T3c

            Path(dir_out+f'/flm/').mkdir(parents=True, exist_ok=True)
            hp.write_alm(flmt2_095_filename,flmt2_095)
            hp.write_alm(flmt2_150_filename,flmt2_150)
            hp.write_alm(flmt2_220_filename,flmt2_220)

        tlm1_150 += flmt1_150; tlm1_220 += flmt1_220; tlm1_95 += flmt1_095
        tlm2_150 += flmt2_150; tlm2_220 += flmt2_220; tlm2_95 += flmt2_095

    # Adding noise!
    if append == 'profhrd' or append == 'profhrd_unl':
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

    if append == 'profhrd' or append == 'profhrd_unl':
        tlm1 = hp.almxfl(tlm1_95,w_Tmv[0][:lmax+1]) + hp.almxfl(tlm1_150,w_Tmv[1][:lmax+1]) + hp.almxfl(tlm1_220,w_Tmv[2][:lmax+1])
        elm1 = hp.almxfl(elm1_95,w_Emv[0][:lmax+1]) + hp.almxfl(elm1_150,w_Emv[1][:lmax+1]) + hp.almxfl(elm1_220,w_Emv[2][:lmax+1])
        blm1 = hp.almxfl(blm1_95,w_Bmv[0][:lmax+1]) + hp.almxfl(blm1_150,w_Bmv[1][:lmax+1]) + hp.almxfl(blm1_220,w_Bmv[2][:lmax+1])
        tlm2 = hp.almxfl(tlm2_95,w_Tmv[0][:lmax+1]) + hp.almxfl(tlm2_150,w_Tmv[1][:lmax+1]) + hp.almxfl(tlm2_220,w_Tmv[2][:lmax+1])
        elm2 = hp.almxfl(elm2_95,w_Emv[0][:lmax+1]) + hp.almxfl(elm2_150,w_Emv[1][:lmax+1]) + hp.almxfl(elm2_220,w_Emv[2][:lmax+1])
        blm2 = hp.almxfl(blm2_95,w_Bmv[0][:lmax+1]) + hp.almxfl(blm2_150,w_Bmv[1][:lmax+1]) + hp.almxfl(blm2_220,w_Bmv[2][:lmax+1])

    # Get signal + noise spectra for constructing fl filters
    print('Getting signal + noise residuals spectra for filtering')
    # If lmaxT != lmaxP, we add artificial noise in TT for ell > lmaxT
    artificial_noise = np.zeros(lmax+1)
    artificial_noise[lmaxT+2:] = 1.e10
    totalcls_filename = dir_out+f'totalcls/totalcls_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_profhrd.npy'
    if os.path.isfile(totalcls_filename):
        totalcls = np.load(totalcls_filename)
        cltt = totalcls[:,0]; clee = totalcls[:,1]; clbb = totalcls[:,2]; clte = totalcls[:,3]
    else:
        # Combine cross frequency spectra with ILC weights
        # Second dimension order TT, EE, BB
        ret = np.zeros((lmax+1,3))
        for a in range(3):
            if a == 0: b='tt'; c=1; w1=w_Tmv; w2=w_Tmv
            if a == 1: b='ee'; c=2; w1=w_Emv; w2=w_Emv
            if a == 2: b='bb'; c=3; w1=w_Bmv; w2=w_Bmv
            for ll in l:
                # At each ell, have 3x3 matrix with each block containing Cl for different frequency combinations
                clmat = np.zeros((3,3))
                if a == 0:
                    clmat[0,0] = sl[b][ll] + noise_curves_090_090[ll,c] + tsz_curve_095_095[ll]
                    clmat[1,1] = sl[b][ll] + noise_curves_150_150[ll,c] + tsz_curve_150_150[ll]
                    clmat[2,2] = sl[b][ll] + noise_curves_220_220[ll,c] + tsz_curve_220_220[ll]
                    clmat[0,1] = clmat[1,0] = sl[b][ll] + noise_curves_090_150[ll,c] + tsz_curve_095_150[ll]
                    clmat[0,2] = clmat[2,0] = sl[b][ll] + noise_curves_090_220[ll,c] + tsz_curve_095_220[ll]
                    clmat[1,2] = clmat[2,1] = sl[b][ll] + noise_curves_150_220[ll,c] + tsz_curve_150_220[ll]
                else:
                    clmat[0,0] = sl[b][ll] + noise_curves_090_090[ll,c]
                    clmat[1,1] = sl[b][ll] + noise_curves_150_150[ll,c]
                    clmat[2,2] = sl[b][ll] + noise_curves_220_220[ll,c]
                    clmat[0,1] = clmat[1,0] = sl[b][ll] + noise_curves_090_150[ll,c]
                    clmat[0,2] = clmat[2,0] = sl[b][ll] + noise_curves_090_220[ll,c]
                    clmat[1,2] = clmat[2,1] = sl[b][ll] + noise_curves_150_220[ll,c]
                ret[ll,a]=np.dot(w1[:,ll], np.dot(clmat, w2[:,ll].T))
        cltt = ret[:,0] + artificial_noise; clee = ret[:,1]; clbb = ret[:,2]; clte = sl['te'][:lmax+1]
        totalcls = np.vstack((cltt,clee,clbb,clte)).T
        np.save(totalcls_filename,totalcls)

    if not gmv:
        print('Creating filters...')
        # Even when you are computing noiseless sims for the N1 calculation, you want the filter to still include residuals to suppress modes exactly as in the signal map
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
        glm,clm = q_original.eval(qe,almbar1,almbar2,u=u)
        # Save plm and clm
        Path(dir_out).mkdir(parents=True, exist_ok=True)
        np.save(filename_sqe,glm)
    else:
        q_gmv = qest.qest_gmv(config,cls)
        glm,clm = q_gmv.eval(qe,alm1all,alm2all,totalcls,u=u,crossilc=False)
        # Save plm and clm
        Path(dir_out).mkdir(parents=True, exist_ok=True)
        np.save(filename_gmv,glm)

elapsed = time() - time0
elapsed /= 60
print('Time taken (minutes): ', elapsed)
