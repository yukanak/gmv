#!/usr/bin/env python3
# Run like python3 get_plms_unified.py TT 100 101
import os, sys
import numpy as np
import healpy as hp
from pathlib import Path
from time import time
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import utils
import qest

qe = str(sys.argv[1])
sim1 = int(sys.argv[2])
sim2 = int(sys.argv[3])
####################################
# EDIT THIS PART!
config_file = 'mh_yuka.yaml'
config = utils.parse_yaml(config_file)
lmax = config['lmax']
nside = config['nside']
# Even if your sims here don't have added noise, you can keep the noise files defined to include the nl in the filtering
# (e.g. when you are computing noiseless sims for the N1 calculation but you want the filter to still include nl to suppress modes exactly as in the signal map)
noise_file_090_090='noise_curves/nl_fromstack_090_090.txt'
noise_file_150_150='noise_curves/nl_fromstack_150_150.txt'
noise_file_220_220='noise_curves/nl_fromstack_220_220.txt'
noise_file_090_150='noise_curves/nl_fromstack_090_150.txt'
noise_file_090_220='noise_curves/nl_fromstack_090_220.txt'
noise_file_150_220='noise_curves/nl_fromstack_150_220.txt'
fsky_corr=1

# Sims used for actual reconstruction and N0 calculation, lensed CMB + foregrounds + noise, using MV ILC map + tSZ nulled map (MH)
append = f'mh'
# Sims used for N1 calculation, these are lensed with the same phi but different CMB realizations, no foregrounds or noise
#append = 'mh_cmbonly_phi1_tqu1tqu2'
#append = 'mh_cmbonly_phi1_tqu2tqu1'
# Sims used for N0 calculation for subtracting from N1, lensed CMB + no foregrounds + no noise
#append = 'mh_cmbonly'
# Unlensed sims + no foregrounds + no noise
#append = 'mh_unl_cmbonly'
# Unlensed sims + foregrounds + noise
#append = 'mh_unl'

# Full sky single frequency foreground sims
flm_150ghz_sim1 = f'/scratch/users/yukanaka/fg/totfg_150ghz_seed{sim1}_alm_lmax{lmax}.fits'
flm_150ghz_sim2 = f'/scratch/users/yukanaka/fg/totfg_150ghz_seed{sim2}_alm_lmax{lmax}.fits'
flm_220ghz_sim1 = f'/scratch/users/yukanaka/fg/totfg_220ghz_seed{sim1}_alm_lmax{lmax}.fits'
flm_220ghz_sim2 = f'/scratch/users/yukanaka/fg/totfg_220ghz_seed{sim2}_alm_lmax{lmax}.fits'
flm_95ghz_sim1 = f'/scratch/users/yukanaka/fg/totfg_95ghz_seed{sim1}_alm_lmax{lmax}.fits'
flm_95ghz_sim2 = f'/scratch/users/yukanaka/fg/totfg_95ghz_seed{sim2}_alm_lmax{lmax}.fits'

# CMB is same at all frequencies; also full sky here
# From amscott:/sptlocal/analysis/eete+lensing_19-20/resources/sims/planck2018/inputcmb/
alm_cmb_sim1 = f'/scratch/users/yukanaka/lensing19-20/inputcmb/tqu1/len/alms/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb1_seed{sim1}_alm_lmax{lmax}.fits'
alm_cmb_sim2 = f'/scratch/users/yukanaka/lensing19-20/inputcmb/tqu1/len/alms/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb1_seed{sim2}_alm_lmax{lmax}.fits'
alm_cmb_sim1_tqu2 = f'/scratch/users/yukanaka/lensing19-20/inputcmb/tqu2/len/alms/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb2_seed{sim1}_alm_lmax{lmax}.fits'
# Unlensed alms sampled from lensed theory spectra
unl_map_sim1 = f'/scratch/users/yukanaka/full_res_maps/unl_from_lensed_cls/unl_from_lensed_cls_seed{sim1}_lmax{lmax}_nside{nside}_20230905.fits'
unl_map_sim2 = f'/scratch/users/yukanaka/full_res_maps/unl_from_lensed_cls/unl_from_lensed_cls_seed{sim2}_lmax{lmax}_nside{nside}_20230905.fits'

# ILC weights
# Dimension (3, 6001) for 90, 150, 220 GHz respectively
weights_tsz_null_T = 'ilc_weights/weights1d_TT_spt3g_cmbynull.dat'
weights_mv_ilc_T = 'ilc_weights/weights1d_TT_spt3g_cmbmv.dat'
weights_mv_ilc_E = 'ilc_weights/weights1d_EE_spt3g_cmbmv.dat'
weights_mv_ilc_B = 'ilc_weights/weights1d_BB_spt3g_cmbmv.dat'
weights_cib_null_T = 'ilc_weights/weights1d_TT_spt3g_cmbcibnull.dat'
####################################
time0 = time()

if qe == 'TTEETE' or qe == 'TBEB' or qe == 'all' or qe == 'TTEETEprf':
    gmv = True
elif qe == 'TT' or qe == 'TE' or qe == 'EE' or qe == 'TB' or qe == 'EB' or qe == 'TTprf':
    gmv = False
else:
    print('Invalid qe!')

cambini = config['cambini']
dir_out = config['dir_out']
lmax = config['lmax']
lmaxT = config['lmaxT']
lmaxP = config['lmaxP']
lmin = config['lminT']
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
    if append == 'mh' or append == 'mh_cmbonly':
        tlm1,elm1,blm1 = hp.read_alm(alm_cmb_sim1,hdu=[1,2,3])
        tlm2,elm2,blm2 = hp.read_alm(alm_cmb_sim2,hdu=[1,2,3])
    elif append == 'mh_cmbonly_phi1_tqu1tqu2':
        tlm1,elm1,blm1 = hp.read_alm(alm_cmb_sim1,hdu=[1,2,3])
        tlm2,elm2,blm2 = hp.read_alm(alm_cmb_sim1_tqu2,hdu=[1,2,3])
    elif append == 'mh_cmbonly_phi1_tqu2tqu1':
        tlm1,elm1,blm1 = hp.read_alm(alm_cmb_sim1_tqu2,hdu=[1,2,3])
        tlm2,elm2,blm2 = hp.read_alm(alm_cmb_sim1,hdu=[1,2,3])
    elif append == 'mh_unl_cmbonly' or append == 'mh_unl':
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
    if append == 'mh' or append == 'mh_unl':
        tlm1_150 = tlm1.copy(); tlm1_220 = tlm1.copy(); tlm1_95 = tlm1.copy()
        elm1_150 = elm1.copy(); elm1_220 = elm1.copy(); elm1_95 = elm1.copy()
        blm1_150 = blm1.copy(); blm1_220 = blm1.copy(); blm1_95 = blm1.copy()
        tlm2_150 = tlm2.copy(); tlm2_220 = tlm2.copy(); tlm2_95 = tlm2.copy()
        elm2_150 = elm2.copy(); elm2_220 = elm2.copy(); elm2_95 = elm2.copy()
        blm2_150 = blm2.copy(); blm2_220 = blm2.copy(); blm2_95 = blm2.copy()

    # Adding foregrounds!
    if append == 'mh' or append == 'mh_unl':
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
    if append == 'mh' or append == 'mh_unl':
        if noise_file_090_090 is not None:
            noise_curves_090_090 = np.nan_to_num(np.loadtxt(noise_file_090_090))
            noise_curves_150_150 = np.nan_to_num(np.loadtxt(noise_file_150_150))
            noise_curves_220_220 = np.nan_to_num(np.loadtxt(noise_file_220_220))
            noise_curves_090_150 = np.nan_to_num(np.loadtxt(noise_file_090_150))
            noise_curves_090_220 = np.nan_to_num(np.loadtxt(noise_file_090_220))
            noise_curves_150_220 = np.nan_to_num(np.loadtxt(noise_file_150_220))
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
        else:
            print("No noise files given!")

    if append == 'mh' or append == 'mh_unl':
        # Load weights
        w_tsz_null = np.loadtxt(weights_tsz_null_T)
        w_Tmv = np.loadtxt(weights_mv_ilc_T)
        w_Emv = np.loadtxt(weights_mv_ilc_E)
        w_Bmv = np.loadtxt(weights_mv_ilc_B)
        # MV ILC for sim 1, tSZ nulled for sim 2
        tlm1 = hp.almxfl(tlm1_95,w_Tmv[0][:lmax+1]) + hp.almxfl(tlm1_150,w_Tmv[1][:lmax+1]) + hp.almxfl(tlm1_220,w_Tmv[2][:lmax+1])
        elm1 = hp.almxfl(elm1_95,w_Emv[0][:lmax+1]) + hp.almxfl(elm1_150,w_Emv[1][:lmax+1]) + hp.almxfl(elm1_220,w_Emv[2][:lmax+1])
        blm1 = hp.almxfl(blm1_95,w_Bmv[0][:lmax+1]) + hp.almxfl(blm1_150,w_Bmv[1][:lmax+1]) + hp.almxfl(blm1_220,w_Bmv[2][:lmax+1])
        tlm2 = hp.almxfl(tlm2_95,w_tsz_null[0][:lmax+1]) + hp.almxfl(tlm2_150,w_tsz_null[1][:lmax+1]) + hp.almxfl(tlm2_220,w_tsz_null[2][:lmax+1])
        elm2 = hp.almxfl(elm2_95,w_Emv[0][:lmax+1]) + hp.almxfl(elm2_150,w_Emv[1][:lmax+1]) + hp.almxfl(elm2_220,w_Emv[2][:lmax+1])
        blm2 = hp.almxfl(blm2_95,w_Bmv[0][:lmax+1]) + hp.almxfl(blm2_150,w_Bmv[1][:lmax+1]) + hp.almxfl(blm2_220,w_Bmv[2][:lmax+1])

    # Get signal + noise residuals spectra for constructing fl filters
    print('Getting signal + noise residuals spectra for filtering')
    # If lmaxT != lmaxP, we add artificial noise in TT for ell > lmaxT
    #artificial_noise = np.zeros(lmax+1)
    #artificial_noise[lmaxT+2:] = 1.e10
    #if not (append == 'mh' or append == 'mh_unl'):
    #    print('WARNING: even for CMB only sims, we want the filters to have the noise residuals if being used for N1 calculation!')
    #cltt = hp.alm2cl(tlm1,tlm2) + artificial_noise
    #clee = hp.alm2cl(elm1,elm2)
    #clbb = hp.alm2cl(blm1,blm2)
    #clte = hp.alm2cl(tlm1,elm2)
    #totalcls = np.vstack((cltt,clee,clbb,clte)).T
    #np.save(dir_out+f'totalcls/totalcls_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy',totalcls)
    totalcls = np.load(dir_out+f'totalcls/totalcls_average_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_mh.npy')
    cltt = totalcls[:,0]; clee = totalcls[:,1]; clbb = totalcls[:,2]; clte = totalcls[:,3]
    
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
    
    # Run healqest
    print('Running healqest...')
    if not gmv:
        q_original = qest.qest(config,cls)
        glm,clm = q_original.eval(qe,almbar1,almbar2)
        # Save plm and clm
        Path(dir_out).mkdir(parents=True, exist_ok=True)
        np.save(filename_sqe,glm)
    else:
        q_gmv = qest.qest_gmv(config,cls)
        glm,clm = q_gmv.eval(qe,alm1all,alm2all,totalcls)
        # Save plm and clm
        Path(dir_out).mkdir(parents=True, exist_ok=True)
        np.save(filename_gmv,glm)

elapsed = time() - time0
elapsed /= 60
print('Time taken (minutes): ', elapsed)
