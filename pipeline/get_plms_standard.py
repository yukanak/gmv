#!/usr/bin/env python3
# Run like python3 get_plms.py TT 100 101 append test_yuka.yaml agora
# Note: argument append should be either 'standard' (used for actual reconstruction and N0 calculation, lensed CMB + Yuuki's foreground sims + noise),
# 'standard_cmbonly_phi1_tqu1tqu2', 'standard_cmbonly_phi1_tqu2tqu1' (used for N1 calculation, these are lensed with the same phi but different CMB realizations, no foregrounds or noise),
# 'standard_cmbonly' (used for N0 calculation for subtracting from N1, lensed CMB + no foregrounds + no noise),
# For cinv-style, append should be 'standard_cinv', 'standard_cmbonly_phi1_tqu1tqu2_cinv', etc.
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
    fg_model = str(sys.argv[6])

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

    if append[-4:] != 'cinv' and (qe == 'TT' or qe == 'TE' or  qe == 'ET' or qe == 'EE' or qe == 'TB' or  qe == 'BT' or qe == 'EB' or  qe == 'BE' or qe == 'TTprf' or qe == 'T1T2' or qe == 'T2T1'):
        # SQE
        gmv = False
        filename = dir_out+f'/plm_{qe}_healqest_sqe_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy'
    else:
        # GMV
        gmv = True
        filename = dir_out+f'/plm_{qe}_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy'

    if append[-4:] == 'cinv' and gmv:
        cinv = True
    else:
        cinv = False

    print(f'cinv is {cinv}, gmv is {gmv}, filename {filename}')

    if os.path.isfile(filename):
        print('File already exists!')
    else:
        do_reconstruction(qe,sim1,sim2,append,config_file,filename,gmv,cinv,fg_model)

    elapsed = time() - time0
    elapsed /= 60
    print('Time taken (minutes): ', elapsed)

def do_reconstruction(qe,sim1,sim2,append,config_file,filename,gmv,cinv,fg_model):
    '''
    Function to do the actual reconstruction.
    '''
    print(f'Doing reconstruction for sims {sim1} and {sim2}, qe {qe}, append {append}, fg_model {fg_model}')

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

    # CMB is same at all frequencies; full sky
    # From amscott:/sptlocal/analysis/eete+lensing_19-20/resources/sims/planck2018/inputcmb/
    if sim1 == 'r':
        alm_cmb_sim2 = f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/tqu1/len/alms/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb1_seed{sim2}_alm_lmax{lmax}.fits'
    elif sim2 == 'r':
        alm_cmb_sim1 = f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/tqu1/len/alms/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb1_seed{sim1}_alm_lmax{lmax}.fits'
    else:
        alm_cmb_sim1 = f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/tqu1/len/alms/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb1_seed{sim1}_alm_lmax{lmax}.fits'
        alm_cmb_sim2 = f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/tqu1/len/alms/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb1_seed{sim2}_alm_lmax{lmax}.fits'
        alm_cmb_sim1_tqu2 = f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/tqu2/len/alms/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb2_seed{sim1}_alm_lmax{lmax}.fits'

    # Agora sims
    agora_095 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_95ghz_alm_lmax4096.fits'
    agora_150 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_150ghz_alm_lmax4096.fits'
    agora_220 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_220ghz_alm_lmax4096.fits'

    # Websky sims, these are maps
    websky_095 = '/oak/stanford/orgs/kipac/users/yukanaka/websky/websky_95ghz_lcmb_tsz_cib_ksz_ksz-patchy_muK_nside4096.fits'
    websky_150 = '/oak/stanford/orgs/kipac/users/yukanaka/websky/websky_150ghz_lcmb_tsz_cib_ksz_ksz-patchy_muK_nside4096.fits'
    websky_220 = '/oak/stanford/orgs/kipac/users/yukanaka/websky/websky_220ghz_lcmb_tsz_cib_ksz_ksz-patchy_muK_nside4096.fits'
    # From https://lambda.gsfc.nasa.gov/simulation/mocks_data.html but they claim it's "T,Q,U alms", but I think they mean T,E,B...
    websky_nofg = '/oak/stanford/orgs/kipac/users/yukanaka/websky/lensed_alm.fits'

    # Noise curves
    fsky_corr=1
    noise_curves_090_090 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_090_090.txt'))
    noise_curves_150_150 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_150_150.txt'))
    noise_curves_220_220 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_220_220.txt'))
    noise_curves_090_150 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_090_150.txt'))
    noise_curves_090_220 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_090_220.txt'))
    noise_curves_150_220 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_150_220.txt'))

    # Full sky foreground + noise frequency combined sims
    if fg_model == 'agora':
        if sim1 == 'r':
            fnlm_sim2 = f'/oak/stanford/orgs/kipac/users/yukanaka/gmv/inputs/spt3g_2019_2020_agora/agora_fg_plus_spt3g_20192020_noise_lmax{lmax}_seed{sim2}_mv.fits'
        elif sim2 == 'r':
            fnlm_sim1 = f'/oak/stanford/orgs/kipac/users/yukanaka/gmv/inputs/spt3g_2019_2020_agora/agora_fg_plus_spt3g_20192020_noise_lmax{lmax}_seed{sim1}_mv.fits'
        else:
            fnlm_sim1 = f'/oak/stanford/orgs/kipac/users/yukanaka/gmv/inputs/spt3g_2019_2020_agora/agora_fg_plus_spt3g_20192020_noise_lmax{lmax}_seed{sim1}_mv.fits'
            fnlm_sim2 = f'/oak/stanford/orgs/kipac/users/yukanaka/gmv/inputs/spt3g_2019_2020_agora/agora_fg_plus_spt3g_20192020_noise_lmax{lmax}_seed{sim2}_mv.fits'
    elif fg_model == 'websky':
        if sim1 == 'r':
            fnlm_sim2 = f'/oak/stanford/orgs/kipac/users/yukanaka/gmv/inputs/spt3g_2019_2020_websky/websky_fg_plus_spt3g_20192020_noise_lmax{lmax}_seed{sim2}_mv.fits'
        elif sim2 == 'r':
            fnlm_sim1 = f'/oak/stanford/orgs/kipac/users/yukanaka/gmv/inputs/spt3g_2019_2020_websky/websky_fg_plus_spt3g_20192020_noise_lmax{lmax}_seed{sim1}_mv.fits'
        else:
            fnlm_sim1 = f'/oak/stanford/orgs/kipac/users/yukanaka/gmv/inputs/spt3g_2019_2020_websky/websky_fg_plus_spt3g_20192020_noise_lmax{lmax}_seed{sim1}_mv.fits'
            fnlm_sim2 = f'/oak/stanford/orgs/kipac/users/yukanaka/gmv/inputs/spt3g_2019_2020_websky/websky_fg_plus_spt3g_20192020_noise_lmax{lmax}_seed{sim2}_mv.fits'

    # ILC weights
    if fg_model == 'agora':
        w_agora = np.load('/home/users/yukanaka/gmv/pipeline/ilc_weights/ilc_weights_cmb_spt3g_2yr.npy',allow_pickle=True).item()
        # Dimension (3, 4097) for 90, 150, 220 GHz respectively
        w_Tmv = w_agora['tt']['mv'];
        w_Emv = w_agora['ee']['mv'];
        w_Bmv = w_agora['bb']['mv'];
        w_tsz_null = w_agora['tt']['tsznull']
        w_cib_null_onesed = w_agora['tt']['cibnull']
        #w_tsz_null = np.loadtxt('ilc_weights/weights1d_TT_spt3g_cmbynull.dat')
        #w_Tmv = np.loadtxt('ilc_weights/weights1d_TT_spt3g_cmbmv.dat')
        #w_Emv = np.loadtxt('ilc_weights/weights1d_EE_spt3g_cmbmv.dat')
        #w_Bmv = np.loadtxt('ilc_weights/weights1d_BB_spt3g_cmbmv.dat')
    elif fg_model == 'websky':
        w = np.load('/home/users/yukanaka/gmv/pipeline/ilc_weights/ilc_weights_cmb_spt3g_2yr_websky.npy',allow_pickle=True).item()
        cases = ['mv', 'tsznull', 'cibnull']
        # Dimension (3, 4097) for 90, 150, 220 GHz respectively
        w_Tmv = w['tt']['mv'];
        w_Emv = w['ee']['mv'];
        w_Bmv = w['bb']['mv'];
        w_tsz_null = w['tt']['tsznull']
        w_cib_null = w['tt']['cibnull']

    print('Getting alms...')
    if fg_model == 'agora':
        # Get Agora sim (signal + foregrounds)
        tlm_95_agora, elm_95_agora, blm_95_agora = hp.read_alm(agora_095,hdu=[1,2,3])
        tlm_150_agora, elm_150_agora, blm_150_agora = hp.read_alm(agora_150,hdu=[1,2,3])
        tlm_220_agora, elm_220_agora, blm_220_agora = hp.read_alm(agora_220,hdu=[1,2,3])
    elif fg_model == 'websky':
        t_95_websky, q_95_websky, u_95_websky = hp.read_map(websky_095,field=[1,2,3])
        t_150_websky, q_150_websky, u_150_websky = hp.read_map(websky_150,field=[1,2,3])
        t_220_websky, q_220_websky, u_220_websky = hp.read_map(websky_220,field=[1,2,3])
        tlm_95_websky, elm_95_websky, blm_95_websky = hp.map2alm([t_95_websky, q_95_websky, u_95_websky], lmax=lmax)
        tlm_150_websky, elm_150_websky, blm_150_websky = hp.map2alm([t_150_websky, q_150_websky, u_150_websky], lmax=lmax)
        tlm_220_websky, elm_220_websky, blm_220_websky = hp.map2alm([t_220_websky, q_220_websky, u_220_websky], lmax=lmax)

    # Get full sky CMB alms
    print('Getting alms...')
    if append == 'standard' or append == 'standard_cmbonly' or append == 'standard_cinv' or append == 'standard_cmbonly_cinv':
        if sim1 == 'r':
            tlm2,elm2,blm2 = hp.read_alm(alm_cmb_sim2,hdu=[1,2,3])
        elif sim2 == 'r':
            tlm1,elm1,blm1 = hp.read_alm(alm_cmb_sim1,hdu=[1,2,3])
        else:
            tlm1,elm1,blm1 = hp.read_alm(alm_cmb_sim1,hdu=[1,2,3])
            tlm2,elm2,blm2 = hp.read_alm(alm_cmb_sim2,hdu=[1,2,3])
    elif append == 'standard_cmbonly_phi1_tqu1tqu2' or append == 'standard_cmbonly_phi1_tqu1tqu2_cinv':
        tlm1,elm1,blm1 = hp.read_alm(alm_cmb_sim1,hdu=[1,2,3])
        tlm2,elm2,blm2 = hp.read_alm(alm_cmb_sim1_tqu2,hdu=[1,2,3])
    elif append == 'standard_cmbonly_phi1_tqu2tqu1' or  append == 'standard_cmbonly_phi1_tqu2tqu1_cinv':
        tlm1,elm1,blm1 = hp.read_alm(alm_cmb_sim1_tqu2,hdu=[1,2,3])
        tlm2,elm2,blm2 = hp.read_alm(alm_cmb_sim1,hdu=[1,2,3])

    # Adding foregrounds and noise!
    if append == 'standard' or append == 'standard_cinv':
        if sim1 == 'r':
            if fg_model == 'agora':
                tlm1_150 = tlm_150_agora; tlm1_220 = tlm_220_agora; tlm1_95 = tlm_95_agora
                elm1_150 = elm_150_agora; elm1_220 = elm_220_agora; elm1_95 = elm_95_agora
                blm1_150 = blm_150_agora; blm1_220 = blm_220_agora; blm1_95 = blm_95_agora
            elif fg_model == 'websky':
                tlm1_150 = tlm_150_websky; tlm1_220 = tlm_220_websky; tlm1_95 = tlm_95_websky
                elm1_150 = elm_150_websky; elm1_220 = elm_220_websky; elm1_95 = elm_95_websky
                blm1_150 = blm_150_websky; blm1_220 = blm_220_websky; blm1_95 = blm_95_websky

            tfnlm2, efnlm2, bfnlm2 = hp.read_alm(fnlm_sim2,hdu=[1,2,3])
            tfnlm2 = utils.reduce_lmax(tfnlm2,lmax=lmax); efnlm2 = utils.reduce_lmax(efnlm2,lmax=lmax); bfnlm2 = utils.reduce_lmax(bfnlm2,lmax=lmax)
            tlm2 += tfnlm2
            elm2 += efnlm2
            blm2 += bfnlm2

            sim1 = 999
        elif sim2 == 'r':
            tfnlm1, efnlm1, bfnlm1 = hp.read_alm(fnlm_sim1,hdu=[1,2,3])
            tfnlm1 = utils.reduce_lmax(tfnlm1,lmax=lmax); efnlm1 = utils.reduce_lmax(efnlm1,lmax=lmax); bfnlm1 = utils.reduce_lmax(bfnlm1,lmax=lmax)
            tlm1 += tfnlm1
            elm1 += efnlm1
            blm1 += bfnlm1

            if fg_model == 'agora':
                tlm2_150 = tlm_150_agora; tlm2_220 = tlm_220_agora; tlm2_95 = tlm_95_agora
                elm2_150 = elm_150_agora; elm2_220 = elm_220_agora; elm2_95 = elm_95_agora
                blm2_150 = blm_150_agora; blm2_220 = blm_220_agora; blm2_95 = blm_95_agora
            elif fg_model == 'websky':
                tlm2_150 = tlm_150_websky; tlm2_220 = tlm_220_websky; tlm2_95 = tlm_95_websky
                elm2_150 = elm_150_websky; elm2_220 = elm_220_websky; elm2_95 = elm_95_websky
                blm2_150 = blm_150_websky; blm2_220 = blm_220_websky; blm2_95 = blm_95_websky

            sim2 = 999
        else:
            tfnlm1, efnlm1, bfnlm1 = hp.read_alm(fnlm_sim1,hdu=[1,2,3])
            tfnlm1 = utils.reduce_lmax(tfnlm1,lmax=lmax); efnlm1 = utils.reduce_lmax(efnlm1,lmax=lmax); bfnlm1 = utils.reduce_lmax(bfnlm1,lmax=lmax)
            tlm1 += tfnlm1
            elm1 += efnlm1
            blm1 += bfnlm1

            tfnlm2, efnlm2, bfnlm2 = hp.read_alm(fnlm_sim2,hdu=[1,2,3])
            tfnlm2 = utils.reduce_lmax(tfnlm2,lmax=lmax); efnlm2 = utils.reduce_lmax(efnlm2,lmax=lmax); bfnlm2 = utils.reduce_lmax(bfnlm2,lmax=lmax)
            tlm2 += tfnlm2
            elm2 += efnlm2
            blm2 += bfnlm2

    # Adding noise (ONLY to sim r)!
    if sim1 == 999 and (append == 'standard' or append == 'standard_cinv'):
        nlm1_090_filename = dir_out + f'nlm/nlm_090_lmax{lmax}_seed{sim1}.alm'
        nlm1_150_filename = dir_out + f'nlm/nlm_150_lmax{lmax}_seed{sim1}.alm'
        nlm1_220_filename = dir_out + f'nlm/nlm_220_lmax{lmax}_seed{sim1}.alm'
        nlmt1_090,nlme1_090,nlmb1_090 = hp.read_alm(nlm1_090_filename,hdu=[1,2,3])
        nlmt1_150,nlme1_150,nlmb1_150 = hp.read_alm(nlm1_150_filename,hdu=[1,2,3])
        nlmt1_220,nlme1_220,nlmb1_220 = hp.read_alm(nlm1_220_filename,hdu=[1,2,3])
        tlm1_150 += nlmt1_150; tlm1_220 += nlmt1_220; tlm1_95 += nlmt1_090
        elm1_150 += nlme1_150; elm1_220 += nlme1_220; elm1_95 += nlme1_090
        blm1_150 += nlmb1_150; blm1_220 += nlmb1_220; blm1_95 += nlmb1_090
    elif sim2 == 999 and (append == 'standard' or append == 'standard_cinv'):
        nlm2_090_filename = dir_out + f'nlm/nlm_090_lmax{lmax}_seed{sim2}.alm'
        nlm2_150_filename = dir_out + f'nlm/nlm_150_lmax{lmax}_seed{sim2}.alm'
        nlm2_220_filename = dir_out + f'nlm/nlm_220_lmax{lmax}_seed{sim2}.alm'
        nlmt2_090,nlme2_090,nlmb2_090 = hp.read_alm(nlm2_090_filename,hdu=[1,2,3])
        nlmt2_150,nlme2_150,nlmb2_150 = hp.read_alm(nlm2_150_filename,hdu=[1,2,3])
        nlmt2_220,nlme2_220,nlmb2_220 = hp.read_alm(nlm2_220_filename,hdu=[1,2,3])
        tlm2_150 += nlmt2_150; tlm2_220 += nlmt2_220; tlm2_95 += nlmt2_090
        elm2_150 += nlme2_150; elm2_220 += nlme2_220; elm2_95 += nlme2_090
        blm2_150 += nlmb2_150; blm2_220 += nlmb2_220; blm2_95 += nlmb2_090

    # ILC combine across frequencies (ONLY for sim r)
    if sim1 == 999:
        tlm1 = hp.almxfl(tlm1_95,w_Tmv[0][:lmax+1]) + hp.almxfl(tlm1_150,w_Tmv[1][:lmax+1]) + hp.almxfl(tlm1_220,w_Tmv[2][:lmax+1])
        elm1 = hp.almxfl(elm1_95,w_Emv[0][:lmax+1]) + hp.almxfl(elm1_150,w_Emv[1][:lmax+1]) + hp.almxfl(elm1_220,w_Emv[2][:lmax+1])
        blm1 = hp.almxfl(blm1_95,w_Bmv[0][:lmax+1]) + hp.almxfl(blm1_150,w_Bmv[1][:lmax+1]) + hp.almxfl(blm1_220,w_Bmv[2][:lmax+1])
    elif sim2 == 999:
        tlm2 = hp.almxfl(tlm2_95,w_Tmv[0][:lmax+1]) + hp.almxfl(tlm2_150,w_Tmv[1][:lmax+1]) + hp.almxfl(tlm2_220,w_Tmv[2][:lmax+1])
        elm2 = hp.almxfl(elm2_95,w_Emv[0][:lmax+1]) + hp.almxfl(elm2_150,w_Emv[1][:lmax+1]) + hp.almxfl(elm2_220,w_Emv[2][:lmax+1])
        blm2 = hp.almxfl(blm2_95,w_Bmv[0][:lmax+1]) + hp.almxfl(blm2_150,w_Bmv[1][:lmax+1]) + hp.almxfl(blm2_220,w_Bmv[2][:lmax+1])

    # Get signal + noise residuals spectra for constructing fl filters
    print('Getting signal + noise residuals spectra for filtering')
    # If lmaxT != lmaxP, we add artificial noise in TT for ell > lmaxT
    artificial_noise = np.zeros(lmax+1)
    artificial_noise[lmaxT+2:] = 1.e10
    totalcls_filename = dir_out+f'totalcls/totalcls_average_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_standard.npy'
    if os.path.isfile(totalcls_filename):
        totalcls = np.load(totalcls_filename)
        cltt = totalcls[:,0]; clee = totalcls[:,1]; clbb = totalcls[:,2]; clte = totalcls[:,3]
    elif append == 'standard' or append == 'standard_cinv':
        print(f"Averaged totalcls file doesn't exist yet, getting the totalcls for sim {sim1}, need to average later")
        cltt_mv = hp.alm2cl(tlm1,tlm1) + artificial_noise
        clee = hp.alm2cl(elm1,elm1)
        clbb = hp.alm2cl(blm1,blm1)
        clte = hp.alm2cl(tlm1,elm1)
        totalcls = np.vstack((cltt_mv,clee,clbb,clte)).T
        np.save(dir_out+f'totalcls/totalcls_seed1_{sim1}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_standard.npy',totalcls)
        return
    else:
        print('WARNING: even for CMB-only sims, we want the filters to have the noise residuals if being used for N1 calculation!')
        print("Averaged totalcls file doesn't exist, run this script with append == 'standard'")
        return

    if cinv:
        print('Doing the 1/Dl for GMV...')
        invDl = np.zeros(lmax+1, dtype=np.complex_)
        invDl[lmin:] = 1./(cltt[lmin:]*clee[lmin:] - clte[lmin:]**2)
        flb = np.zeros(lmax+1); flb[lmin:] = 1./clbb[lmin:]

        if qe[0] == 'T': almbar1 = hp.almxfl((hp.almxfl(tlm1,clee)-hp.almxfl(elm1,clte)),invDl)
        if qe[0] == 'E': almbar1 = hp.almxfl((hp.almxfl(elm1,cltt)-hp.almxfl(tlm1,clte)),invDl)
        if qe[0] == 'B': almbar1 = hp.almxfl(blm1,flb)

        if qe[1] == 'T': almbar2 = hp.almxfl((hp.almxfl(tlm2,clee)-hp.almxfl(elm2,clte)),invDl)
        if qe[1] == 'E': almbar2 = hp.almxfl((hp.almxfl(elm2,cltt)-hp.almxfl(tlm2,clte)),invDl)
        if qe[1] == 'B': almbar2 = hp.almxfl(blm2,flb)

    elif gmv:
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

    else:
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

    # Run healqest
    print('Running healqest...')
    if gmv and not cinv:
        q_gmv = qest.qest_gmv(config,cls)
        glm,clm = q_gmv.eval(qe,alm1all,alm2all,totalcls,crossilc=False)
    else:
        q = qest.qest(config,cls)
        glm,clm = q.eval(qe,almbar1,almbar2)
    # Save plm
    Path(dir_out).mkdir(parents=True, exist_ok=True)
    np.save(filename,glm)
    return

if __name__ == '__main__':

    main()
