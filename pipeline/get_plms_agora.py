#!/usr/bin/env python3
# Run like python3 get_plms.py TT 100 101 append test_yuka.yaml
# Note: argument append should be 'standard', 'mh', 'crossilc_onesed', 'crossilc_twoseds'
# For cinv-style, append should be 'standard_cinv', etc.
# If MH or cross-ILC, have another argument 'noT3' or 'withT3'
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
    fg_model = str(sys.argv[4])
    if len(sys.argv) > 5:
        T3_opt = str(sys.argv[5])
    else:
        T3_opt = ''

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

    if append[-4:] != 'cinv' and (qe == 'TT' or qe == 'TE' or  qe == 'ET' or qe == 'EE' or qe == 'TB' or  qe == 'BT' or qe == 'EB' or  qe == 'BE' or qe == 'TTprf' or qe == 'T1T2' or qe == 'T2T1' or qe == 'T2E1' or qe == 'E2T1' or qe == 'E2E1' or qe == 'T2B1' or qe == 'B2T1' or qe == 'E2B1' or qe == 'B2E1'):
        # SQE
        gmv = False
        if T3_opt == 'noT3' or T3_opt == 'withT3':
            filename = dir_out+f'/plm_{qe}_healqest_sqe_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_{T3_opt}.npy'
        else:
            #TODO
            #filename = dir_out+f'/plm_{qe}_healqest_sqe_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_tszcibonly.npy'
            #filename = dir_out+f'/plm_{qe}_healqest_sqe_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_nonoise.npy'
            #filename = dir_out+f'/plm_{qe}_healqest_sqe_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_nofg.npy'
            filename = dir_out+f'/plm_{qe}_healqest_sqe_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy'
    elif append[-4:] == 'cinv':
        gmv = True
        if T3_opt == 'noT3' or T3_opt == 'withT3':
            filename = dir_out+f'/plm_{qe}_healqest_gmv_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_{T3_opt}.npy'
        else:
            #TODO
            #filename = dir_out+f'/plm_{qe}_healqest_gmv_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_lcmbonly.npy'
            #filename = dir_out+f'/plm_{qe}_healqest_gmv_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_tszcibonly.npy'
            #filename = dir_out+f'/plm_{qe}_healqest_gmv_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_nonoise.npy'
            #filename = dir_out+f'/plm_{qe}_healqest_gmv_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_nofg.npy'
            filename = dir_out+f'/plm_{qe}_healqest_gmv_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy'
    else:
        # GMV
        gmv = True
        if T3_opt == 'noT3' or T3_opt == 'withT3':
            filename = dir_out+f'/plm_{qe}_healqest_gmv_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}_{T3_opt}.npy'
        else:
            filename = dir_out+f'/plm_{qe}_healqest_gmv_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy'

    if append[-4:] == 'cinv' and gmv:
        cinv = True
    else:
        cinv = False

    print(f'cinv is {cinv}, gmv is {gmv}, filename {filename}')

    if os.path.isfile(filename):
        print('File already exists!')
    else:
        do_reconstruction(qe,append,config_file,filename,gmv,cinv,T3_opt,fg_model)

    elapsed = time() - time0
    elapsed /= 60
    print('Time taken (minutes): ', elapsed)

def do_reconstruction(qe,append,config_file,filename,gmv,cinv,T3_opt,fg_model):
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

    # Agora sims
    agora_095 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_95ghz_alm_lmax4096.fits'
    agora_150 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_150ghz_alm_lmax4096.fits'
    agora_220 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_220ghz_alm_lmax4096.fits'
    agora_095_nofg = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_lcmb_spt3g_95ghz_alm_lmax4096.fits'
    agora_150_nofg = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_lcmb_spt3g_150ghz_alm_lmax4096.fits'
    agora_220_nofg = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_lcmb_spt3g_220ghz_alm_lmax4096.fits'

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

    # ILC weights
    if fg_model == 'agora':
        w_agora = np.load('/home/users/yukanaka/gmv/pipeline/ilc_weights/ilc_weights_cmb_spt3g_2yr.npy',allow_pickle=True).item()
        # Dimension (3, 4097) for 90, 150, 220 GHz respectively
        w_Tmv = w_agora['tt']['mv'];
        w_Emv = w_agora['ee']['mv'];
        w_Bmv = w_agora['bb']['mv'];
        w_tsz_null = w_agora['tt']['tsznull']
        w_cib_null = w_agora['tt']['cibnull']
        # Dimension (3, 6001) for 90, 150, 220 GHz respectively
        #w_tsz_null = np.loadtxt('ilc_weights/weights1d_TT_spt3g_cmbynull.dat')
        #w_Tmv = np.loadtxt('ilc_weights/weights1d_TT_spt3g_cmbmv.dat')
        #w_Emv = np.loadtxt('ilc_weights/weights1d_EE_spt3g_cmbmv.dat')
        #w_Bmv = np.loadtxt('ilc_weights/weights1d_BB_spt3g_cmbmv.dat')
        #if append == 'crossilc_onesed' or append == 'crossilc_onesed_cinv':
        #    w_cib_null = np.load('ilc_weights/weights_cmb_mv_cmbfree_cibfree_spt3g1920.npy',allow_pickle=True)
        #    w_cib_null_95 = w_cib_null.item()['cmbcibfree'][95][1]
        #    w_cib_null_150 = w_cib_null.item()['cmbcibfree'][150][1]
        #    w_cib_null_220 = w_cib_null.item()['cmbcibfree'][220][1]
        #elif append == 'crossilc_twoseds' or append == 'crossilc_twoseds_cinv':
        #    w_cib_null = np.load('ilc_weights/weights_cmb_mv_cmbfree_cibfreetwoSEDs_spt3g1920.npy',allow_pickle=True)
        #    w_cib_null_95 = w_cib_null.item()['cmbcibfree'][95][1]
        #    w_cib_null_150 = w_cib_null.item()['cmbcibfree'][150][1]
        #    w_cib_null_220 = w_cib_null.item()['cmbcibfree'][220][1]
    elif fg_model == 'websky':
        w = np.load('/home/users/yukanaka/gmv/pipeline/ilc_weights/ilc_weights_cmb_spt3g_2yr_websky.npy',allow_pickle=True).item()
        cases = ['mv', 'tsznull', 'cibnull']
        # Dimension (3, 4097) for 90, 150, 220 GHz respectively
        w_Tmv = w['tt']['mv'];
        w_Emv = w['ee']['mv'];
        w_Bmv = w['bb']['mv'];
        w_tsz_null = w['tt']['tsznull']
        w_cib_null = w['tt']['cibnull']

    if fg_model == 'agora':
        print('Getting alms...')
        #TODO: comment in following six lines and comment out rest if nofg sanity check for standard
        '''
        tlm_mv, elm, blm = hp.read_alm(agora_095_nofg,hdu=[1,2,3])
        # Add GAUSSIAN foregrounds and noise so I can still use N0 and N1 from sims
        fnlm_sim = f'/oak/stanford/orgs/kipac/users/yukanaka/gmv/inputs/spt3g_2019_2020_agora/agora_fg_plus_spt3g_20192020_noise_lmax{lmax}_seed999_mv.fits'
        tfnlm, efnlm, bfnlm = hp.read_alm(fnlm_sim,hdu=[1,2,3])
        tfnlm = utils.reduce_lmax(tfnlm,lmax=lmax); efnlm = utils.reduce_lmax(efnlm,lmax=lmax); bfnlm = utils.reduce_lmax(bfnlm,lmax=lmax)
        tlm_mv += tfnlm; elm += efnlm; blm += bfnlm
        '''
        #TODO: comment in following three lines and comment out rest if doing sanity check where I only have one NG fg at a time for standard
        '''
        tlm_95, elm_95, blm_95 = hp.read_alm('/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lcmb_ltsz_lcib_spt3g_95ghz_alm_lmax4096.fits',hdu=[1,2,3])
        tlm_150, elm_150, blm_150 = hp.read_alm('/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lcmb_ltsz_lcib_spt3g_150ghz_alm_lmax4096.fits',hdu=[1,2,3])
        tlm_220, elm_220, blm_220 = hp.read_alm('/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lcmb_ltsz_lcib_spt3g_220ghz_alm_lmax4096.fits',hdu=[1,2,3])
        '''
        # Get Agora sim (signal + foregrounds)
        tlm_95, elm_95, blm_95 = hp.read_alm(agora_095,hdu=[1,2,3])
        tlm_150, elm_150, blm_150 = hp.read_alm(agora_150,hdu=[1,2,3])
        tlm_220, elm_220, blm_220 = hp.read_alm(agora_220,hdu=[1,2,3])
    elif fg_model == 'websky':
        print('Getting alms...')
        #TODO: comment in following two lines and comment out rest if lcmb only WebSky sanity check
        '''
        tlm_mv, elm, blm = hp.read_alm(websky_nofg,hdu=[1,2,3])
        tlm_mv = utils.reduce_lmax(tlm_mv,lmax=lmax); elm = utils.reduce_lmax(elm,lmax=lmax); blm = utils.reduce_lmax(blm,lmax=lmax);
        '''
        # Get Websky sim (signal + foregrounds)
        t_95_websky, q_95_websky, u_95_websky = hp.read_map(websky_095,field=[1,2,3])
        t_150_websky, q_150_websky, u_150_websky = hp.read_map(websky_150,field=[1,2,3])
        t_220_websky, q_220_websky, u_220_websky = hp.read_map(websky_220,field=[1,2,3])
        tlm_95_websky, elm_95_websky, blm_95_websky = hp.map2alm([t_95_websky, q_95_websky, u_95_websky], lmax=lmax)
        tlm_150_websky, elm_150_websky, blm_150_websky = hp.map2alm([t_150_websky, q_150_websky, u_150_websky], lmax=lmax)
        tlm_220_websky, elm_220_websky, blm_220_websky = hp.map2alm([t_220_websky, q_220_websky, u_220_websky], lmax=lmax)

    # Adding noise! (Using sim 999)
    #TODO: comment out entire block if lcmb only WebSky sanity check OR nofg sanity check for Agora standard OR one fg sanity check
    nlm1_090_filename = dir_out + f'nlm/nlm_090_lmax{lmax}_seed999.alm'
    nlm1_150_filename = dir_out + f'nlm/nlm_150_lmax{lmax}_seed999.alm'
    nlm1_220_filename = dir_out + f'nlm/nlm_220_lmax{lmax}_seed999.alm'
    if os.path.isfile(nlm1_090_filename):
        nlmt1_090,nlme1_090,nlmb1_090 = hp.read_alm(nlm1_090_filename,hdu=[1,2,3])
        nlmt1_150,nlme1_150,nlmb1_150 = hp.read_alm(nlm1_150_filename,hdu=[1,2,3])
        nlmt1_220,nlme1_220,nlmb1_220 = hp.read_alm(nlm1_220_filename,hdu=[1,2,3])
    tlm_150 += nlmt1_150; tlm_220 += nlmt1_220; tlm_95 += nlmt1_090
    elm_150 += nlme1_150; elm_220 += nlme1_220; elm_95 += nlme1_090
    blm_150 += nlmb1_150; blm_220 += nlmb1_220; blm_95 += nlmb1_090

    # ILC combine frequencies
    #TODO: comment out entire block if lcmb only WebSky sanity check OR nofg sanity check for Agora standard
    if fg_model == 'agora':
        tlm_mv = hp.almxfl(tlm_95,w_Tmv[0][:lmax+1]) + hp.almxfl(tlm_150,w_Tmv[1][:lmax+1]) + hp.almxfl(tlm_220,w_Tmv[2][:lmax+1])
        tlm_tszn = hp.almxfl(tlm_95,w_tsz_null[0][:lmax+1]) + hp.almxfl(tlm_150,w_tsz_null[1][:lmax+1]) + hp.almxfl(tlm_220,w_tsz_null[2][:lmax+1])
        tlm_cibn = hp.almxfl(tlm_95,w_cib_null[0][:lmax+1]) + hp.almxfl(tlm_150,w_cib_null[1][:lmax+1]) + hp.almxfl(tlm_220,w_cib_null[2][:lmax+1])
        elm = hp.almxfl(elm_95,w_Emv[0][:lmax+1]) + hp.almxfl(elm_150,w_Emv[1][:lmax+1]) + hp.almxfl(elm_220,w_Emv[2][:lmax+1])
        blm = hp.almxfl(blm_95,w_Bmv[0][:lmax+1]) + hp.almxfl(blm_150,w_Bmv[1][:lmax+1]) + hp.almxfl(blm_220,w_Bmv[2][:lmax+1])
    elif fg_model == 'websky':
        # ILC combine frequencies                                               
        tlm_mv = hp.almxfl(tlm_95,w_Tmv[0][:lmax+1]) + hp.almxfl(tlm_150,w_Tmv[1][:lmax+1]) + hp.almxfl(tlm_220,w_Tmv[2][:lmax+1])
        tlm_tszn = hp.almxfl(tlm_95,w_tsz_null[0][:lmax+1]) + hp.almxfl(tlm_150,w_tsz_null[1][:lmax+1]) + hp.almxfl(tlm_220,w_tsz_null[2][:lmax+1])
        tlm_cibn = hp.almxfl(tlm_95,w_cib_null[0][:lmax+1]) + hp.almxfl(tlm_150,w_cib_null[0][:lmax+1]) + hp.almxfl(tlm_220,w_cib_null[0][:lmax+1])
        elm = hp.almxfl(elm_95,w_Emv[0][:lmax+1]) + hp.almxfl(elm_150,w_Emv[1][:lmax+1]) + hp.almxfl(elm_220,w_Emv[2][:lmax+1])
        blm = hp.almxfl(blm_95,w_Bmv[0][:lmax+1]) + hp.almxfl(blm_150,w_Bmv[1][:lmax+1]) + hp.almxfl(blm_220,w_Bmv[2][:lmax+1])

    #TODO: comment in following lines if doing sanity check where I only have one NG fg at a time for standard
    '''
    # Add GAUSSIAN foregrounds and noise so I can still use N0 and N1 from sims
    fnlm_sim = f'/oak/stanford/orgs/kipac/users/yukanaka/gmv/inputs/spt3g_2019_2020_agora/agora_fg_plus_spt3g_20192020_noise_noltszlcib_lmax{lmax}_seed999_mv.fits'
    tfnlm, efnlm, bfnlm = hp.read_alm(fnlm_sim,hdu=[1,2,3])
    tfnlm = utils.reduce_lmax(tfnlm,lmax=lmax); efnlm = utils.reduce_lmax(efnlm,lmax=lmax); bfnlm = utils.reduce_lmax(bfnlm,lmax=lmax)
    tlm_mv += tfnlm; elm += efnlm; blm += bfnlm
    '''

    # Get signal + noise residuals spectra for constructing fl filters
    print('Getting signal + noise residuals spectra for filtering')
    # If lmaxT != lmaxP, we add artificial noise in TT for ell > lmaxT
    artificial_noise = np.zeros(lmax+1)
    artificial_noise[lmaxT+2:] = 1.e10
    if cinv:
        append_shortened = append[:-5]
    else:
        append_shortened = append
    # Use the SAME filter for these Agora sims as what I used for the Gaussian sims, otherwise sim response from Gaussian sims won't match when I use it for Agora
    totalcls_filename = dir_out+f'totalcls/totalcls_average_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append_shortened}.npy'
    totalcls = np.load(totalcls_filename)
    if append == 'standard' or append == 'standard_cinv':
        cltt = totalcls[:,0]; clee = totalcls[:,1]; clbb = totalcls[:,2]; clte = totalcls[:,3]
    elif append == 'mh' or append == 'crossilc_onesed' or append == 'crossilc_twoseds' or append == 'mh_cinv' or append == 'crossilc_onesed_cinv' or append == 'crossilc_twoseds_cinv':
        # totalcls: T3T3, EE, BB, T3E, T1T1, T2T2, T1T2, T1T3, T2T3, T1E, T2E
        cltt1 = totalcls[:,4]; cltt2 = totalcls[:,5]; clttx = totalcls[:,6]; cltt3 = totalcls[:,0]; clee = totalcls[:,1]; clbb = totalcls[:,2]; clte = totalcls[:,3]

    if append == 'standard_cinv':
        print('Doing the 1/Dl for GMV...')
        invDl = np.zeros(lmax+1, dtype=np.complex_)
        invDl[lmin:] = 1./(cltt[lmin:]*clee[lmin:] - clte[lmin:]**2)
        flb = np.zeros(lmax+1); flb[lmin:] = 1./clbb[lmin:]

        if qe[0] == 'T': almbar1 = hp.almxfl((hp.almxfl(tlm_mv,clee)-hp.almxfl(elm,clte)),invDl)
        if qe[0] == 'E': almbar1 = hp.almxfl((hp.almxfl(elm,cltt)-hp.almxfl(tlm_mv,clte)),invDl)
        if qe[0] == 'B': almbar1 = hp.almxfl(blm,flb)

        if qe[1] == 'T': almbar2 = hp.almxfl((hp.almxfl(tlm_mv,clee)-hp.almxfl(elm,clte)),invDl)
        if qe[1] == 'E': almbar2 = hp.almxfl((hp.almxfl(elm,cltt)-hp.almxfl(tlm_mv,clte)),invDl)
        if qe[1] == 'B': almbar2 = hp.almxfl(blm,flb)

    elif append == 'mh_cinv':
        print('Doing the 1/Dl for GMV...')
        invDl1 = np.zeros(lmax+1, dtype=np.complex_)
        invDl2 = np.zeros(lmax+1, dtype=np.complex_)
        invDl3 = np.zeros(lmax+1, dtype=np.complex_)
        invDl1[lmin:] = 1./(cltt1[lmin:]*clee[lmin:] - clte[lmin:]**2)
        invDl2[lmin:] = 1./(cltt2[lmin:]*clee[lmin:] - clte[lmin:]**2)
        invDl3[lmin:] = 1./(cltt3[lmin:]*clee[lmin:] - clte[lmin:]**2)
        flb = np.zeros(lmax+1); flb[lmin:] = 1./clbb[lmin:]

        if qe == 'T1T2':
            almbar1 = hp.almxfl((hp.almxfl(tlm_mv,clee)-hp.almxfl(elm,clte)),invDl1)
            almbar2 = hp.almxfl((hp.almxfl(tlm_tszn,clee)-hp.almxfl(elm,clte)),invDl2)
        elif qe == 'T2T1':
            almbar1 = hp.almxfl((hp.almxfl(tlm_tszn,clee)-hp.almxfl(elm,clte)),invDl2)
            almbar2 = hp.almxfl((hp.almxfl(tlm_mv,clee)-hp.almxfl(elm,clte)),invDl1)
        elif qe == 'T2E1':
            if T3_opt == 'noT3':
                almbar1 = hp.almxfl((hp.almxfl(tlm_tszn,clee)-hp.almxfl(elm,clte)),invDl2)
                almbar2 = hp.almxfl((hp.almxfl(elm,cltt1)-hp.almxfl(tlm_mv,clte)),invDl1)
            elif T3_opt == 'withT3':
                almbar1 = hp.almxfl((hp.almxfl(tlm_mv,clee)-hp.almxfl(elm,clte)),invDl3)
                almbar2 = hp.almxfl((hp.almxfl(elm,cltt3)-hp.almxfl(tlm_mv,clte)),invDl3)
        elif qe == 'E2T1':
            if T3_opt == 'noT3':
                almbar1 = hp.almxfl((hp.almxfl(elm,cltt2)-hp.almxfl(tlm_tszn,clte)),invDl2)
                almbar2 = hp.almxfl((hp.almxfl(tlm_mv,clee)-hp.almxfl(elm,clte)),invDl1)
            elif T3_opt == 'withT3':
                almbar1 = hp.almxfl((hp.almxfl(elm,cltt3)-hp.almxfl(tlm_mv,clte)),invDl3)
                almbar2 = hp.almxfl((hp.almxfl(tlm_mv,clee)-hp.almxfl(elm,clte)),invDl3)
        elif qe == 'E2E1':
            if T3_opt == 'noT3':
                almbar1 = hp.almxfl((hp.almxfl(elm,cltt2)-hp.almxfl(tlm_tszn,clte)),invDl2)
                almbar2 = hp.almxfl((hp.almxfl(elm,cltt1)-hp.almxfl(tlm_mv,clte)),invDl1)
            elif T3_opt == 'withT3':
                almbar1 = hp.almxfl((hp.almxfl(elm,cltt3)-hp.almxfl(tlm_mv,clte)),invDl3)
                almbar2 = hp.almxfl((hp.almxfl(elm,cltt3)-hp.almxfl(tlm_mv,clte)),invDl3)
        elif qe == 'T2B1':
            if T3_opt == 'noT3':
                almbar1 = hp.almxfl((hp.almxfl(tlm_tszn,clee)-hp.almxfl(elm,clte)),invDl2)
                almbar2 = hp.almxfl(blm,flb)
            elif T3_opt == 'withT3':
                almbar1 = hp.almxfl((hp.almxfl(tlm_mv,clee)-hp.almxfl(elm,clte)),invDl3)
                almbar2 = hp.almxfl(blm,flb)
        elif qe == 'B2T1':
            if T3_opt == 'noT3':
                almbar1 = hp.almxfl(blm,flb)
                almbar2 = hp.almxfl((hp.almxfl(tlm_mv,clee)-hp.almxfl(elm,clte)),invDl1)
            elif T3_opt == 'withT3':
                almbar1 = hp.almxfl(blm,flb)
                almbar2 = hp.almxfl((hp.almxfl(tlm_mv,clee)-hp.almxfl(elm,clte)),invDl3)
        elif qe == 'E2B1':
            if T3_opt == 'noT3':
                almbar1 = hp.almxfl((hp.almxfl(elm,cltt2)-hp.almxfl(tlm_tszn,clte)),invDl2)
                almbar2 = hp.almxfl(blm,flb)
            elif T3_opt == 'withT3':
                almbar1 = hp.almxfl((hp.almxfl(elm,cltt3)-hp.almxfl(tlm_mv,clte)),invDl3)
                almbar2 = hp.almxfl(blm,flb)
        elif qe == 'B2E1':
            if T3_opt == 'noT3':
                almbar1 = hp.almxfl(blm,flb)
                almbar2 = hp.almxfl((hp.almxfl(elm,cltt1)-hp.almxfl(tlm_mv,clte)),invDl1)
            elif T3_opt == 'withT3':
                almbar1 = hp.almxfl(blm,flb)
                almbar2 = hp.almxfl((hp.almxfl(elm,cltt3)-hp.almxfl(tlm_mv,clte)),invDl3)
        else:
            if T3_opt == 'noT3':
                if qe[0] == 'T': almbar1 = hp.almxfl((hp.almxfl(tlm_mv,clee)-hp.almxfl(elm,clte)),invDl1)
                elif qe[0] == 'E': almbar1 = hp.almxfl((hp.almxfl(elm,cltt1)-hp.almxfl(tlm_mv,clte)),invDl1)
                elif qe[0] == 'B': almbar1 = hp.almxfl(blm,flb)

                if qe[1] == 'T': almbar2 = hp.almxfl((hp.almxfl(tlm_tszn,clee)-hp.almxfl(elm,clte)),invDl2)
                elif qe[1] == 'E': almbar2 = hp.almxfl((hp.almxfl(elm,cltt2)-hp.almxfl(tlm_tszn,clte)),invDl2)
                elif qe[1] == 'B': almbar2 = hp.almxfl(blm,flb)
            elif T3_opt == 'withT3':
                if qe[0] == 'T': almbar1 = hp.almxfl((hp.almxfl(tlm_mv,clee)-hp.almxfl(elm,clte)),invDl3)
                elif qe[0] == 'E': almbar1 = hp.almxfl((hp.almxfl(elm,cltt3)-hp.almxfl(tlm_mv,clte)),invDl3)
                elif qe[0] == 'B': almbar1 = hp.almxfl(blm,flb)

                if qe[1] == 'T': almbar2 = hp.almxfl((hp.almxfl(tlm_mv,clee)-hp.almxfl(elm,clte)),invDl3)
                elif qe[1] == 'E': almbar2 = hp.almxfl((hp.almxfl(elm,cltt3)-hp.almxfl(tlm_mv,clte)),invDl3)
                elif qe[1] == 'B': almbar2 = hp.almxfl(blm,flb)

    elif append == 'crossilc_onesed_cinv' or append == 'crossilc_twoseds_cinv':
        print('Doing the 1/Dl for GMV...')
        invDl1 = np.zeros(lmax+1, dtype=np.complex_)
        invDl2 = np.zeros(lmax+1, dtype=np.complex_)
        invDl3 = np.zeros(lmax+1, dtype=np.complex_)
        invDl1[lmin:] = 1./(cltt1[lmin:]*clee[lmin:] - clte[lmin:]**2)
        invDl2[lmin:] = 1./(cltt2[lmin:]*clee[lmin:] - clte[lmin:]**2)
        invDl3[lmin:] = 1./(cltt3[lmin:]*clee[lmin:] - clte[lmin:]**2)
        flb = np.zeros(lmax+1); flb[lmin:] = 1./clbb[lmin:]

        if qe == 'T1T2':
            almbar1 = hp.almxfl((hp.almxfl(tlm_cibn,clee)-hp.almxfl(elm,clte)),invDl1)
            almbar2 = hp.almxfl((hp.almxfl(tlm_tszn,clee)-hp.almxfl(elm,clte)),invDl2)
        elif qe == 'T2T1':
            almbar1 = hp.almxfl((hp.almxfl(tlm_tszn,clee)-hp.almxfl(elm,clte)),invDl2)
            almbar2 = hp.almxfl((hp.almxfl(tlm_cibn,clee)-hp.almxfl(elm,clte)),invDl1)
        elif qe == 'T2E1':
            if T3_opt == 'noT3':
                almbar1 = hp.almxfl((hp.almxfl(tlm_tszn,clee)-hp.almxfl(elm,clte)),invDl2)
                almbar2 = hp.almxfl((hp.almxfl(elm,cltt1)-hp.almxfl(tlm_cibn,clte)),invDl1)
            elif T3_opt == 'withT3':
                almbar1 = hp.almxfl((hp.almxfl(tlm_mv,clee)-hp.almxfl(elm,clte)),invDl3)
                almbar2 = hp.almxfl((hp.almxfl(elm,cltt3)-hp.almxfl(tlm_mv,clte)),invDl3)
        elif qe == 'E2T1':
            if T3_opt == 'noT3':
                almbar1 = hp.almxfl((hp.almxfl(elm,cltt2)-hp.almxfl(tlm_tszn,clte)),invDl2)
                almbar2 = hp.almxfl((hp.almxfl(tlm_cibn,clee)-hp.almxfl(elm,clte)),invDl1)
            elif T3_opt == 'withT3':
                almbar1 = hp.almxfl((hp.almxfl(elm,cltt3)-hp.almxfl(tlm_mv,clte)),invDl3)
                almbar2 = hp.almxfl((hp.almxfl(tlm_mv,clee)-hp.almxfl(elm,clte)),invDl3)
        elif qe == 'E2E1':
            if T3_opt == 'noT3':
                almbar1 = hp.almxfl((hp.almxfl(elm,cltt2)-hp.almxfl(tlm_tszn,clte)),invDl2)
                almbar2 = hp.almxfl((hp.almxfl(elm,cltt1)-hp.almxfl(tlm_cibn,clte)),invDl1)
            elif T3_opt == 'withT3':
                almbar1 = hp.almxfl((hp.almxfl(elm,cltt3)-hp.almxfl(tlm_mv,clte)),invDl3)
                almbar2 = hp.almxfl((hp.almxfl(elm,cltt3)-hp.almxfl(tlm_mv,clte)),invDl3)
        elif qe == 'T2B1':
            if T3_opt == 'noT3':
                almbar1 = hp.almxfl((hp.almxfl(tlm_tszn,clee)-hp.almxfl(elm,clte)),invDl2)
                almbar2 = hp.almxfl(blm,flb)
            elif T3_opt == 'withT3':
                almbar1 = hp.almxfl((hp.almxfl(tlm_mv,clee)-hp.almxfl(elm,clte)),invDl3)
                almbar2 = hp.almxfl(blm,flb)
        elif qe == 'B2T1':
            if T3_opt == 'noT3':
                almbar1 = hp.almxfl(blm,flb)
                almbar2 = hp.almxfl((hp.almxfl(tlm_cibn,clee)-hp.almxfl(elm,clte)),invDl1)
            elif T3_opt == 'withT3':
                almbar1 = hp.almxfl(blm,flb)
                almbar2 = hp.almxfl((hp.almxfl(tlm_mv,clee)-hp.almxfl(elm,clte)),invDl3)
        elif qe == 'E2B1':
            if T3_opt == 'noT3':
                almbar1 = hp.almxfl((hp.almxfl(elm,cltt2)-hp.almxfl(tlm_tszn,clte)),invDl2)
                almbar2 = hp.almxfl(blm,flb)
            elif T3_opt == 'withT3':
                almbar1 = hp.almxfl((hp.almxfl(elm,cltt3)-hp.almxfl(tlm_mv,clte)),invDl3)
                almbar2 = hp.almxfl(blm,flb)
        elif qe == 'B2E1':
            if T3_opt == 'noT3':
                almbar1 = hp.almxfl(blm,flb)
                almbar2 = hp.almxfl((hp.almxfl(elm,cltt1)-hp.almxfl(tlm_cibn,clte)),invDl1)
            elif T3_opt == 'withT3':
                almbar1 = hp.almxfl(blm,flb)
                almbar2 = hp.almxfl((hp.almxfl(elm,cltt3)-hp.almxfl(tlm_mv,clte)),invDl3)
        else:
            if T3_opt == 'noT3':
                if qe[0] == 'T': almbar1 = hp.almxfl((hp.almxfl(tlm_cibn,clee)-hp.almxfl(elm,clte)),invDl1)
                elif qe[0] == 'E': almbar1 = hp.almxfl((hp.almxfl(elm,cltt1)-hp.almxfl(tlm_cibn,clte)),invDl1)
                elif qe[0] == 'B': almbar1 = hp.almxfl(blm,flb)

                if qe[1] == 'T': almbar2 = hp.almxfl((hp.almxfl(tlm_tszn,clee)-hp.almxfl(elm,clte)),invDl2)
                elif qe[1] == 'E': almbar2 = hp.almxfl((hp.almxfl(elm,cltt2)-hp.almxfl(tlm_tszn,clte)),invDl2)
                elif qe[1] == 'B': almbar2 = hp.almxfl(blm,flb)
            elif T3_opt == 'withT3':
                if qe[0] == 'T': almbar1 = hp.almxfl((hp.almxfl(tlm_mv,clee)-hp.almxfl(elm,clte)),invDl3)
                elif qe[0] == 'E': almbar1 = hp.almxfl((hp.almxfl(elm,cltt3)-hp.almxfl(tlm_mv,clte)),invDl3)
                elif qe[0] == 'B': almbar1 = hp.almxfl(blm,flb)

                if qe[1] == 'T': almbar2 = hp.almxfl((hp.almxfl(tlm_mv,clee)-hp.almxfl(elm,clte)),invDl3)
                elif qe[1] == 'E': almbar2 = hp.almxfl((hp.almxfl(elm,cltt3)-hp.almxfl(tlm_mv,clte)),invDl3)
                elif qe[1] == 'B': almbar2 = hp.almxfl(blm,flb)

    elif append == 'standard' and gmv:
        print('Doing the 1/Dl for GMV...')
        invDl = np.zeros(lmax+1, dtype=np.complex_)
        invDl[lmin:] = 1./(cltt[lmin:]*clee[lmin:] - clte[lmin:]**2)
        flb = np.zeros(lmax+1); flb[lmin:] = 1./clbb[lmin:]

        # Order is TT, EE, TE, ET, TB, BT, EB, BE
        alm1all = np.zeros((len(tlm_mv),8), dtype=np.complex_)
        alm2all = np.zeros((len(tlm_mv),8), dtype=np.complex_)
        # TT
        alm1all[:,0] = hp.almxfl(tlm_mv,invDl)
        alm2all[:,0] = hp.almxfl(tlm_mv,invDl)
        # EE
        alm1all[:,1] = hp.almxfl(elm,invDl)
        alm2all[:,1] = hp.almxfl(elm,invDl)
        # TE
        alm1all[:,2] = hp.almxfl(tlm_mv,invDl)
        alm2all[:,2] = hp.almxfl(elm,invDl)
        # ET
        alm1all[:,3] = hp.almxfl(elm,invDl)
        alm2all[:,3] = hp.almxfl(tlm_mv,invDl)
        # TB
        alm1all[:,4] = hp.almxfl(tlm_mv,invDl)
        alm2all[:,4] = hp.almxfl(blm,flb)
        # BT
        alm1all[:,5] = hp.almxfl(blm,flb)
        alm2all[:,5] = hp.almxfl(tlm_mv,invDl)
        # EB
        alm1all[:,6] = hp.almxfl(elm,invDl)
        alm2all[:,6] = hp.almxfl(blm,flb)
        # BE
        alm1all[:,7] = hp.almxfl(blm,flb)
        alm2all[:,7] = hp.almxfl(elm,invDl)

    elif append == 'mh' and gmv:
        # TODO: This is outdated because it's only 12 ests not 16
        print('Doing the 1/Dl for GMV...')
        invDl1 = np.zeros(lmax+1, dtype=np.complex_)
        invDl2 = np.zeros(lmax+1, dtype=np.complex_)
        invDl3 = np.zeros(lmax+1, dtype=np.complex_)
        invDl1[lmin:] = 1./(cltt1[lmin:]*clee[lmin:] - clte[lmin:]**2)
        invDl2[lmin:] = 1./(cltt2[lmin:]*clee[lmin:] - clte[lmin:]**2)
        invDl3[lmin:] = 1./(cltt3[lmin:]*clee[lmin:] - clte[lmin:]**2)
        flb = np.zeros(lmax+1); flb[lmin:] = 1./clbb[lmin:]

        # Order is T1T2, T2T1, EE, E2E1, TE, T2E1, ET, E2T1, TB, T2B1, BT, B2T1, EB, E2B1, BE, B2E1
        alm1all = np.zeros((len(tlm_mv),16), dtype=np.complex_)
        alm2all = np.zeros((len(tlm_tszn),16), dtype=np.complex_)
        # T1T2
        alm1all[:,0] = hp.almxfl(tlm_mv,invDl1)
        alm2all[:,0] = hp.almxfl(tlm_tszn,invDl2)
        # T2T1
        alm1all[:,1] = hp.almxfl(tlm_tszn,invDl2)
        alm2all[:,1] = hp.almxfl(tlm_mv,invDl1)
        if T3_opt == 'noT3':
            # EE
            alm1all[:,2] = hp.almxfl(elm,invDl1)
            alm2all[:,2] = hp.almxfl(elm,invDl2)
            # E2E1
            alm1all[:,3] = hp.almxfl(elm,invDl2)
            alm2all[:,3] = hp.almxfl(elm,invDl1)
            # TE
            alm1all[:,4] = hp.almxfl(tlm_mv,invDl1)
            alm2all[:,4] = hp.almxfl(elm,invDl2)
            # T2E1
            alm1all[:,5] = hp.almxfl(tlm_tszn,invDl2)
            alm2all[:,5] = hp.almxfl(elm,invDl1)
            # ET
            alm1all[:,6] = hp.almxfl(elm,invDl1)
            alm2all[:,6] = hp.almxfl(tlm_tszn,invDl2)
            # E2T1
            alm1all[:,7] = hp.almxfl(elm,invDl2)
            alm2all[:,7] = hp.almxfl(tlm_mv,invDl1)
            # TB
            alm1all[:,8] = hp.almxfl(tlm_mv,invDl1)
            alm2all[:,8] = hp.almxfl(blm,flb)
            # T2B1
            alm1all[:,9] = hp.almxfl(tlm_tszn,invDl2)
            alm2all[:,9] = hp.almxfl(blm,flb)
            # BT
            alm1all[:,10] = hp.almxfl(blm,flb)
            alm2all[:,10] = hp.almxfl(tlm_tszn,invDl2)
            # B2T1
            alm1all[:,11] = hp.almxfl(blm,flb)
            alm2all[:,11] = hp.almxfl(tlm_mv,invDl1)
            # EB
            alm1all[:,12] = hp.almxfl(elm,invDl1)
            alm2all[:,12] = hp.almxfl(blm,flb)
            # E2B1
            alm1all[:,13] = hp.almxfl(elm,invDl2)
            alm2all[:,13] = hp.almxfl(blm,flb)
            # BE
            alm1all[:,14] = hp.almxfl(blm,flb)
            alm2all[:,14] = hp.almxfl(elm,invDl2)
            # B2E1
            alm1all[:,15] = hp.almxfl(blm,flb)
            alm2all[:,15] = hp.almxfl(elm,invDl1)
        elif T3_opt == 'withT3':
            # EE
            alm1all[:,2] = hp.almxfl(elm,invDl3)
            alm2all[:,2] = hp.almxfl(elm,invDl3)
            # E2E1
            alm1all[:,3] = hp.almxfl(elm,invDl3)
            alm2all[:,3] = hp.almxfl(elm,invDl3)
            # TE
            alm1all[:,4] = hp.almxfl(tlm_mv,invDl3)
            alm2all[:,4] = hp.almxfl(elm,invDl3)
            # T2E1
            alm1all[:,5] = hp.almxfl(tlm_mv,invDl3)
            alm2all[:,5] = hp.almxfl(elm,invDl3)
            # ET
            alm1all[:,6] = hp.almxfl(elm,invDl3)
            alm2all[:,6] = hp.almxfl(tlm_mv,invDl3)
            # E2T1
            alm1all[:,7] = hp.almxfl(elm,invDl3)
            alm2all[:,7] = hp.almxfl(tlm_mv,invDl3)
            # TB
            alm1all[:,8] = hp.almxfl(tlm_mv,invDl3)
            alm2all[:,8] = hp.almxfl(blm,flb)
            # T2B1
            alm1all[:,9] = hp.almxfl(tlm_mv,invDl3)
            alm2all[:,9] = hp.almxfl(blm,flb)
            # BT
            alm1all[:,10] = hp.almxfl(blm,flb)
            alm2all[:,10] = hp.almxfl(tlm_mv,invDl3)
            # B2T1
            alm1all[:,11] = hp.almxfl(blm,flb)
            alm2all[:,11] = hp.almxfl(tlm_mv,invDl3)
            # EB
            alm1all[:,12] = hp.almxfl(elm,invDl3)
            alm2all[:,12] = hp.almxfl(blm,flb)
            # E2B1
            alm1all[:,13] = hp.almxfl(elm,invDl3)
            alm2all[:,13] = hp.almxfl(blm,flb)
            # BE
            alm1all[:,14] = hp.almxfl(blm,flb)
            alm2all[:,14] = hp.almxfl(elm,invDl3)
            # B2E1
            alm1all[:,15] = hp.almxfl(blm,flb)
            alm2all[:,15] = hp.almxfl(elm,invDl3)

    elif (append == 'crossilc_onesed' or append == 'crossilc_twoseds') and gmv:
        print('Doing the 1/Dl for GMV...')
        invDl1 = np.zeros(lmax+1, dtype=np.complex_)
        invDl2 = np.zeros(lmax+1, dtype=np.complex_)
        invDl3 = np.zeros(lmax+1, dtype=np.complex_)
        invDl1[lmin:] = 1./(cltt1[lmin:]*clee[lmin:] - clte[lmin:]**2)
        invDl2[lmin:] = 1./(cltt2[lmin:]*clee[lmin:] - clte[lmin:]**2)
        invDl3[lmin:] = 1./(cltt3[lmin:]*clee[lmin:] - clte[lmin:]**2)
        flb = np.zeros(lmax+1); flb[lmin:] = 1./clbb[lmin:]

        # Order is T1T2, T2T1, EE, E2E1, TE, T2E1, ET, E2T1, TB, T2B1, BT, B2T1, EB, E2B1, BE, B2E1
        alm1all = np.zeros((len(tlm_mv),16), dtype=np.complex_)
        alm2all = np.zeros((len(tlm_tszn),16), dtype=np.complex_)
        # T1T2
        alm1all[:,0] = hp.almxfl(tlm_cibn,invDl1)
        alm2all[:,0] = hp.almxfl(tlm_tszn,invDl2)
        # T2T1
        alm1all[:,1] = hp.almxfl(tlm_tszn,invDl2)
        alm2all[:,1] = hp.almxfl(tlm_cibn,invDl1)
        if T3_opt == 'noT3':
            # EE
            alm1all[:,2] = hp.almxfl(elm,invDl1)
            alm2all[:,2] = hp.almxfl(elm,invDl2)
            # E2E1
            alm1all[:,3] = hp.almxfl(elm,invDl2)
            alm2all[:,3] = hp.almxfl(elm,invDl1)
            # TE
            alm1all[:,4] = hp.almxfl(tlm_cibn,invDl1)
            alm2all[:,4] = hp.almxfl(elm,invDl2)
            # T2E1
            alm1all[:,5] = hp.almxfl(tlm_tszn,invDl2)
            alm2all[:,5] = hp.almxfl(elm,invDl1)
            # ET
            alm1all[:,6] = hp.almxfl(elm,invDl1)
            alm2all[:,6] = hp.almxfl(tlm_tszn,invDl2)
            # E2T1
            alm1all[:,7] = hp.almxfl(elm,invDl2)
            alm2all[:,7] = hp.almxfl(tlm_cibn,invDl1)
            # TB
            alm1all[:,8] = hp.almxfl(tlm_cibn,invDl1)
            alm2all[:,8] = hp.almxfl(blm,flb)
            # T2B1
            alm1all[:,9] = hp.almxfl(tlm_tszn,invDl2)
            alm2all[:,9] = hp.almxfl(blm,flb)
            # BT
            alm1all[:,10] = hp.almxfl(blm,flb)
            alm2all[:,10] = hp.almxfl(tlm_tszn,invDl2)
            # B2T1
            alm1all[:,11] = hp.almxfl(blm,flb)
            alm2all[:,11] = hp.almxfl(tlm_cibn,invDl1)
            # EB
            alm1all[:,12] = hp.almxfl(elm,invDl1)
            alm2all[:,12] = hp.almxfl(blm,flb)
            # E2B1
            alm1all[:,13] = hp.almxfl(elm,invDl2)
            alm2all[:,13] = hp.almxfl(blm,flb)
            # BE
            alm1all[:,14] = hp.almxfl(blm,flb)
            alm2all[:,14] = hp.almxfl(elm,invDl2)
            # B2E1
            alm1all[:,15] = hp.almxfl(blm,flb)
            alm2all[:,15] = hp.almxfl(elm,invDl1)
        elif T3_opt == 'withT3':
            # EE
            alm1all[:,2] = hp.almxfl(elm,invDl3)
            alm2all[:,2] = hp.almxfl(elm,invDl3)
            # E2E1
            alm1all[:,3] = hp.almxfl(elm,invDl3)
            alm2all[:,3] = hp.almxfl(elm,invDl3)
            # TE
            alm1all[:,4] = hp.almxfl(tlm_mv,invDl3)
            alm2all[:,4] = hp.almxfl(elm,invDl3)
            # T2E1
            alm1all[:,5] = hp.almxfl(tlm_mv,invDl3)
            alm2all[:,5] = hp.almxfl(elm,invDl3)
            # ET
            alm1all[:,6] = hp.almxfl(elm,invDl3)
            alm2all[:,6] = hp.almxfl(tlm_mv,invDl3)
            # E2T1
            alm1all[:,7] = hp.almxfl(elm,invDl3)
            alm2all[:,7] = hp.almxfl(tlm_mv,invDl3)
            # TB
            alm1all[:,8] = hp.almxfl(tlm_mv,invDl3)
            alm2all[:,8] = hp.almxfl(blm,flb)
            # T2B1
            alm1all[:,9] = hp.almxfl(tlm_mv,invDl3)
            alm2all[:,9] = hp.almxfl(blm,flb)
            # BT
            alm1all[:,10] = hp.almxfl(blm,flb)
            alm2all[:,10] = hp.almxfl(tlm_mv,invDl3)
            # B2T1
            alm1all[:,11] = hp.almxfl(blm,flb)
            alm2all[:,11] = hp.almxfl(tlm_mv,invDl3)
            # EB
            alm1all[:,12] = hp.almxfl(elm,invDl3)
            alm2all[:,12] = hp.almxfl(blm,flb)
            # E2B1
            alm1all[:,13] = hp.almxfl(elm,invDl3)
            alm2all[:,13] = hp.almxfl(blm,flb)
            # BE
            alm1all[:,14] = hp.almxfl(blm,flb)
            alm2all[:,14] = hp.almxfl(elm,invDl3)
            # B2E1
            alm1all[:,15] = hp.almxfl(blm,flb)
            alm2all[:,15] = hp.almxfl(elm,invDl3)

    elif append == 'standard':
        # SQE
        print('Creating filters...')
        # Create 1/cl filters
        flt = np.zeros(lmax+1); flt[lmin:] = 1./cltt[lmin:] # MV
        fle = np.zeros(lmax+1); fle[lmin:] = 1./clee[lmin:]
        flb = np.zeros(lmax+1); flb[lmin:] = 1./clbb[lmin:]

        if qe[0] == 'T': almbar1 = hp.almxfl(tlm_mv,flt); flm1 = flt
        if qe[0] == 'E': almbar1 = hp.almxfl(elm,fle); flm1 = fle
        if qe[0] == 'B': almbar1 = hp.almxfl(blm,flb); flm1 = flb

        if qe[1] == 'T': almbar2 = hp.almxfl(tlm_mv,flt); flm2 = flt
        if qe[1] == 'E': almbar2 = hp.almxfl(elm,fle); flm2 = fle
        if qe[1] == 'B': almbar2 = hp.almxfl(blm,flb); flm2 = flb

    elif append == 'mh':
        # SQE
        # Create 1/cl filters
        flt1 = np.zeros(lmax+1); flt1[lmin:] = 1./cltt1[lmin:] # MV
        flt2 = np.zeros(lmax+1); flt2[lmin:] = 1./cltt2[lmin:] # tSZ-null
        fle = np.zeros(lmax+1); fle[lmin:] = 1./clee[lmin:]
        flb = np.zeros(lmax+1); flb[lmin:] = 1./clbb[lmin:]

        if T3_opt == 'noT3':
            if qe == 'T1T2':
                almbar1 = hp.almxfl(tlm_mv,flt1); flm1 = flt1
                almbar2 = hp.almxfl(tlm_tszn,flt2); flm2 = flt2
            elif qe == 'T2T1':
                almbar1 = hp.almxfl(tlm_tszn,flt2); flm1 = flt2
                almbar2 = hp.almxfl(tlm_mv,flt1); flm2 = flt1
            elif qe == 'T2E1':
                almbar1 = hp.almxfl(tlm_tszn,flt2); flm1 = flt2
                almbar2 = hp.almxfl(elm,fle); flm2 = fle
            elif qe == 'E2T1':
                almbar1 = hp.almxfl(elm,fle); flm1 = fle
                almbar2 = hp.almxfl(tlm_mv,flt1); flm2 = flt1
            elif qe == 'E2E1':
                almbar1 = hp.almxfl(elm,fle); flm1 = fle
                almbar2 = hp.almxfl(elm,fle); flm2 = fle
            elif qe == 'T2B1':
                almbar1 = hp.almxfl(tlm_tszn,flt2); flm1 = flt2
                almbar2 = hp.almxfl(blm,flb); flm2 = flb
            elif qe == 'B2T1':
                almbar1 = hp.almxfl(blm,flb); flm1 = flb
                almbar2 = hp.almxfl(tlm_mv,flt1); flm2 = flt1
            elif qe == 'E2B1':
                almbar1 = hp.almxfl(elm,fle); flm1 = fle
                almbar2 = hp.almxfl(blm,flb); flm2 = flb
            elif qe == 'B2E1':
                almbar1 = hp.almxfl(blm,flb); flm1 = flb
                almbar2 = hp.almxfl(elm,fle); flm2 = fle
            else:
                if qe[0] == 'T': almbar1 = hp.almxfl(tlm_mv,flt1); flm1 = flt1
                elif qe[0] == 'E': almbar1 = hp.almxfl(elm,fle); flm1 = fle
                elif qe[0] == 'B': almbar1 = hp.almxfl(blm,flb); flm1 = flb

                if qe[1] == 'T': almbar2 = hp.almxfl(tlm_tszn,flt2); flm2 = flt2
                elif qe[1] == 'E': almbar2 = hp.almxfl(elm,fle); flm2 = fle
                elif qe[1] == 'B': almbar2 = hp.almxfl(blm,flb); flm2 = flb
        elif T3_opt == 'withT3':
            if qe == 'T1T2':
                almbar1 = hp.almxfl(tlm_mv,flt1); flm1 = flt1
                almbar2 = hp.almxfl(tlm_tszn,flt2); flm2 = flt2
            elif qe == 'T2T1':
                almbar1 = hp.almxfl(tlm_tszn,flt2); flm1 = flt2
                almbar2 = hp.almxfl(tlm_mv,flt1); flm2 = flt1
            elif qe == 'T2E1':
                almbar1 = hp.almxfl(tlm_mv,flt1); flm1 = flt1
                almbar2 = hp.almxfl(elm,fle); flm2 = fle
            elif qe == 'E2T1':
                almbar1 = hp.almxfl(elm,fle); flm1 = fle
                almbar2 = hp.almxfl(tlm_mv,flt1); flm2 = flt1
            elif qe == 'E2E1':
                almbar1 = hp.almxfl(elm,fle); flm1 = fle
                almbar2 = hp.almxfl(elm,fle); flm2 = fle
            elif qe == 'T2B1':
                almbar1 = hp.almxfl(tlm_mv,flt1); flm1 = flt1
                almbar2 = hp.almxfl(blm,flb); flm2 = flb
            elif qe == 'B2T1':
                almbar1 = hp.almxfl(blm,flb); flm1 = flb
                almbar2 = hp.almxfl(tlm_mv,flt1); flm2 = flt1
            elif qe == 'E2B1':
                almbar1 = hp.almxfl(elm,fle); flm1 = fle
                almbar2 = hp.almxfl(blm,flb); flm2 = flb
            elif qe == 'B2E1':
                almbar1 = hp.almxfl(blm,flb); flm1 = flb
                almbar2 = hp.almxfl(elm,fle); flm2 = fle
            else:
                if qe[0] == 'T': almbar1 = hp.almxfl(tlm_mv,flt1); flm1 = flt1
                elif qe[0] == 'E': almbar1 = hp.almxfl(elm,fle); flm1 = fle
                elif qe[0] == 'B': almbar1 = hp.almxfl(blm,flb); flm1 = flb

                if qe[1] == 'T': almbar2 = hp.almxfl(tlm_mv,flt1); flm2 = flt1
                elif qe[1] == 'E': almbar2 = hp.almxfl(elm,fle); flm2 = fle
                elif qe[1] == 'B': almbar2 = hp.almxfl(blm,flb); flm2 = flb

    elif append == 'crossilc_onesed' or append == 'crossilc_twoseds':
        # SQE
        # Create 1/cl filters
        flt1 = np.zeros(lmax+1); flt1[lmin:] = 1./cltt1[lmin:] # CIB-null
        flt2 = np.zeros(lmax+1); flt2[lmin:] = 1./cltt2[lmin:] # tSZ-null
        flt3 = np.zeros(lmax+1); flt3[lmin:] = 1./cltt3[lmin:] # MV
        fle = np.zeros(lmax+1); fle[lmin:] = 1./clee[lmin:]
        flb = np.zeros(lmax+1); flb[lmin:] = 1./clbb[lmin:]

        if T3_opt == 'noT3':
            if qe == 'T1T2':
                almbar1 = hp.almxfl(tlm_cibn,flt1); flm1 = flt1
                almbar2 = hp.almxfl(tlm_tszn,flt2); flm2 = flt2
            elif qe == 'T2T1':
                almbar1 = hp.almxfl(tlm_tszn,flt2); flm1 = flt2
                almbar2 = hp.almxfl(tlm_cibn,flt1); flm2 = flt1
            elif qe == 'T2E1':
                almbar1 = hp.almxfl(tlm_tszn,flt2); flm1 = flt2
                almbar2 = hp.almxfl(elm,fle); flm2 = fle
            elif qe == 'E2T1':
                almbar1 = hp.almxfl(elm,fle); flm1 = fle
                almbar2 = hp.almxfl(tlm_cibn,flt1); flm2 = flt1
            elif qe == 'E2E1':
                almbar1 = hp.almxfl(elm,fle); flm1 = fle
                almbar2 = hp.almxfl(elm,fle); flm2 = fle
            elif qe == 'T2B1':
                almbar1 = hp.almxfl(tlm_tszn,flt2); flm1 = flt2
                almbar2 = hp.almxfl(blm,flb); flm2 = flb
            elif qe == 'B2T1':
                almbar1 = hp.almxfl(blm,flb); flm1 = flb
                almbar2 = hp.almxfl(tlm_cibn,flt1); flm2 = flt1
            elif qe == 'E2B1':
                almbar1 = hp.almxfl(elm,fle); flm1 = fle
                almbar2 = hp.almxfl(blm,flb); flm2 = flb
            elif qe == 'B2E1':
                almbar1 = hp.almxfl(blm,flb); flm1 = flb
                almbar2 = hp.almxfl(elm,fle); flm2 = fle
            else:
                if qe[0] == 'T': almbar1 = hp.almxfl(tlm_cibn,flt1); flm1 = flt1
                elif qe[0] == 'E': almbar1 = hp.almxfl(elm,fle); flm1 = fle
                elif qe[0] == 'B': almbar1 = hp.almxfl(blm,flb); flm1 = flb

                if qe[1] == 'T': almbar2 = hp.almxfl(tlm_tszn,flt2); flm2 = flt2
                elif qe[1] == 'E': almbar2 = hp.almxfl(elm,fle); flm2 = fle
                elif qe[1] == 'B': almbar2 = hp.almxfl(blm,flb); flm2 = flb
        elif T3_opt == 'withT3':
            if qe == 'T1T2':
                almbar1 = hp.almxfl(tlm_cibn,flt1); flm1 = flt1
                almbar2 = hp.almxfl(tlm_tszn,flt2); flm2 = flt2
            elif qe == 'T2T1':
                almbar1 = hp.almxfl(tlm_tszn,flt2); flm1 = flt2
                almbar2 = hp.almxfl(tlm_cibn,flt1); flm2 = flt1
            elif qe == 'T2E1':
                almbar1 = hp.almxfl(tlm_mv,flt3); flm1 = flt3
                almbar2 = hp.almxfl(elm,fle); flm2 = fle
            elif qe == 'E2T1':
                almbar1 = hp.almxfl(elm,fle); flm1 = fle
                almbar2 = hp.almxfl(tlm_mv,flt3); flm2 = flt3
            elif qe == 'E2E1':
                almbar1 = hp.almxfl(elm,fle); flm1 = fle
                almbar2 = hp.almxfl(elm,fle); flm2 = fle
            elif qe == 'T2B1':
                almbar1 = hp.almxfl(tlm_mv,flt3); flm1 = flt3
                almbar2 = hp.almxfl(blm,flb); flm2 = flb
            elif qe == 'B2T1':
                almbar1 = hp.almxfl(blm,flb); flm1 = flb
                almbar2 = hp.almxfl(tlm_mv,flt1); flm2 = flt3
            elif qe == 'E2B1':
                almbar1 = hp.almxfl(elm,fle); flm1 = fle
                almbar2 = hp.almxfl(blm,flb); flm2 = flb
            elif qe == 'B2E1':
                almbar1 = hp.almxfl(blm,flb); flm1 = flb
                almbar2 = hp.almxfl(elm,fle); flm2 = fle
            else:
                if qe[0] == 'T': almbar1 = hp.almxfl(tlm_mv,flt3); flm1 = flt3
                elif qe[0] == 'E': almbar1 = hp.almxfl(elm,fle); flm1 = fle
                elif qe[0] == 'B': almbar1 = hp.almxfl(blm,flb); flm1 = flb

                if qe[1] == 'T': almbar2 = hp.almxfl(tlm_mv,flt3); flm2 = flt3
                elif qe[1] == 'E': almbar2 = hp.almxfl(elm,fle); flm2 = fle
                elif qe[1] == 'B': almbar2 = hp.almxfl(blm,flb); flm2 = flb

    # Run healqest
    print('Running healqest...')
    if qe == 'T1T2' or qe == 'T2T1': qe='TT'
    if qe == 'T2E1': qe='TE'
    if qe == 'E2T1': qe='ET'
    if qe == 'E2E1': qe='EE'
    if qe == 'T2B1': qe='TB'
    if qe == 'B2T1': qe='BT'
    if qe == 'E2B1': qe='EB'
    if qe == 'B2E1': qe='BE'
    if gmv and not cinv:
        q_gmv = qest.qest_gmv(config,cls)
        if append == 'standard':
            glm,clm = q_gmv.eval(qe,alm1all,alm2all,totalcls,crossilc=False)
        elif append == 'mh' or append == 'crossilc_onesed' or append == 'crossilc_twoseds':
            # TODO: withT3 = False is hard-coded in weights
            glm,clm = q_gmv.eval(qe,alm1all,alm2all,totalcls,crossilc=True)
    else:
        q = qest.qest(config,cls)
        glm,clm = q.eval(qe,almbar1,almbar2)
    # Save plm
    Path(dir_out).mkdir(parents=True, exist_ok=True)
    np.save(filename,glm)
    return

if __name__ == '__main__':

    main()
