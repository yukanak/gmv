import os, sys
import numpy as np
import healpy as hp
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import healqest_utils as utils

model_type = 'websky'
#model_type = 'agora'

dir_out = '/oak/stanford/orgs/kipac/users/yukanaka/outputs/'
lmax = 4096

if model_type == 'websky':
    # Websky sims, foregrounds-only
    websky_095_T = '/oak/stanford/orgs/kipac/users/yukanaka/websky/websky_95ghz_tsz_cib_ksz_ksz-patchy_muK_nside4096.fits'
    websky_150_T = '/oak/stanford/orgs/kipac/users/yukanaka/websky/websky_150ghz_tsz_cib_ksz_ksz-patchy_muK_nside4096.fits'
    websky_220_T = '/oak/stanford/orgs/kipac/users/yukanaka/websky/websky_220ghz_tsz_cib_ksz_ksz-patchy_muK_nside4096.fits'
    
    # ILC weights
    w = np.load('/home/users/yukanaka/gmv/pipeline/ilc_weights/ilc_weights_cmb_spt3g_2yr_websky.npy',allow_pickle=True).item()
    cases = ['mv', 'tsznull', 'cibnull']
    # Dimension (3, 4097) for 90, 150, 220 GHz respectively
    w_Tmv = w['tt']['mv'];
    w_Emv = w['ee']['mv'];
    w_Bmv = w['bb']['mv'];
    w_tsz_null = w['tt']['tsznull']
    w_cib_null = w['tt']['cibnull']

    print('Getting alms...')
    # Get WebSky foreground-only
    tlm_fg_95 = hp.read_map(websky_095_T); tlm_fg_95 = hp.map2alm(tlm_fg_95);
    tlm_fg_150 = hp.read_map(websky_150_T); tlm_fg_150 = hp.map2alm(tlm_fg_150);
    tlm_fg_220 = hp.read_map(websky_220_T); tlm_fg_220 = hp.map2alm(tlm_fg_220);
    # ILC combine frequencies
    tlm_mv = hp.almxfl(tlm_fg_95,w_Tmv[0][:lmax+1]) + hp.almxfl(tlm_fg_150,w_Tmv[1][:lmax+1]) + hp.almxfl(tlm_fg_220,w_Tmv[2][:lmax+1])
    tlm_tszn = hp.almxfl(tlm_fg_95,w_tsz_null[0][:lmax+1]) + hp.almxfl(tlm_fg_150,w_tsz_null[1][:lmax+1]) + hp.almxfl(tlm_fg_220,w_tsz_null[2][:lmax+1])
    tlm_cibn = hp.almxfl(tlm_fg_95,w_cib_null[0][:lmax+1]) + hp.almxfl(tlm_fg_150,w_cib_null[1][:lmax+1]) + hp.almxfl(tlm_fg_220,w_cib_null[2][:lmax+1])
    # WEBSKY FG
    fltt_mv_websky = hp.alm2cl(tlm_mv,tlm_mv)
    fltt_tszn_websky = hp.alm2cl(tlm_tszn,tlm_tszn)
    fltt_cibn_onesed_websky = hp.alm2cl(tlm_cibn,tlm_cibn)
    # Synfast
    for sim in np.arange(250)+1:
        np.random.seed(sim)
        tlm_mv_out,_,_ = hp.synalm([fltt_mv_websky,fltt_mv_websky*0,fltt_mv_websky*0,fltt_mv_websky*0],new=True,lmax=lmax)
        tlm_tszn_out,_,_ = hp.synalm([fltt_tszn_websky,fltt_tszn_websky*0,fltt_tszn_websky*0,fltt_tszn_websky*0],new=True,lmax=lmax)
        tlm_cibn_onesed_out,_,_ = hp.synalm([fltt_cibn_onesed_websky,fltt_cibn_onesed_websky*0,fltt_cibn_onesed_websky*0,fltt_cibn_onesed_websky*0],new=True,lmax=lmax)

        # Load noise
        nlm_090_filename = dir_out + f'nlm/nlm_090_lmax{lmax}_seed{sim}.alm'
        nlm_150_filename = dir_out + f'nlm/nlm_150_lmax{lmax}_seed{sim}.alm'
        nlm_220_filename = dir_out + f'nlm/nlm_220_lmax{lmax}_seed{sim}.alm'
        nlmt_090,nlme_090,nlmb_090 = hp.read_alm(nlm_090_filename,hdu=[1,2,3])
        nlmt_150,nlme_150,nlmb_150 = hp.read_alm(nlm_150_filename,hdu=[1,2,3])
        nlmt_220,nlme_220,nlmb_220 = hp.read_alm(nlm_220_filename,hdu=[1,2,3])
        # ILC combine frequencies
        nlmt_mv = hp.almxfl(nlmt_090,w_Tmv[0][:lmax+1]) + hp.almxfl(nlmt_150,w_Tmv[1][:lmax+1]) + hp.almxfl(nlmt_220,w_Tmv[2][:lmax+1])
        nlmt_tszn = hp.almxfl(nlmt_090,w_tsz_null[0][:lmax+1]) + hp.almxfl(nlmt_150,w_tsz_null[1][:lmax+1]) + hp.almxfl(nlmt_220,w_tsz_null[2][:lmax+1])
        nlmt_cibn_onesed = hp.almxfl(nlmt_090,w_cib_null[0][:lmax+1]) + hp.almxfl(nlmt_150,w_cib_null[0][:lmax+1]) + hp.almxfl(nlmt_220,w_cib_null[0][:lmax+1])
        nlme_mv = hp.almxfl(nlme_090,w_Emv[0][:lmax+1]) + hp.almxfl(nlme_150,w_Emv[1][:lmax+1]) + hp.almxfl(nlme_220,w_Emv[2][:lmax+1])
        nlmb_mv = hp.almxfl(nlmb_090,w_Bmv[0][:lmax+1]) + hp.almxfl(nlmb_150,w_Bmv[1][:lmax+1]) + hp.almxfl(nlmb_220,w_Bmv[2][:lmax+1])

        # Add noise
        tlm_mv_out += nlmt_mv; tlm_tszn_out += nlmt_tszn; tlm_cibn_onesed_out += nlmt_cibn_onesed;
        elm_mv_out = nlme_mv;
        blm_mv_out = nlmb_mv;

        # Save
        hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/gmv/inputs/spt3g_2019_2020_websky/websky_fg_plus_spt3g_20192020_noise_lmax{lmax}_seed{sim}_mv.fits',[tlm_mv_out,elm_mv_out,blm_mv_out])
        hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/gmv/inputs/spt3g_2019_2020_websky/websky_fg_plus_spt3g_20192020_noise_lmax{lmax}_seed{sim}_tszn.fits',[tlm_tszn_out,elm_mv_out,blm_mv_out])
        hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/gmv/inputs/spt3g_2019_2020_websky/websky_fg_plus_spt3g_20192020_noise_lmax{lmax}_seed{sim}_cibn_onesed.fits',[tlm_cibn_onesed_out,elm_mv_out,blm_mv_out])

elif model_type == 'agora':
    # Point source masked Agora foregrounds-only
    agora_095 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lcib_lksz_lrad_ltsz_spt3g_95ghz_alm_lmax4096.fits'
    agora_150 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lcib_lksz_lrad_ltsz_spt3g_150ghz_alm_lmax4096.fits'
    agora_220 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lcib_lksz_lrad_ltsz_spt3g_220ghz_alm_lmax4096.fits'
    # ILC weights
    w_agora = np.load('/home/users/yukanaka/gmv/pipeline/ilc_weights/ilc_weights_cmb_spt3g_2yr.npy',allow_pickle=True).item()
    cases = ['mv', 'tsznull', 'cibnull']
    # Dimension (3, 4097) for 90, 150, 220 GHz respectively
    agora_w_Tmv = w_agora['tt']['mv'];
    agora_w_Emv = w_agora['ee']['mv'];
    agora_w_Bmv = w_agora['bb']['mv'];
    agora_w_tsz_null = w_agora['tt']['tsznull']
    agora_w_cib_null_onesed = w_agora['tt']['cibnull']
    #agora_w_tsz_null = np.loadtxt('ilc_weights/weights1d_TT_spt3g_cmbynull.dat')
    #agora_w_Tmv = np.loadtxt('ilc_weights/weights1d_TT_spt3g_cmbmv.dat')
    #agora_w_Emv = np.loadtxt('ilc_weights/weights1d_EE_spt3g_cmbmv.dat')
    #agora_w_Bmv = np.loadtxt('ilc_weights/weights1d_BB_spt3g_cmbmv.dat')
    #agora_w_cib_null_onesed = np.load('ilc_weights/weights_cmb_mv_cmbfree_cibfree_spt3g1920.npy',allow_pickle=True)
    #agora_w_cib_null_onesed_95 = agora_w_cib_null_onesed.item()['cmbcibfree'][95][1]
    #agora_w_cib_null_onesed_150 = agora_w_cib_null_onesed.item()['cmbcibfree'][150][1]
    #agora_w_cib_null_onesed_220 = agora_w_cib_null_onesed.item()['cmbcibfree'][220][1]
    #agora_w_cib_null_twoseds = np.load('ilc_weights/weights_cmb_mv_cmbfree_cibfreetwoSEDs_spt3g1920.npy',allow_pickle=True)
    #agora_w_cib_null_twoseds_95 = agora_w_cib_null_twoseds.item()['cmbcibfree'][95][1]
    #agora_w_cib_null_twoseds_150 = agora_w_cib_null_twoseds.item()['cmbcibfree'][150][1]
    #agora_w_cib_null_twoseds_220 = agora_w_cib_null_twoseds.item()['cmbcibfree'][220][1]

    print('Getting alms...')
    # Get Agora foreground-only
    tlm_fg_95, elm_fg_95, blm_fg_95 = hp.read_alm(agora_095,hdu=[1,2,3])
    tlm_fg_150, elm_fg_150, blm_fg_150 = hp.read_alm(agora_150,hdu=[1,2,3])
    tlm_fg_220, elm_fg_220, blm_fg_220 = hp.read_alm(agora_220,hdu=[1,2,3])

    #TODO
    '''
    # Get Agora ONE NG FG
    #tlm_one_95 = hp.read_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_ltsz_spt3g_95ghz_alm_lmax{lmax}.fits')
    #tlm_one_150 = hp.read_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_ltsz_spt3g_150ghz_alm_lmax{lmax}.fits')
    #tlm_one_220 = hp.read_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_ltsz_spt3g_220ghz_alm_lmax{lmax}.fits')
    tlm_one_95, elm_one_95, blm_one_95 = hp.read_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_ltsz_lcib_spt3g_95ghz_alm_lmax{lmax}.fits',hdu=[1,2,3])
    tlm_one_150, elm_one_150, blm_one_150 = hp.read_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_ltsz_lcib_spt3g_150ghz_alm_lmax{lmax}.fits',hdu=[1,2,3])
    tlm_one_220, elm_one_220, blm_one_220 = hp.read_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_ltsz_lcib_spt3g_220ghz_alm_lmax{lmax}.fits',hdu=[1,2,3])
    # Get Agora foreground-only EXCLUDING THE ONE FG
    tlm_fg_95 -= tlm_one_95; elm_fg_95 -= elm_one_95; blm_fg_95 -= blm_one_95
    tlm_fg_150 -= tlm_one_150; elm_fg_150 -= elm_one_150; blm_fg_150 -= blm_one_150
    tlm_fg_220 -= tlm_one_220; elm_fg_220 -= elm_one_220; blm_fg_220 -= blm_one_220
    '''

    # ILC combine frequencies
    tlm_mv = hp.almxfl(tlm_fg_95,agora_w_Tmv[0][:lmax+1]) + hp.almxfl(tlm_fg_150,agora_w_Tmv[1][:lmax+1]) + hp.almxfl(tlm_fg_220,agora_w_Tmv[2][:lmax+1])
    tlm_tszn = hp.almxfl(tlm_fg_95,agora_w_tsz_null[0][:lmax+1]) + hp.almxfl(tlm_fg_150,agora_w_tsz_null[1][:lmax+1]) + hp.almxfl(tlm_fg_220,agora_w_tsz_null[2][:lmax+1])
    tlm_cibn_onesed = hp.almxfl(tlm_fg_95,agora_w_cib_null_onesed[0][:lmax+1]) + hp.almxfl(tlm_fg_150,agora_w_cib_null_onesed[1][:lmax+1]) + hp.almxfl(tlm_fg_220,agora_w_cib_null_onesed[2][:lmax+1])
    elm = hp.almxfl(elm_fg_95,agora_w_Emv[0][:lmax+1]) + hp.almxfl(elm_fg_150,agora_w_Emv[1][:lmax+1]) + hp.almxfl(elm_fg_220,agora_w_Emv[2][:lmax+1])
    blm = hp.almxfl(blm_fg_95,agora_w_Bmv[0][:lmax+1]) + hp.almxfl(blm_fg_150,agora_w_Bmv[1][:lmax+1]) + hp.almxfl(blm_fg_220,agora_w_Bmv[2][:lmax+1])
    # AGORA FG
    fltt_mv_agora = hp.alm2cl(tlm_mv,tlm_mv)
    fltt_tszn_agora = hp.alm2cl(tlm_tszn,tlm_tszn)
    fltt_cibn_onesed_agora = hp.alm2cl(tlm_cibn_onesed,tlm_cibn_onesed)
    flee_mv_agora = hp.alm2cl(elm,elm)
    flbb_mv_agora = hp.alm2cl(blm,blm)
    flte_mv_agora = hp.alm2cl(tlm_mv,elm)
    flte_tszn_agora = hp.alm2cl(tlm_tszn,elm)
    flte_cibn_onesed_agora = hp.alm2cl(tlm_cibn_onesed,elm)
    # Synfast
    for sim in np.arange(250)+1:
        #TODO
        #sim = 999
        np.random.seed(sim)
        tlm_mv_out,elm_mv_out,blm_mv_out = hp.synalm([fltt_mv_agora,flee_mv_agora,flbb_mv_agora,flte_mv_agora],new=True,lmax=lmax)
        tlm_tszn_out,_,_ = hp.synalm([fltt_tszn_agora,flee_mv_agora,flbb_mv_agora,flte_tszn_agora],new=True,lmax=lmax)
        tlm_cibn_onesed_out,_,_ = hp.synalm([fltt_cibn_onesed_agora,flee_mv_agora,flbb_mv_agora,flte_cibn_onesed_agora],new=True,lmax=lmax)

        # Load noise
        nlm_090_filename = dir_out + f'nlm/nlm_090_lmax{lmax}_seed{sim}.alm'
        nlm_150_filename = dir_out + f'nlm/nlm_150_lmax{lmax}_seed{sim}.alm'
        nlm_220_filename = dir_out + f'nlm/nlm_220_lmax{lmax}_seed{sim}.alm'
        nlmt_090,nlme_090,nlmb_090 = hp.read_alm(nlm_090_filename,hdu=[1,2,3])
        nlmt_150,nlme_150,nlmb_150 = hp.read_alm(nlm_150_filename,hdu=[1,2,3])
        nlmt_220,nlme_220,nlmb_220 = hp.read_alm(nlm_220_filename,hdu=[1,2,3])
        # ILC combine frequencies
        nlmt_mv = hp.almxfl(nlmt_090,agora_w_Tmv[0][:lmax+1]) + hp.almxfl(nlmt_150,agora_w_Tmv[1][:lmax+1]) + hp.almxfl(nlmt_220,agora_w_Tmv[2][:lmax+1])
        nlmt_tszn = hp.almxfl(nlmt_090,agora_w_tsz_null[0][:lmax+1]) + hp.almxfl(nlmt_150,agora_w_tsz_null[1][:lmax+1]) + hp.almxfl(nlmt_220,agora_w_tsz_null[2][:lmax+1])
        nlmt_cibn_onesed = hp.almxfl(nlmt_090,agora_w_cib_null_onesed[0][:lmax+1]) + hp.almxfl(nlmt_150,agora_w_cib_null_onesed[1][:lmax+1]) + hp.almxfl(nlmt_220,agora_w_cib_null_onesed[2][:lmax+1])
        nlme_mv = hp.almxfl(nlme_090,agora_w_Emv[0][:lmax+1]) + hp.almxfl(nlme_150,agora_w_Emv[1][:lmax+1]) + hp.almxfl(nlme_220,agora_w_Emv[2][:lmax+1])
        nlmb_mv = hp.almxfl(nlmb_090,agora_w_Bmv[0][:lmax+1]) + hp.almxfl(nlmb_150,agora_w_Bmv[1][:lmax+1]) + hp.almxfl(nlmb_220,agora_w_Bmv[2][:lmax+1])

        # Add noise
        tlm_mv_out += nlmt_mv; tlm_tszn_out += nlmt_tszn; tlm_cibn_onesed_out += nlmt_cibn_onesed
        elm_mv_out += nlme_mv;
        blm_mv_out += nlmb_mv;

        # Save
        hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/gmv/inputs/spt3g_2019_2020_agora/agora_fg_plus_spt3g_20192020_noise_lmax{lmax}_seed{sim}_mv.fits',[tlm_mv_out,elm_mv_out,blm_mv_out],overwrite=True)
        hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/gmv/inputs/spt3g_2019_2020_agora/agora_fg_plus_spt3g_20192020_noise_lmax{lmax}_seed{sim}_tszn.fits',[tlm_tszn_out,elm_mv_out,blm_mv_out],overwrite=True)
        hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/gmv/inputs/spt3g_2019_2020_agora/agora_fg_plus_spt3g_20192020_noise_lmax{lmax}_seed{sim}_cibn_onesed.fits',[tlm_cibn_onesed_out,elm_mv_out,blm_mv_out],overwrite=True)
        #TODO
        #hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/gmv/inputs/spt3g_2019_2020_agora/agora_fg_plus_spt3g_20192020_noise_noltszlcib_lmax{lmax}_seed{sim}_mv.fits',[tlm_mv_out,elm_mv_out,blm_mv_out],overwrite=True)
        #hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/gmv/inputs/spt3g_2019_2020_agora/agora_fg_plus_spt3g_20192020_noise_noltszlcib_lmax{lmax}_seed{sim}_tszn.fits',[tlm_tszn_out,elm_mv_out,blm_mv_out],overwrite=True)
        #hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/gmv/inputs/spt3g_2019_2020_agora/agora_fg_plus_spt3g_20192020_noise_noltszlcib_lmax{lmax}_seed{sim}_cibn_onesed.fits',[tlm_cibn_onesed_out,elm_mv_out,blm_mv_out],overwrite=True)

