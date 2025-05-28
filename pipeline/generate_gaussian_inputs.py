import os, sys
import numpy as np
import healpy as hp
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import healqest_utils as utils

model_type = 'websky'

dir_out = '/oak/stanford/orgs/kipac/users/yukanaka/outputs/'
lmax = 4096

if model_type == 'websky':
    # Websky sims
    websky_095_T = '/oak/stanford/orgs/kipac/users/yukanaka/websky/websky_spt_95ghz_lcmb_tsz_cib_ksz_kszpatchy_muk_alm_lmax4096.fits'
    websky_150_T = '/oak/stanford/orgs/kipac/users/yukanaka/websky/websky_spt_150ghz_lcmb_tsz_cib_ksz_kszpatchy_muk_alm_lmax4096.fits'
    websky_220_T = '/oak/stanford/orgs/kipac/users/yukanaka/websky/websky_spt_220ghz_lcmb_tsz_cib_ksz_kszpatchy_muk_alm_lmax4096.fits'
    # From https://lambda.gsfc.nasa.gov/simulation/mocks_data.html but they claim it's "T,Q,U alms", but I think they mean T,E,B...
    websky_nofg = '/oak/stanford/orgs/kipac/users/yukanaka/websky/lensed_alm.fits'
    # ILC weights
    # These are dimensions (4097, 3) initially; then transpose to make it (3, 4097)
    w_tsz_null = np.load('/oak/stanford/orgs/kipac/users/yukanaka/websky/weights_websky_cmbrec_tsznull_lmax4096.npy').T
    w_Tmv = np.load('/oak/stanford/orgs/kipac/users/yukanaka/websky/weights_websky_cmbrec_mv_lmax4096.npy').T
    w_cib_null = np.load('/oak/stanford/orgs/kipac/users/yukanaka/websky/weights_websky_cmbrec_cibnull_lmax4096.npy').T
    # Dimension (3, 6001) for 90, 150, 220 GHz respectively
    w_Emv = np.loadtxt('ilc_weights/weights1d_EE_spt3g_cmbmv.dat')
    w_Bmv = np.loadtxt('ilc_weights/weights1d_BB_spt3g_cmbmv.dat')

    # Get Websky sim (signal + foregrounds)
    print('Getting alms...')
    tlm_95_websky = hp.read_alm(websky_095_T)
    tlm_150_websky = hp.read_alm(websky_150_T)
    tlm_220_websky = hp.read_alm(websky_220_T)
    tlm_lcmb_95_websky, elm_95_websky, blm_95_websky = hp.read_alm(websky_nofg,hdu=[1,2,3])
    tlm_lcmb_150_websky, elm_150_websky, blm_150_websky = hp.read_alm(websky_nofg,hdu=[1,2,3])
    tlm_lcmb_220_websky, elm_220_websky, blm_220_websky = hp.read_alm(websky_nofg,hdu=[1,2,3])
    tlm_lcmb_95_websky = utils.reduce_lmax(tlm_lcmb_95_websky,lmax=lmax); tlm_lcmb_150_websky = utils.reduce_lmax(tlm_lcmb_150_websky,lmax=lmax); tlm_lcmb_220_websky = utils.reduce_lmax(tlm_lcmb_220_websky,lmax=lmax);
    tlm_95_websky = utils.reduce_lmax(tlm_95_websky,lmax=lmax); tlm_150_websky = utils.reduce_lmax(tlm_150_websky,lmax=lmax); tlm_220_websky = utils.reduce_lmax(tlm_220_websky,lmax=lmax);
    elm_95_websky = utils.reduce_lmax(elm_95_websky,lmax=lmax); elm_150_websky = utils.reduce_lmax(elm_150_websky,lmax=lmax); elm_220_websky = utils.reduce_lmax(elm_220_websky,lmax=lmax);
    blm_95_websky = utils.reduce_lmax(blm_95_websky,lmax=lmax); blm_150_websky = utils.reduce_lmax(blm_150_websky,lmax=lmax); blm_220_websky = utils.reduce_lmax(blm_220_websky,lmax=lmax);
    # Get Websky foreground-only
    tlm_fg_95 = tlm_95_websky - tlm_lcmb_95_websky
    tlm_fg_150 = tlm_150_websky - tlm_lcmb_150_websky
    tlm_fg_220 = tlm_220_websky - tlm_lcmb_220_websky
    # ILC combine frequencies
    tlm_mv = hp.almxfl(tlm_fg_95,w_Tmv[0][:lmax+1]) + hp.almxfl(tlm_fg_150,w_Tmv[1][:lmax+1]) + hp.almxfl(tlm_fg_220,w_Tmv[2][:lmax+1])
    tlm_tszn = hp.almxfl(tlm_fg_95,w_tsz_null[0][:lmax+1]) + hp.almxfl(tlm_fg_150,w_tsz_null[1][:lmax+1]) + hp.almxfl(tlm_fg_220,w_tsz_null[2][:lmax+1])
    tlm_cibn = hp.almxfl(tlm_fg_95,w_cib_null[0][:lmax+1]) + hp.almxfl(tlm_fg_150,w_cib_null[1][:lmax+1]) + hp.almxfl(tlm_fg_220,w_cib_null[2][:lmax+1])
    # WEBSKY FG
    fltt_mv_websky = hp.alm2cl(tlm_mv,tlm_mv)
    fltt_tszn_websky = hp.alm2cl(tlm_tszn,tlm_tszn)
    fltt_cibn_onesed_websky = hp.alm2cl(tlm_cibn,tlm_cibn)
    # Synfast
    for sim in np.arange(161,250)+1:
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
    # Agora sims (TOTAL, CMB + foregrounds)
    agora_095 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_95ghz_alm_lmax4096.fits'
    agora_150 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_150ghz_alm_lmax4096.fits'
    agora_220 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_220ghz_alm_lmax4096.fits'
    # Lensed CMB-only Agora sims
    agora_lcmb_095 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lcmb_spt3g_95ghz_alm_lmax4096.fits'
    agora_lcmb_150 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lcmb_spt3g_150ghz_alm_lmax4096.fits'
    agora_lcmb_220 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lcmb_spt3g_220ghz_alm_lmax4096.fits'
    # ILC weights
    # Dimension (3, 6001) for 90, 150, 220 GHz respectively
    agora_w_tsz_null = np.loadtxt('ilc_weights/weights1d_TT_spt3g_cmbynull.dat')
    agora_w_Tmv = np.loadtxt('ilc_weights/weights1d_TT_spt3g_cmbmv.dat')
    agora_w_Emv = np.loadtxt('ilc_weights/weights1d_EE_spt3g_cmbmv.dat')
    agora_w_Bmv = np.loadtxt('ilc_weights/weights1d_BB_spt3g_cmbmv.dat')
    agora_w_cib_null_onesed = np.load('ilc_weights/weights_cmb_mv_cmbfree_cibfree_spt3g1920.npy',allow_pickle=True)
    agora_w_cib_null_onesed_95 = agora_w_cib_null_onesed.item()['cmbcibfree'][95][1]
    agora_w_cib_null_onesed_150 = agora_w_cib_null_onesed.item()['cmbcibfree'][150][1]
    agora_w_cib_null_onesed_220 = agora_w_cib_null_onesed.item()['cmbcibfree'][220][1]
    agora_w_cib_null_twoseds = np.load('ilc_weights/weights_cmb_mv_cmbfree_cibfreetwoSEDs_spt3g1920.npy',allow_pickle=True)
    agora_w_cib_null_twoseds_95 = agora_w_cib_null_twoseds.item()['cmbcibfree'][95][1]
    agora_w_cib_null_twoseds_150 = agora_w_cib_null_twoseds.item()['cmbcibfree'][150][1]
    agora_w_cib_null_twoseds_220 = agora_w_cib_null_twoseds.item()['cmbcibfree'][220][1]

    # Get Agora sim (signal + foregrounds)
    print('Getting alms...')
    tlm_95, elm_95, blm_95 = hp.read_alm(agora_095,hdu=[1,2,3])
    tlm_150, elm_150, blm_150 = hp.read_alm(agora_150,hdu=[1,2,3])
    tlm_220, elm_220, blm_220 = hp.read_alm(agora_220,hdu=[1,2,3])
    # Get Agora lensed CMB-only
    tlm_lcmb_95, elm_lcmb_95, blm_lcmb_95 = hp.read_alm(agora_lcmb_095,hdu=[1,2,3])
    tlm_lcmb_150, elm_lcmb_150, blm_lcmb_150 = hp.read_alm(agora_lcmb_150,hdu=[1,2,3])
    tlm_lcmb_220, elm_lcmb_220, blm_lcmb_220 = hp.read_alm(agora_lcmb_220,hdu=[1,2,3])
    # Get Agora foreground-only
    tlm_fg_95 = tlm_95 - tlm_lcmb_95; elm_fg_95 = elm_95 - elm_lcmb_95; blm_fg_95 = blm_95 - blm_lcmb_95
    tlm_fg_150 = tlm_150 - tlm_lcmb_150; elm_fg_150 = elm_150 - elm_lcmb_150; blm_fg_150 = blm_150 - blm_lcmb_150
    tlm_fg_220 = tlm_220 - tlm_lcmb_220; elm_fg_220 = elm_220 - elm_lcmb_220; blm_fg_220 = blm_220 - blm_lcmb_220
    # ILC combine frequencies
    tlm_mv = hp.almxfl(tlm_fg_95,agora_w_Tmv[0][:lmax+1]) + hp.almxfl(tlm_fg_150,agora_w_Tmv[1][:lmax+1]) + hp.almxfl(tlm_fg_220,agora_w_Tmv[2][:lmax+1])
    tlm_tszn = hp.almxfl(tlm_fg_95,agora_w_tsz_null[0][:lmax+1]) + hp.almxfl(tlm_fg_150,agora_w_tsz_null[1][:lmax+1]) + hp.almxfl(tlm_fg_220,agora_w_tsz_null[2][:lmax+1])
    tlm_cibn_onesed = hp.almxfl(tlm_fg_95,agora_w_cib_null_onesed_95[:lmax+1]) + hp.almxfl(tlm_fg_150,agora_w_cib_null_onesed_150[:lmax+1]) + hp.almxfl(tlm_fg_220,agora_w_cib_null_onesed_220[:lmax+1])
    tlm_cibn_twoseds = hp.almxfl(tlm_fg_95,agora_w_cib_null_twoseds_95[:lmax+1]) + hp.almxfl(tlm_fg_150,agora_w_cib_null_twoseds_150[:lmax+1]) + hp.almxfl(tlm_fg_220,agora_w_cib_null_twoseds_220[:lmax+1])
    elm = hp.almxfl(elm_fg_95,agora_w_Emv[0][:lmax+1]) + hp.almxfl(elm_fg_150,agora_w_Emv[1][:lmax+1]) + hp.almxfl(elm_fg_220,agora_w_Emv[2][:lmax+1])
    blm = hp.almxfl(blm_fg_95,agora_w_Bmv[0][:lmax+1]) + hp.almxfl(blm_fg_150,agora_w_Bmv[1][:lmax+1]) + hp.almxfl(blm_fg_220,agora_w_Bmv[2][:lmax+1])
    # AGORA FG
    fltt_mv_agora = hp.alm2cl(tlm_mv,tlm_mv)
    fltt_tszn_agora = hp.alm2cl(tlm_tszn,tlm_tszn)
    fltt_cibn_onesed_agora = hp.alm2cl(tlm_cibn_onesed,tlm_cibn_onesed)
    fltt_cibn_twoseds_agora = hp.alm2cl(tlm_cibn_twoseds,tlm_cibn_twoseds)
    flee_mv_agora = hp.alm2cl(elm,elm)
    flbb_mv_agora = hp.alm2cl(blm,blm)
    flte_mv_agora = hp.alm2cl(tlm_mv,elm)
    flte_tszn_agora = hp.alm2cl(tlm_tszn,elm)
    flte_cibn_onesed_agora = hp.alm2cl(tlm_cibn_onesed,elm)
    flte_cibn_twoseds_agora = hp.alm2cl(tlm_cibn_twoseds,elm)
    # Synfast
    for sim in np.arange(127,250)+1:
        np.random.seed(sim)
        tlm_mv_out,elm_mv_out,blm_mv_out = hp.synalm([fltt_mv_agora,flee_mv_agora,flbb_mv_agora,flte_mv_agora],new=True,lmax=lmax)
        tlm_tszn_out,_,_ = hp.synalm([fltt_tszn_agora,flee_mv_agora,flbb_mv_agora,flte_tszn_agora],new=True,lmax=lmax)
        tlm_cibn_onesed_out,_,_ = hp.synalm([fltt_cibn_onesed_agora,flee_mv_agora,flbb_mv_agora,flte_cibn_onesed_agora],new=True,lmax=lmax)
        tlm_cibn_twoseds_out,_,_ = hp.synalm([fltt_cibn_twoseds_agora,flee_mv_agora,flbb_mv_agora,flte_cibn_twoseds_agora],new=True,lmax=lmax)

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
        nlmt_cibn_onesed = hp.almxfl(nlmt_090,agora_w_cib_null_onesed_95[:lmax+1]) + hp.almxfl(nlmt_150,agora_w_cib_null_onesed_150[:lmax+1]) + hp.almxfl(nlmt_220,agora_w_cib_null_onesed_220[:lmax+1])
        nlmt_cibn_twoseds = hp.almxfl(nlmt_090,agora_w_cib_null_twoseds_95[:lmax+1]) + hp.almxfl(nlmt_150,agora_w_cib_null_twoseds_150[:lmax+1]) + hp.almxfl(nlmt_220,agora_w_cib_null_twoseds_220[:lmax+1])
        nlme_mv = hp.almxfl(nlme_090,agora_w_Emv[0][:lmax+1]) + hp.almxfl(nlme_150,agora_w_Emv[1][:lmax+1]) + hp.almxfl(nlme_220,agora_w_Emv[2][:lmax+1])
        nlmb_mv = hp.almxfl(nlmb_090,agora_w_Bmv[0][:lmax+1]) + hp.almxfl(nlmb_150,agora_w_Bmv[1][:lmax+1]) + hp.almxfl(nlmb_220,agora_w_Bmv[2][:lmax+1])

        # Add noise
        tlm_mv_out += nlmt_mv; tlm_tszn_out += nlmt_tszn; tlm_cibn_onesed_out += nlmt_cibn_onesed; tlm_cibn_twoseds_out += nlmt_cibn_twoseds
        elm_mv_out += nlme_mv;
        blm_mv_out += nlmb_mv;

        # Save
        hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/gmv/inputs/spt3g_2019_2020_agora/agora_fg_plus_spt3g_20192020_noise_lmax{lmax}_seed{sim}_mv.fits',[tlm_mv_out,elm_mv_out,blm_mv_out])
        hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/gmv/inputs/spt3g_2019_2020_agora/agora_fg_plus_spt3g_20192020_noise_lmax{lmax}_seed{sim}_tszn.fits',[tlm_tszn_out,elm_mv_out,blm_mv_out])
        hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/gmv/inputs/spt3g_2019_2020_agora/agora_fg_plus_spt3g_20192020_noise_lmax{lmax}_seed{sim}_cibn_onesed.fits',[tlm_cibn_onesed_out,elm_mv_out,blm_mv_out])
        hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/gmv/inputs/spt3g_2019_2020_agora/agora_fg_plus_spt3g_20192020_noise_lmax{lmax}_seed{sim}_cibn_twoseds.fits',[tlm_cibn_twoseds_out,elm_mv_out,blm_mv_out])

