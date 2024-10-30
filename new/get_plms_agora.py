#!/usr/bin/env python3
# Run like python3 get_plms.py TT 100 101 append test_yuka.yaml
# Note: argument append should be 'agora_standard', 'agora_mh', 'agora_crossilc_onesed', 'agora_crossilc_twoseds'
# For cinv-style, append should be 'agora_standard_cinv', etc.
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
    if len(sys.argv) > 4:
        T3_opt = str(sys.argv[4])
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

    if append[-4:] != 'cinv' and (qe == 'TT' or qe == 'TE' or  qe == 'ET' or qe == 'EE' or qe == 'TB' or  qe == 'BT' or qe == 'EB' or  qe == 'BE' or qe == 'TTprf' or qe == 'T1T2' or qe == 'T2T1' or qe == 'T2E1' or qe == 'E2T1' or qe == 'E2E1'):
        # SQE
        gmv = False
        if T3_opt == 'noT3' or T3_opt == 'withT3':
            filename = dir_out+f'/plm_{qe}_healqest_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_{T3_opt}.npy'
        else:
            filename = dir_out+f'/plm_{qe}_healqest_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy'
    else:
        # GMV
        gmv = True
        if T3_opt == 'noT3' or T3_opt == 'withT3':
            filename = dir_out+f'/plm_{qe}_healqest_gmv_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_{T3_opt}.npy'
        else:
            filename = dir_out+f'/plm_{qe}_healqest_gmv_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy'

    if append[-4:] == 'cinv' and gmv:
        cinv = True

    print(f'cinv is {cinv}, gmv is {gmv}, filename {filename}')

    if os.path.isfile(filename):
        print('File already exists!')
    else:
        do_reconstruction(qe,append,config_file,filename,gmv,cinv,T3_opt)

    elapsed = time() - time0
    elapsed /= 60
    print('Time taken (minutes): ', elapsed)

def do_reconstruction(qe,append,config_file,filename,gmv,cinv,T3_opt):
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
    w_Tmv = np.loadtxt('ilc_weights/weights1d_TT_spt3g_cmbmv.dat')
    w_Emv = np.loadtxt('ilc_weights/weights1d_EE_spt3g_cmbmv.dat')
    w_Bmv = np.loadtxt('ilc_weights/weights1d_BB_spt3g_cmbmv.dat')
    # These are from Srini... Weird format. Assumes either one or two spectral energy distributions for CIB
    w_srini = np.load('ilc_weights/weights_cmb_mv_cmbfree_cibfree_spt3g1920.npy',allow_pickle=True)
    w_Tmv_srini_95 = w_srini.item()['cmbmv'][95][1]
    w_Tmv_srini_150 = w_srini.item()['cmbmv'][150][1]
    w_Tmv_srini_220 = w_srini.item()['cmbmv'][220][1]
    w_Tmv_srini = np.vstack((w_Tmv_srini_95,w_Tmv_srini_150,w_Tmv_srini_220))
    if append == 'agora_crossilc_onesed' or append == 'agora_crossilc_onesed_cinv':
        w_cib_null = np.load('ilc_weights/weights_cmb_mv_cmbfree_cibfree_spt3g1920.npy',allow_pickle=True)
        w_cib_null_95 = w_cib_null.item()['cmbcibfree'][95][1]
        w_cib_null_150 = w_cib_null.item()['cmbcibfree'][150][1]
        w_cib_null_220 = w_cib_null.item()['cmbcibfree'][220][1]
    elif append == 'agora_crossilc_twoseds' or append == 'agora_crossilc_twoseds_cinv':
        w_cib_null = np.load('ilc_weights/weights_cmb_mv_cmbfree_cibfreetwoSEDs_spt3g1920.npy',allow_pickle=True)
        w_cib_null_95 = w_cib_null.item()['cmbcibfree'][95][1]
        w_cib_null_150 = w_cib_null.item()['cmbcibfree'][150][1]
        w_cib_null_220 = w_cib_null.item()['cmbcibfree'][220][1]

    # Get Agora sim (signal + foregrounds)
    print('Getting alms...')
    tlm_95, elm_95, blm_95 = hp.read_alm(agora_095,hdu=[1,2,3])
    tlm_150, elm_150, blm_150 = hp.read_alm(agora_150,hdu=[1,2,3])
    tlm_220, elm_220, blm_220 = hp.read_alm(agora_220,hdu=[1,2,3])

    # Adding noise! (Using sim 999)
    nltt_090_090 = fsky_corr * noise_curves_090_090[:,1]; nlee_090_090 = fsky_corr * noise_curves_090_090[:,2]; nlbb_090_090 = fsky_corr * noise_curves_090_090[:,3]
    nltt_150_150 = fsky_corr * noise_curves_150_150[:,1]; nlee_150_150 = fsky_corr * noise_curves_150_150[:,2]; nlbb_150_150 = fsky_corr * noise_curves_150_150[:,3]
    nltt_220_220 = fsky_corr * noise_curves_220_220[:,1]; nlee_220_220 = fsky_corr * noise_curves_220_220[:,2]; nlbb_220_220 = fsky_corr * noise_curves_220_220[:,3]
    nltt_090_150 = fsky_corr * noise_curves_090_150[:,1]; nlee_090_150 = fsky_corr * noise_curves_090_150[:,2]; nlbb_090_150 = fsky_corr * noise_curves_090_150[:,3]
    nltt_090_220 = fsky_corr * noise_curves_090_220[:,1]; nlee_090_220 = fsky_corr * noise_curves_090_220[:,2]; nlbb_090_220 = fsky_corr * noise_curves_090_220[:,3]
    nltt_150_220 = fsky_corr * noise_curves_150_220[:,1]; nlee_150_220 = fsky_corr * noise_curves_150_220[:,2]; nlbb_150_220 = fsky_corr * noise_curves_150_220[:,3]
    nlm1_090_filename = dir_out + f'nlm/nlm_090_lmax{lmax}_seed999.alm'
    nlm1_150_filename = dir_out + f'nlm/nlm_150_lmax{lmax}_seed999.alm'
    nlm1_220_filename = dir_out + f'nlm/nlm_220_lmax{lmax}_seed999.alm'
    sim1 = 999
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
    tlm_150 += nlmt1_150; tlm_220 += nlmt1_220; tlm_95 += nlmt1_090
    elm_150 += nlme1_150; elm_220 += nlme1_220; elm_95 += nlme1_090
    blm_150 += nlmb1_150; blm_220 += nlmb1_220; blm_95 += nlmb1_090

    tlm_mv = hp.almxfl(tlm_95,w_Tmv[0][:lmax+1]) + hp.almxfl(tlm_150,w_Tmv[1][:lmax+1]) + hp.almxfl(tlm_220,w_Tmv[2][:lmax+1])
    tlm_tszn = hp.almxfl(tlm_95,w_tsz_null[0][:lmax+1]) + hp.almxfl(tlm_150,w_tsz_null[1][:lmax+1]) + hp.almxfl(tlm_220,w_tsz_null[2][:lmax+1])
    if append == 'agora_crossilc_onesed' or append == 'agora_crossilc_twoseds' or append == 'agora_crossilc_onesed_cinv' or append == 'agora_crossilc_twoseds_cinv':
        tlm_cibn = hp.almxfl(tlm_95,w_cib_null_95[:lmax+1]) + hp.almxfl(tlm_150,w_cib_null_150[:lmax+1]) + hp.almxfl(tlm_220,w_cib_null_220[:lmax+1])
    elm = hp.almxfl(elm_95,w_Emv[0][:lmax+1]) + hp.almxfl(elm_150,w_Emv[1][:lmax+1]) + hp.almxfl(elm_220,w_Emv[2][:lmax+1])
    blm = hp.almxfl(blm_95,w_Bmv[0][:lmax+1]) + hp.almxfl(blm_150,w_Bmv[1][:lmax+1]) + hp.almxfl(blm_220,w_Bmv[2][:lmax+1])

    # Get signal + noise residuals spectra for constructing fl filters
    print('Getting signal + noise residuals spectra for filtering')
    # If lmaxT != lmaxP, we add artificial noise in TT for ell > lmaxT
    artificial_noise = np.zeros(lmax+1)
    artificial_noise[lmaxT+2:] = 1.e10
    append_shortened = append[6:]
    if cinv:
        append_shortened = append_shortened[:-5]
    # Use the SAME filter for these Agora sims as what I used for the Gaussian sims, otherwise sim response from Gaussian sims won't match when I use it for Agora
    totalcls_filename = dir_out+f'totalcls/totalcls_average_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append_shortened}.npy'
    totalcls = np.load(totalcls_filename)
    if append == 'agora_standard' or append == 'agora_standard_cinv':
        cltt = totalcls[:,0]; clee = totalcls[:,1]; clbb = totalcls[:,2]; clte = totalcls[:,3]
    elif append == 'agora_mh' or append == 'agora_crossilc_onesed' or append == 'agora_crossilc_twoseds' or append == 'agora_mh_cinv' or append == 'agora_crossilc_onesed_cinv' or append == 'agora_crossilc_twoseds_cinv':
        # totalcls: T3T3, EE, BB, T3E, T1T1, T2T2, T1T2, T1T3, T2T3, T1E, T2E
        cltt1 = totalcls[:,4]; cltt2 = totalcls[:,5]; clttx = totalcls[:,6]; cltt3 = totalcls[:,0]; clee = totalcls[:,1]; clbb = totalcls[:,2]; clte = totalcls[:,3]

    if append == 'agora_standard_cinv':
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

    elif append == 'agora_mh_cinv':
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

    elif append == 'agora_crossilc_onesed_cinv' or append == 'agora_crossilc_twoseds_cinv':
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

    elif append == 'agora_standard' and gmv:
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

    elif append == 'agora_mh' and gmv:
        print('Doing the 1/Dl for GMV...')
        invDl1 = np.zeros(lmax+1, dtype=np.complex_)
        invDl2 = np.zeros(lmax+1, dtype=np.complex_)
        invDl3 = np.zeros(lmax+1, dtype=np.complex_)
        invDl1[lmin:] = 1./(cltt1[lmin:]*clee[lmin:] - clte[lmin:]**2)
        invDl2[lmin:] = 1./(cltt2[lmin:]*clee[lmin:] - clte[lmin:]**2)
        invDl3[lmin:] = 1./(cltt3[lmin:]*clee[lmin:] - clte[lmin:]**2)
        flb = np.zeros(lmax+1); flb[lmin:] = 1./clbb[lmin:]

        # Order is T1T2, T2T1, EE, TE, ET, TB, BT, EB, BE
        alm1all = np.zeros((len(tlm_mv),9), dtype=np.complex_)
        alm2all = np.zeros((len(tlm_tszn),9), dtype=np.complex_)
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
            # TE
            alm1all[:,3] = hp.almxfl(tlm_mv,invDl1)
            alm2all[:,3] = hp.almxfl(elm,invDl2)
            # ET
            alm1all[:,4] = hp.almxfl(elm,invDl1)
            alm2all[:,4] = hp.almxfl(tlm_tszn,invDl2)
            # TB
            alm1all[:,5] = hp.almxfl(tlm_mv,invDl1)
            alm2all[:,5] = hp.almxfl(blm,flb)
            # BT
            alm1all[:,6] = hp.almxfl(blm,flb)
            alm2all[:,6] = hp.almxfl(tlm_tszn,invDl2)
            # EB
            alm1all[:,7] = hp.almxfl(elm,invDl1)
            alm2all[:,7] = hp.almxfl(blm,flb)
            # BE
            alm1all[:,8] = hp.almxfl(blm,flb)
            alm2all[:,8] = hp.almxfl(elm,invDl2)
        elif T3_opt == 'withT3':
            # EE
            alm1all[:,2] = hp.almxfl(elm,invDl3)
            alm2all[:,2] = hp.almxfl(elm,invDl3)
            # TE
            alm1all[:,3] = hp.almxfl(tlm_mv,invDl3)
            alm2all[:,3] = hp.almxfl(elm,invDl3)
            # ET
            alm1all[:,4] = hp.almxfl(elm,invDl3)
            alm2all[:,4] = hp.almxfl(tlm_mv,invDl3)
            # TB
            alm1all[:,5] = hp.almxfl(tlm_mv,invDl3)
            alm2all[:,5] = hp.almxfl(blm,flb)
            # BT
            alm1all[:,6] = hp.almxfl(blm,flb)
            alm2all[:,6] = hp.almxfl(tlm_mv,invDl3)
            # EB
            alm1all[:,7] = hp.almxfl(elm,invDl3)
            alm2all[:,7] = hp.almxfl(blm,flb)
            # BE
            alm1all[:,8] = hp.almxfl(blm,flb)
            alm2all[:,8] = hp.almxfl(elm,invDl3)

    elif (append == 'agora_crossilc_onesed' or append == 'agora_crossilc_twoseds') and gmv:
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
        if T3_opt == 'noT3':
            # EE
            alm1all[:,2] = hp.almxfl(elm,invDl1)
            alm2all[:,2] = hp.almxfl(elm,invDl2)
            # TE
            alm1all[:,3] = hp.almxfl(tlm_cibn,invDl1)
            alm2all[:,3] = hp.almxfl(elm,invDl2)
            # ET
            alm1all[:,4] = hp.almxfl(elm,invDl1)
            alm2all[:,4] = hp.almxfl(tlm_tszn,invDl2)
            # TB
            alm1all[:,5] = hp.almxfl(tlm_cibn,invDl1)
            alm2all[:,5] = hp.almxfl(blm,flb)
            # BT
            alm1all[:,6] = hp.almxfl(blm,flb)
            alm2all[:,6] = hp.almxfl(tlm_tszn,invDl2)
            # EB
            alm1all[:,7] = hp.almxfl(elm,invDl1)
            alm2all[:,7] = hp.almxfl(blm,flb)
            # BE
            alm1all[:,8] = hp.almxfl(blm,flb)
            alm2all[:,8] = hp.almxfl(elm,invDl2)
        elif T3_opt == 'withT3':
            # EE
            alm1all[:,2] = hp.almxfl(elm,invDl3)
            alm2all[:,2] = hp.almxfl(elm,invDl3)
            # TE
            alm1all[:,3] = hp.almxfl(tlm_mv,invDl3)
            alm2all[:,3] = hp.almxfl(elm,invDl3)
            # ET
            alm1all[:,4] = hp.almxfl(elm,invDl3)
            alm2all[:,4] = hp.almxfl(tlm_mv,invDl3)
            # TB
            alm1all[:,5] = hp.almxfl(tlm_mv,invDl3)
            alm2all[:,5] = hp.almxfl(blm,flb)
            # BT
            alm1all[:,6] = hp.almxfl(blm,flb)
            alm2all[:,6] = hp.almxfl(tlm_mv,invDl3)
            # EB
            alm1all[:,7] = hp.almxfl(elm,invDl3)
            alm2all[:,7] = hp.almxfl(blm,flb)
            # BE
            alm1all[:,8] = hp.almxfl(blm,flb)
            alm2all[:,8] = hp.almxfl(elm,invDl3)

    elif append == 'agora_standard':
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

    elif append == 'agora_mh':
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
            else:
                if qe[0] == 'T': almbar1 = hp.almxfl(tlm_mv,flt1); flm1 = flt1
                elif qe[0] == 'E': almbar1 = hp.almxfl(elm,fle); flm1 = fle
                elif qe[0] == 'B': almbar1 = hp.almxfl(blm,flb); flm1 = flb

                if qe[1] == 'T': almbar2 = hp.almxfl(tlm_mv,flt1); flm2 = flt1
                elif qe[1] == 'E': almbar2 = hp.almxfl(elm,fle); flm2 = fle
                elif qe[1] == 'B': almbar2 = hp.almxfl(blm,flb); flm2 = flb

    elif append == 'agora_crossilc_onesed' or append == 'agora_crossilc_twoseds':
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
    if gmv and not cinv:
        q_gmv = qest.qest_gmv(config,cls)
        if append == 'agora_standard':
            glm,clm = q_gmv.eval(qe,alm1all,alm2all,totalcls,crossilc=False)
        elif append == 'agora_mh' or append == 'agora_crossilc_onesed' or append == 'agora_crossilc_twoseds':
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
