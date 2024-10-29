#!/usr/bin/env python3
# Run like python3 get_plms_example.py TT 100 101 yuuki
import os, sys
import numpy as np
import healpy as hp
from pathlib import Path
from time import time
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import healqest_utils as utils
import qest as qest

def main():
    qe = str(sys.argv[1])
    sim1 = int(sys.argv[2])
    sim2 = int(sys.argv[3])
    append = str(sys.argv[4])
    config_file = 'profhrd_yuka.yaml'

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

    if os.path.isfile(filename_sqe):
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
    print(f'Doing SQE reconstruction for sims {sim1} and {sim2}, qe {qe}, append {append}')

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
    gmv = False
    u = None
    filename_sqe = dir_out+f'/plm_{qe}_healqest_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy'

    # Noise curves
    nltt = (np.pi/180./60.*5.0)**2 / hp.gauss_beam(fwhm=0.000290888,lmax=lmax)**2
    nlee = nlbb = (np.pi/180./60.*5.0)**2 / hp.gauss_beam(fwhm=0.000290888,lmax=lmax)**2

    # CMB is same at all frequencies; full sky
    # From amscott:/sptlocal/analysis/eete+lensing_19-20/resources/sims/planck2018/inputcmb/
    alm_cmb_sim1 = f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/tqu1/len/alms/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb1_seed{sim1}_alm_lmax{lmax}.fits'
    alm_cmb_sim2 = f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/tqu1/len/alms/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb1_seed{sim2}_alm_lmax{lmax}.fits'
    alm_cmb_sim1_tqu2 = f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/tqu2/len/alms/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb2_seed{sim1}_alm_lmax{lmax}.fits'

    # Get full sky CMB alms
    print('Getting alms...')
    if append == 'yuuki' or append == 'yuuki_cmbonly':
        tlm1,elm1,blm1 = hp.read_alm(alm_cmb_sim1,hdu=[1,2,3])
        tlm2,elm2,blm2 = hp.read_alm(alm_cmb_sim2,hdu=[1,2,3])
    elif append == 'yuuki_cmbonly_phi1_tqu1tqu2':
        tlm1,elm1,blm1 = hp.read_alm(alm_cmb_sim1,hdu=[1,2,3])
        tlm2,elm2,blm2 = hp.read_alm(alm_cmb_sim1_tqu2,hdu=[1,2,3])
    elif append == 'yuuki_cmbonly_phi1_tqu2tqu1':
        tlm1,elm1,blm1 = hp.read_alm(alm_cmb_sim1_tqu2,hdu=[1,2,3])
        tlm2,elm2,blm2 = hp.read_alm(alm_cmb_sim1,hdu=[1,2,3])
    tlm1 = utils.reduce_lmax(tlm1,lmax=lmax)
    elm1 = utils.reduce_lmax(elm1,lmax=lmax)
    blm1 = utils.reduce_lmax(blm1,lmax=lmax)
    tlm2 = utils.reduce_lmax(tlm2,lmax=lmax)
    elm2 = utils.reduce_lmax(elm2,lmax=lmax)
    blm2 = utils.reduce_lmax(blm2,lmax=lmax)

    # Adding noise!
    if append == 'yuuki':
        nlm1_filename = dir_out + f'/nlm/yuuki_20240802_nlm_lmax{lmax}_seed{sim1}.alm'
        nlm2_filename = dir_out + f'/nlm/yuuki_20240802_nlm_lmax{lmax}_seed{sim2}.alm'
        if os.path.isfile(nlm1_filename):
            nlmt1,nlme1,nlmb1 = hp.read_alm(nlm1_filename,hdu=[1,2,3])
        else:
            np.random.seed(4190002645+sim1)
            nlmt1,nlme1,nlmb1 = hp.synalm([nltt,nlee,nlbb,nltt*0],new=True,lmax=lmax)
            Path(dir_out+f'/nlm/').mkdir(parents=True, exist_ok=True)
            hp.write_alm(nlm1_filename,[nlmt1,nlme1,nlmb1])
        if os.path.isfile(nlm2_filename):
            nlmt2,nlme2,nlmb2 = hp.read_alm(nlm2_filename,hdu=[1,2,3])
        else:
            np.random.seed(4190002645+sim2)
            nlmt2,nlme2,nlmb2 = hp.synalm([nltt,nlee,nlbb,nltt*0],new=True,lmax=lmax)
            Path(dir_out+f'/nlm/').mkdir(parents=True, exist_ok=True)
            hp.write_alm(nlm2_filename,[nlmt2,nlme2,nlmb2])
        tlm1 += nlmt1; elm1 += nlme1; blm1 += nlmb1
        tlm2 += nlmt2; elm2 += nlme2; blm2 += nlmb2

    # Get signal + noise residuals spectra for constructing fl filters
    print('Getting signal + noise residuals spectra for filtering')
    # If lmaxT != lmaxP, we add artificial noise in TT for ell > lmaxT
    artificial_noise = np.zeros(lmax+1)
    artificial_noise[lmaxT+2:] = 1.e10
    # Even if your sims here don't have added noise, you still want to include the nl in the filtering
    # (e.g. when you are computing noiseless sims for the N1 calculation, you want the filter to still include nl to suppress modes exactly as in the signal map)
    cltt = sl['tt'][:lmax+1] + nltt[:lmax+1] + artificial_noise
    clee = sl['ee'][:lmax+1] + nlee[:lmax+1]
    clbb = sl['bb'][:lmax+1] + nlbb[:lmax+1]
    clte = sl['te'][:lmax+1]

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

    # Run healqest
    print('Running healqest...')
    q_original = qest.qest(config,cls)
    glm,clm = q_original.eval(qe,almbar1,almbar2)
    # Save plm and clm
    Path(dir_out).mkdir(parents=True, exist_ok=True)
    np.save(filename_sqe,glm)

if __name__ == '__main__':

    main()
