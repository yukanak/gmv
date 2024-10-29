#!/usr/bin/env python3
# Run like python3 get_plms_example.py TT 100 101 append test_yuka.yaml
# Note: argument append should be either 'profhrd_flatgaussian' (used for actual reconstruction and N0 calculation, lensed CMB + foregrounds in T + noise),
# 'profhrd_flatgaussian_cmbonly_phi1_tqu1tqu2', 'profhrd_flatgaussian_cmbonly_phi1_tqu2tqu1' (used for N1 calculation, these are lensed with the same phi but different CMB realizations, no foregrounds or noise),
# 'profhrd_flatgaussian_cmbonly' (used for N0 calculation for subtracting from N1, lensed CMB + no foregrounds + no noise),
# 'profhrd_flatgaussian_unl_cmbonly' (unlensed sims + no foregrounds + no noise), or 'profhrd_flatgaussian_unl' (unlensed sims + foregrounds + noise)
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
    sim1 = int(sys.argv[2])
    sim2 = int(sys.argv[3])
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

    if False:#os.path.isfile(filename_sqe) or os.path.isfile(filename_gmv):
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

    # Noise curves
    noise_file='noise_curves/nl_cmbmv_20192020.dat'
    fsky_corr = 25.308939726920805

    # Foregrounds are Gaussian random samples from a flat Cl power spectrum at 2.18e-05 uK^2
    fgtt =  np.ones(lmax+1) * 2.18e-5

    # Full sky CMB signal maps; same at all frequencies
    # From amscott:/sptlocal/analysis/eete+lensing_19-20/resources/sims/planck2018/inputcmb/
    alm_cmb_sim1 = f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/tqu1/len/alms/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb1_seed{sim1}_alm_lmax{lmax}.fits'
    alm_cmb_sim2 = f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/tqu1/len/alms/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb1_seed{sim2}_alm_lmax{lmax}.fits'
    alm_cmb_sim1_tqu2 = f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/tqu2/len/alms/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb2_seed{sim1}_alm_lmax{lmax}.fits'

    # Unlensed CMB alms sampled from lensed theory spectra
    unl_map_sim1 = f'/oak/stanford/orgs/kipac/users/yukanaka/unl_from_lensed_cls/unl_from_lensed_cls_seed{sim1}_lmax{lmax}_nside{nside}_20230905.fits'
    unl_map_sim2 = f'/oak/stanford/orgs/kipac/users/yukanaka/unl_from_lensed_cls/unl_from_lensed_cls_seed{sim2}_lmax{lmax}_nside{nside}_20230905.fits'

    # Profile
    profile_filename = 'fg_profiles/flat_profile.pkl'
    if os.path.isfile(profile_filename):
        u = pickle.load(open(profile_filename,'rb'))
    else:
        u = np.ones(lmax+1, dtype=np.complex_)
        with open(profile_filename,'wb') as f:
            pickle.dump(u,f)

    if qe == 'TTEETE' or qe == 'TBEB' or qe == 'all' or qe == 'TTEETEprf':
        gmv = True
    elif qe == 'TT' or qe == 'TE' or  qe == 'ET' or qe == 'EE' or qe == 'TB' or  qe == 'BT' or qe == 'EB' or  qe == 'BE' or qe == 'TTprf' or qe == 'T1T2' or qe == 'T2T1':
        gmv = False
    else:
        print('Invalid qe!')

    # Get full sky CMB alms
    print('Getting alms...')
    if append == f'profhrd_flatgaussian' or append == 'profhrd_flatgaussian_cmbonly':
        tlm1,elm1,blm1 = hp.read_alm(alm_cmb_sim1,hdu=[1,2,3])
        tlm2,elm2,blm2 = hp.read_alm(alm_cmb_sim2,hdu=[1,2,3])
    elif append == 'profhrd_flatgaussian_cmbonly_phi1_tqu1tqu2':
        tlm1,elm1,blm1 = hp.read_alm(alm_cmb_sim1,hdu=[1,2,3])
        tlm2,elm2,blm2 = hp.read_alm(alm_cmb_sim1_tqu2,hdu=[1,2,3])
    elif append == 'profhrd_flatgaussian_cmbonly_phi1_tqu2tqu1':
        tlm1,elm1,blm1 = hp.read_alm(alm_cmb_sim1_tqu2,hdu=[1,2,3])
        tlm2,elm2,blm2 = hp.read_alm(alm_cmb_sim1,hdu=[1,2,3])
    elif append == 'profhrd_flatgaussian_unl' or append == 'profhrd_flatgaussian_unl_cmbonly':
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

    # Adding foregrounds!
    if append == f'profhrd_flatgaussian' or append == 'profhrd_flatgaussian_unl':
        flmt1_filename = dir_out + f'/flm/flmt_flatgaussian_lmax{lmax}_seed{sim1}.alm'
        flmt2_filename = dir_out + f'/flm/flmt_flatgaussian_lmax{lmax}_seed{sim2}.alm'
        if os.path.isfile(flmt1_filename):
            flmt1 = hp.read_alm(flmt1_filename,hdu=1)
        else:
            np.random.seed(3241998+sim1)
            flmt1 = hp.synalm(fgtt,lmax=lmax)
            Path(dir_out+f'/flm/').mkdir(parents=True, exist_ok=True)
            hp.write_alm(flmt1_filename,flmt1)
        if os.path.isfile(flmt2_filename):
            flmt2 = hp.read_alm(flmt2_filename,hdu=1)
        else:
            np.random.seed(3241998+sim2)
            flmt2 = hp.synalm(fgtt,lmax=lmax)
            Path(dir_out+f'/flm/').mkdir(parents=True, exist_ok=True)
            hp.write_alm(flmt2_filename,flmt2)
        tlm1 += flmt1
        tlm2 += flmt2

    # Adding noise!
    if append == 'profhrd_flatgaussian' or append == 'profhrd_flatgaussian_unl':
        noise_curves = np.loadtxt(noise_file)
        nltt = fsky_corr * noise_curves[:,1]; nlee = fsky_corr * noise_curves[:,2]; nlbb = fsky_corr * noise_curves[:,2]
        nlm1_filename = dir_out + f'/nlm/2019_2020_ilc_noise_nlm_lmax{lmax}_seed{sim1}.alm'
        nlm2_filename = dir_out + f'/nlm/2019_2020_ilc_noise_nlm_lmax{lmax}_seed{sim2}.alm'
        if os.path.isfile(nlm1_filename):
            nlmt1,nlme1,nlmb1 = hp.read_alm(nlm1_filename,hdu=[1,2,3])
        else:
            np.random.seed(hash('tora')%2**32+sim1)
            nlmt1,nlme1,nlmb1 = hp.synalm([nltt,nlee,nlbb,nltt*0],new=True,lmax=lmax)
            Path(dir_out+f'/nlm/').mkdir(parents=True, exist_ok=True)
            hp.write_alm(nlm1_filename,[nlmt1,nlme1,nlmb1])
        if os.path.isfile(nlm2_filename):
            nlmt2,nlme2,nlmb2 = hp.read_alm(nlm2_filename,hdu=[1,2,3])
        else:
            np.random.seed(hash('tora')%2**32+sim2)
            nlmt2,nlme2,nlmb2 = hp.synalm([nltt,nlee,nlbb,nltt*0],new=True,lmax=lmax)
            Path(dir_out+f'/nlm/').mkdir(parents=True, exist_ok=True)
            hp.write_alm(nlm2_filename,[nlmt2,nlme2,nlmb2])
        tlm1 += nlmt1; elm1 += nlme1; blm1 += nlmb1
        tlm2 += nlmt2; elm2 += nlme2; blm2 += nlmb2

    # Get signal + noise spectra for constructing fl filters
    print('Getting signal + noise residuals spectra for filtering')
    # If lmaxT != lmaxP, we add artificial noise in TT for ell > lmaxT
    artificial_noise = np.zeros(lmax+1)
    artificial_noise[lmaxT+2:] = 1.e10
    totalcls_filename = dir_out+f'totalcls/totalcls_average_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_profhrd_flatgaussian_theory.npy'
    if os.path.isfile(totalcls_filename):
        totalcls = np.load(totalcls_filename)
        cltt = totalcls[:,0]; clee = totalcls[:,1]; clbb = totalcls[:,2]; clte = totalcls[:,3]
    else:
        noise_curves = np.loadtxt(noise_file)
        nltt = fsky_corr * noise_curves[:,1]; nlee = fsky_corr * noise_curves[:,2]; nlbb = fsky_corr * noise_curves[:,2]
        # Resulting spectra
        cltt = sl['tt'][:lmax+1] + nltt[:lmax+1] + fgtt + artificial_noise
        clee = sl['ee'][:lmax+1] + nlee[:lmax+1]
        clbb = sl['bb'][:lmax+1] + nlbb[:lmax+1]
        clte = sl['te'][:lmax+1]
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
        return
    else:
        q_gmv = qest.qest_gmv(config,cls)
        glm,clm = q_gmv.eval(qe,alm1all,alm2all,totalcls,u=u,crossilc=False)
        # Save plm and clm
        Path(dir_out).mkdir(parents=True, exist_ok=True)
        np.save(filename_gmv,glm)
        return

if __name__ == '__main__':

    main()
