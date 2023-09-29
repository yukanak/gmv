#!/usr/bin/env python3
import sys
import numpy as np
import healpy as hp
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import utils
import weights
import qest
import camb
from pathlib import Path
import os
import matplotlib.pyplot as plt

####################################
fluxlim = 0.200
cambini = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_params.ini'
dir_out = '/scratch/users/yukanaka/gmv/'
config_file = 'profhrd_yuka.yaml'
noise_file='nl_cmbmv_20192020.dat'
#noise_file = None
fsky_corr=25.308939726920805
append = 'cmbonly'
sim1 = 1
sim2 = 122
####################################
config = utils.parse_yaml(config_file)
lmax = config['lmax']
lmaxT = config['lmaxt']
lmaxP = config['lmaxp']
lmin = config['lmint']
nside = config['nside']
cltype = config['cltype']
cls = config['cls']
sl = {ee:config['cls'][cltype][ee] for ee in config['cls'][cltype].keys()}
clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
fgtt =  np.ones(lmax+1) * 2.18e-5
l = np.arange(0,lmax+1)

alm_cmb = f'/scratch/users/yukanaka/lensing19-20/inputcmb/tqu1/len/alms/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb1_seed{sim1}_alm_lmax{lmax}.fits'
alm_cmb_old = f'/scratch/users/yukanaka/spt3g_planck2018alms_lowpass5000/lensedTQU1phi1_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_seed{sim2}_lmax9000_nside8192_interp1.0_method1_pol_1_lensed_alm_lowpass5000.fits'
flm = f'/scratch/users/yukanaka/rand_ptsrc_rlz/src_fluxlim{fluxlim:.3f}_alm_set1_rlz{sim1}.fits'
####################################
tlm,elm,blm = hp.read_alm(alm_cmb,hdu=[1,2,3])
tlm = utils.reduce_lmax(tlm,lmax=lmax)
elm = utils.reduce_lmax(elm,lmax=lmax)
blm = utils.reduce_lmax(blm,lmax=lmax)
#tlm_old,elm_old,blm_old = hp.read_alm(alm_cmb_old,hdu=[1,2,3])
#tlm_old = utils.reduce_lmax(tlm_old,lmax=lmax)
#elm_old = utils.reduce_lmax(elm_old,lmax=lmax)
#blm_old = utils.reduce_lmax(blm_old,lmax=lmax)
input_plm = hp.read_alm(f'/scratch/users/yukanaka/lensing19-20/inputcmb/phi/phi_lmax_{lmax}/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_phi1_seed{sim1}_lmax{lmax}.alm')
#input_plm_old = hp.read_alm(f'/scratch/users/yukanaka/input_phi1/phi_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_seed{sim2}_alm_lmax5000.fits')
#input_plm_old = utils.reduce_lmax(input_plm_old,lmax=lmax)
flm = hp.read_alm(flm,hdu=[1])
flm = utils.reduce_lmax(flm,lmax=lmax)
tlm_with_fg = tlm + flm

ttspec = hp.alm2cl(tlm,tlm, lmax=lmax)
eespec = hp.alm2cl(elm,elm, lmax=lmax)
bbspec = hp.alm2cl(blm,blm, lmax=lmax)
#ttspec_old = hp.alm2cl(tlm_old,tlm_old, lmax=lmax)
#eespec_old = hp.alm2cl(elm_old,elm_old, lmax=lmax)
#bbspec_old = hp.alm2cl(blm_old,blm_old, lmax=lmax)
ppspec = hp.alm2cl(input_plm,input_plm, lmax=lmax)
#ppspec_old = hp.alm2cl(input_plm_old,input_plm_old, lmax=lmax)
fgspec = hp.alm2cl(flm,flm, lmax=lmax)

plt.figure(0)
plt.clf()
plt.plot(l, sl['tt'] * (l*(l+1))**2/4, color='firebrick', label='Theory TT')
plt.plot(l, sl['ee'] * (l*(l+1))**2/4, color='darkblue', label='Theory EE')
plt.plot(l, sl['bb'] * (l*(l+1))**2/4, color='forestgreen', label='Theory BB')
plt.plot(l, slpp * (l*(l+1))**2/4, color='darkorange', label='Theory KK')
plt.plot(l, fgtt * (l*(l+1))**2/4, color='darkorchid', label='Theory TT Foregrounds')
plt.plot(l, (sl['tt'] + fgtt) * (l*(l+1))**2/4, color='goldenrod', label='Theory TT + Foregrounds')
plt.plot(l, ttspec * (l*(l+1))**2/4, color='lightcoral', linestyle='--', label='TT')
plt.plot(l, eespec * (l*(l+1))**2/4, color='cornflowerblue', linestyle='--', label='EE')
plt.plot(l, bbspec * (l*(l+1))**2/4, color='lightgreen', linestyle='--', label='BB')
plt.plot(l, ppspec * (l*(l+1))**2/4, color='orange', linestyle='--', label='KK')
plt.plot(l, fgspec * (l*(l+1))**2/4, color='thistle', linestyle='--', label='TT Foregrounds')
plt.plot(l, (ttspec + fgspec) * (l*(l+1))**2/4, color='palegoldenrod', linestyle='--', label='TT + Foregrounds')
plt.xlabel('$\ell$')
plt.title('Input Spectra')
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-15,1e10)
plt.legend(loc='lower left', fontsize='x-small')
plt.savefig('/scratch/users/yukanaka/gmv/figs/input_spectra.png')

"""
plt.figure(0)
plt.clf()

for sim in np.arange(100)+1:
    nlm_filename = f'/scratch/users/yukanaka/gmv/nlm/2019_2020_ilc_noise_nlm_lmax{lmax}_seed{sim}.npy'
    nlm = np.load(nlm_filename,allow_pickle=True)
    nlmt = nlm[:,0]
    nlme = nlm[:,0]
    nlmb = nlm[:,0]
    nltt = hp.alm2cl(nlmt,nlmt,lmax=lmax)
    nlee = hp.alm2cl(nlme,nlme,lmax=lmax)
    nlbb = hp.alm2cl(nlmb,nlmb,lmax=lmax)
    plt.plot(l, nlbb * (l*(l+1))**2/4)

plt.xlabel('$\ell$')
plt.title('Noise Realizations, BB')
plt.xscale('log')
plt.yscale('log')
plt.savefig('/scratch/users/yukanaka/gmv/figs/input_noise_bb.png')
"""    
