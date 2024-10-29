import matplotlib.pyplot as plt
import os, sys
import numpy as np
import healpy as hp
import pickle
from pathlib import Path
from time import time
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import healqest_utils as utils
import qest

config_file = 'test_yuka.yaml'

config = utils.parse_yaml(config_file)
#lmax = config['lensrec']['lmax']
#lmax = config['lensrec']['lmax']
lmax = 6000
dir_out = config['dir_out']
l = np.arange(0,lmax+1)
clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)

# Agora lensed CMB only
lcmb = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_lcmb_spt3g_150ghz_alm_lmax4096.fits'
#tlm_cmb,elm_cmb,blm_cmb = hp.read_alm(lcmb,hdu=[1,2,3])
#cltt_cmb = hp.alm2cl(tlm_cmb,tlm_cmb) * (l*(l+1))/(2*np.pi)

# Agora CIB
#lcib = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_lcib_spt3g_150ghz_alm_lmax6000.fits'
#tlm_cib,elm_cib,blm_cib = hp.read_alm(lcib,hdu=[1,2,3])
#cltt_cib = hp.alm2cl(tlm_cib,tlm_cib) * (l*(l+1))/(2*np.pi)

# Agora tSZ
#ltsz = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_ltsz_spt3g_150ghz_alm_lmax6000.fits'
#tlm_tsz,elm_tsz,blm_tsz = hp.read_alm(ltsz,hdu=[1,2,3])
#cltt_tsz = hp.alm2cl(tlm_tsz,tlm_tsz) * (l*(l+1))/(2*np.pi)

# Total foregrounds from sims
fg_sim1 = '/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_150ghz_seed1_alm_lmax4096.fits'
fg_sim150 = '/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_150ghz_seed150_alm_lmax4096.fits'
tflm_sim1,eflm_sim1,bflm_sim1 = hp.read_alm(fg_sim1,hdu=[1,2,3])
tflm_sim150,eflm_sim150,bflm_sim150 = hp.read_alm(fg_sim150,hdu=[1,2,3])
l_4096 = np.arange(0,4096+1)
fltt_sim1 = hp.alm2cl(tflm_sim1,tflm_sim1) * (l_4096*(l_4096+1))/(2*np.pi)
fltt_sim150 = hp.alm2cl(tflm_sim150,tflm_sim150) * (l_4096*(l_4096+1))/(2*np.pi)

plt.figure(0)
plt.clf()

plt.plot(l, sltt * (l*(l+1))/(2*np.pi), color='k', linestyle='-', label=f'CMB')
#plt.plot(l, cltt_cib, color='forestgreen', linestyle='-', label=f'tSZ')
#plt.plot(l, cltt_tsz, color='darkorange', linestyle='-', label=f'CIB')
plt.plot(l_4096, fltt_sim1, color='forestgreen', linestyle='-', label=f'fltt, sim 1')
plt.plot(l_4096, fltt_sim150, color='darkorange', linestyle='-', label=f'fltt, sim 2')

plt.ylabel("$C_\ell^{TT}$")
plt.ylabel("$\ell(\ell+1)$$C_\ell^{TT}$ / $2\pi$ $[\mu K^2]$")
plt.xlabel('$\ell$')
plt.title(f'150 GHz TT Power Spectrum')
plt.legend(loc='upper right', fontsize='small')
plt.xscale('log')
plt.yscale('log')
plt.xlim(200,lmax)
plt.ylim(1,1e4)
#plt.savefig(dir_out+f'/figs/foregrounds.png',bbox_inches='tight')
plt.savefig(dir_out+f'/figs/foregrounds_check_sim1_vs_sim2.png',bbox_inches='tight')
