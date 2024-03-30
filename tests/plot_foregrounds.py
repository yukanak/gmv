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

# Agora lensed CMB only
lcmb = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_lcmb_spt3g_150ghz_alm_lmax4096.fits'
# CIB
lcib = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_lcib_spt3g_150ghz_alm_lmax4096.fits'
# tSZ
ltsz = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_ltsz_spt3g_150ghz_alm_lmax4096.fits'

tlm_cmb,elm_cmb,blm_cmb = hp.read_alm(lcmb,hdu=[1,2,3])
tlm_cib,elm_cib,blm_cib = hp.read_alm(lcib,hdu=[1,2,3])
tlm_tsz,elm_tsz,blm_tsz = hp.read_alm(ltsz,hdu=[1,2,3])

cltt_cmb = hp.alm2cl(tlm_cmb,tlm_cmb)
cltt_cib = hp.alm2cl(tlm_cib,tlm_cib)
cltt_tsz = hp.alm2cl(tlm_tsz,tlm_tsz)

plt.figure(0)
plt.clf()

plt.plot(l, cltt_cmb, color='firebrick', linestyle='-', label=f'CMB')
plt.plot(l, cltt_cib, color='forestgreen', linestyle='-', label=f'tSZ')
plt.plot(l, cltt_tsz, color='darkorange', linestyle='-', label=f'CIB')

plt.ylabel("$C_\ell^{TT}$")
plt.xlabel('$\ell$')
plt.title(f'150 GHz TT Power Spectrum')
plt.legend(loc='upper left', fontsize='small', bbox_to_anchor=(1,0.5))
plt.xscale('log')
plt.xlim(10,lmax)
#plt.ylim(-0.3,0.3)
plt.savefig(dir_out+f'/figs/foregrounds.png',bbox_inches='tight')



