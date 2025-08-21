#!/usr/bin/env python3
import os, sys
import numpy as np
import healpy as hp
import pickle
from pathlib import Path
from time import time
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import healqest_utils as utils
import qest
import matplotlib.pyplot as plt

config_file = 'test_yuka_lmaxT3500.yaml'
config = utils.parse_yaml(config_file)
dir_out = config['dir_out']
cltype = config['lensrec']['cltype']
cls = config['cls']
sl = {ee:config['cls'][cltype][ee] for ee in config['cls'][cltype].keys()}

# ILC weights
#==================== AGORA ====================#
'''
# Dimension (3, 6001) for 90, 150, 220 GHz respectively
w_tsz_null = np.loadtxt('ilc_weights/weights1d_TT_spt3g_cmbynull.dat')
w_Tmv = np.loadtxt('ilc_weights/weights1d_TT_spt3g_cmbmv.dat')
w_Emv = np.loadtxt('ilc_weights/weights1d_EE_spt3g_cmbmv.dat')
w_Bmv = np.loadtxt('ilc_weights/weights1d_BB_spt3g_cmbmv.dat')
# Dimension (3, 5000)
w_cib_null_onesed = np.load('ilc_weights/weights_cmb_mv_cmbfree_cibfree_spt3g1920.npy',allow_pickle=True)
w_cib_null_onesed = np.vstack([w_cib_null_onesed.item()['cmbcibfree'][95][1], w_cib_null_onesed.item()['cmbcibfree'][150][1], w_cib_null_onesed.item()['cmbcibfree'][220][1]])
w_cib_null_twoseds = np.load('ilc_weights/weights_cmb_mv_cmbfree_cibfreetwoSEDs_spt3g1920.npy',allow_pickle=True)
w_cib_null_twoseds = np.vstack([w_cib_null_twoseds.item()['cmbcibfree'][95][1], w_cib_null_twoseds.item()['cmbcibfree'][150][1], w_cib_null_twoseds.item()['cmbcibfree'][220][1]])
'''
# What I used previously for full-sky test
w_tsz_null_fullskytest = np.loadtxt('ilc_weights/weights1d_TT_spt3g_cmbynull.dat')
w_cib_null_twoseds = np.load('ilc_weights/weights_cmb_mv_cmbfree_cibfreetwoSEDs_spt3g1920.npy',allow_pickle=True)
w_cib_null_twoseds = np.vstack([w_cib_null_twoseds.item()['cmbcibfree'][95][1], w_cib_null_twoseds.item()['cmbcibfree'][150][1], w_cib_null_twoseds.item()['cmbcibfree'][220][1]])

# FROM CROSSOVER, 1D weights
w_Tmv = np.load('ilc_weights/from_crossover_20250722/weights1d_TT_spt3g20192020_052425_cmbmv_crosstf.npy',allow_pickle=True)
w_Emv = w_Bmv = np.load('ilc_weights/from_crossover_20250722/weights1d_EE_spt3g20192020_052425_cmbmv_crosstf.npy',allow_pickle=True)
w_tsz_null = np.loadtxt('ilc_weights/from_crossover_20250722/weights1d_TT_spt3g_cmbynull.dat')
w_cib_null = np.loadtxt('ilc_weights/from_crossover_20250722/weights1d_TT_spt3g_cmbcibnull.dat')

# FROM CROSSOVER, these are 2D weights used in the 2019/2020 analysis
w_Tmv_2d_yuuki = np.load('ilc_weights/from_crossover_20250722/weights2d_TT_spt3g20192020_052425_cmbmv_crosstf.npy',allow_pickle=True)
w_Emv_2d_yuuki = w_Bmv_yuuki = np.load('ilc_weights/from_crossover_20250722/weights2d_EE_spt3g20192020_052425_cmbmv_crosstf.npy',allow_pickle=True)
w_tsz_null_2d_yuuki = np.load('ilc_weights/from_crossover_20250722/weights2d_TT_spt3g20192020_052425_cmbynull_crosstf.npy',allow_pickle=True)
w_cib_null_2d_yuuki = np.load('ilc_weights/from_crossover_20250722/weights2d_TT_spt3g20192020_052425_cmbcibnull_crosstf.npy',allow_pickle=True)
# Convert to 1D for plotting!!!
n_freq, n_alm = w_Tmv_2d_yuuki.shape
lmax = hp.Alm.getlmax(n_alm)
l = np.arange(0,lmax+1)
# Average over m for each ell
w_Tmv_yuuki = np.zeros((n_freq, lmax+1))
w_Emv_yuuki = np.zeros((n_freq, lmax+1))
w_tsz_null_yuuki = np.zeros((n_freq, lmax+1))
w_cib_null_yuuki = np.zeros((n_freq, lmax+1))
for ell in l[500:]:
    #m_vals = np.arange(ell + 1)
    m_vals = 500
    idxs = hp.Alm.getidx(lmax, ell, m_vals)
    w_Tmv_yuuki[:, ell] = w_Tmv_2d_yuuki[:, idxs]#np.mean(w_Tmv_2d_yuuki[:, idxs], axis=1)
    w_Emv_yuuki[:, ell] = w_Emv_2d_yuuki[:, idxs]#np.mean(w_Emv_2d_yuuki[:, idxs], axis=1)
    w_tsz_null_yuuki[:, ell] = w_tsz_null_2d_yuuki[:, idxs]#np.mean(w_tsz_null_2d_yuuki[:, idxs], axis=1)
    w_cib_null_yuuki[:, ell] = w_cib_null_2d_yuuki[:, idxs]#np.mean(w_cib_null_2d_yuuki[:, idxs], axis=1)

# BY ANA
w_ana = np.load('/home/users/yukanaka/gmv/pipeline/ilc_weights/ilc_weights_cmb_spt3g_2yr.npy',allow_pickle=True).item()
cases = ['mv', 'tsznull', 'cibnull']
# Dimension (3, 4097) for 90, 150, 220 GHz respectively
w_Tmv_ana = w_ana['tt']['mv']; w_tsz_null_ana = w_ana['tt']['tsznull']; w_cib_null_ana = w_ana['tt']['cibnull']
w_Emv_ana = w_ana['ee']['mv'];
w_Bmv_ana = w_ana['bb']['mv'];
#===============================================#

#==================== WEBSKY ====================#
# These are dimensions (4097, 3) initially; then transpose to make it (3, 4097)
w_tsz_null_websky = np.load('/oak/stanford/orgs/kipac/users/yukanaka/websky/weights_websky_cmbrec_tsznull_lmax4096.npy').T
w_Tmv_websky = np.load('/oak/stanford/orgs/kipac/users/yukanaka/websky/weights_websky_cmbrec_mv_lmax4096.npy').T
w_cib_null_websky = np.load('/oak/stanford/orgs/kipac/users/yukanaka/websky/weights_websky_cmbrec_cibnull_lmax4096.npy').T
# Dimension (3, 6001) for 90, 150, 220 GHz respectively
w_Emv_websky = np.loadtxt('ilc_weights/weights1d_EE_spt3g_cmbmv.dat')
w_Bmv_websky = np.loadtxt('ilc_weights/weights1d_BB_spt3g_cmbmv.dat')
#================================================#

lmax = 4096
l = np.arange(0,lmax+1)

# Plot
plt.figure(0)
plt.clf()
#plt.grid(True, linestyle="--", alpha=0.5)
plt.axhline(y=0, color='gray', linestyle='--')
#plt.plot(l,w_Tmv_yuuki[0,:lmax+1],color='teal',linestyle='-',label='90 GHz T 2D')
#plt.plot(l,w_Emv_yuuki[0,:lmax+1],color='teal',linestyle='--',label='90 GHz P 2D')
#plt.plot(l,w_Tmv_yuuki[1,:lmax+1],color='orange',linestyle='-',label='150 GHz T 2D')
#plt.plot(l,w_Emv_yuuki[1,:lmax+1],color='orange',linestyle='--',label='150 GHz P 2D')
#plt.plot(l,w_Tmv_yuuki[2,:lmax+1],color='purple',linestyle='-',label='220 GHz T 2D')
#plt.plot(l,w_Emv_yuuki[2,:lmax+1],color='purple',linestyle='--',label='220 GHz P 2D')
#plt.plot(l,w_Tmv_ana[0,:lmax+1],color='teal',linestyle='--',label='90 GHz T, Ana')
#plt.plot(l,w_Tmv[0,:lmax+1],color='lightsteelblue',linestyle='-',label='90 GHz T')
#plt.plot(l,w_Emv[0,:lmax+1],color='lightsteelblue',linestyle='--',label='90 GHz P')
#plt.plot(l,w_Tmv_ana[1,:lmax+1],color='orange',linestyle='--',label='150 GHz T, Ana')
#plt.plot(l,w_Tmv[1,:lmax+1],color='bisque',linestyle='-',label='150 GHz T')
#plt.plot(l,w_Emv[1,:lmax+1],color='bisque',linestyle='--',label='150 GHz P')
#plt.plot(l,w_Tmv_ana[2,:lmax+1],color='purple',linestyle='--',label='220 GHz T, Ana')
#plt.plot(l,w_Tmv[2,:lmax+1],color='pink',linestyle='-',label='220 GHz T')
#plt.plot(l,w_Emv[2,:lmax+1],color='pink',linestyle='--',label='220 GHz P')
# tSZ-NULL
plt.plot(l,w_tsz_null_yuuki[0,:lmax+1],color='paleturquoise',linestyle='-',label='90 GHz T, 2D sliced at m = 500, 19/20')
plt.plot(l,w_tsz_null_ana[0,:lmax+1],color='teal',linestyle='--',label='90 GHz T, Ana')
plt.plot(l,w_tsz_null_fullskytest[0,:lmax+1],color='darkblue',linestyle='--',alpha=0.5,label='90 GHz T, used in full-sky test')
plt.plot(l,w_tsz_null[0,:lmax+1],color='lightsteelblue',linestyle='-',label='90 GHz T, 1D, 19/20')
plt.plot(l,w_tsz_null_yuuki[1,:lmax+1],color='palegoldenrod',linestyle='-',label='150 GHz T, 2D sliced at m = 500, 19/20')
plt.plot(l,w_tsz_null_ana[1,:lmax+1],color='orange',linestyle='--',label='150 GHz T, Ana')
plt.plot(l,w_tsz_null_fullskytest[1,:lmax+1],color='sandybrown',linestyle='--',alpha=0.5,label='150 GHz T, used in full-sky test')
plt.plot(l,w_tsz_null[1,:lmax+1],color='bisque',linestyle='-',label='150 GHz T, 1D, 19/20')
plt.plot(l,w_tsz_null_yuuki[2,:lmax+1],color='lightcoral',linestyle='-',label='220 GHz T, 2D sliced at m = 500, 19/20')
plt.plot(l,w_tsz_null_ana[2,:lmax+1],color='purple',linestyle='--',label='220 GHz T, Ana')
plt.plot(l,w_tsz_null_fullskytest[2,:lmax+1],color='orchid',linestyle='--',alpha=0.5,label='220 GHz T, used in full-sky test')
plt.plot(l,w_tsz_null[2,:lmax+1],color='pink',linestyle='-',label='220 GHz T, 1D, 19/20')
# CIB-NULL
#plt.plot(l,w_cib_null_yuuki[0,:lmax+1],color='paleturquoise',linestyle='-',label='90 GHz T, 2D sliced at m = 500, 19/20')
#plt.plot(l,w_cib_null_ana[0,:lmax+1],color='teal',linestyle='--',label='90 GHz T, Ana')
#plt.plot(l,w_cib_null_twoseds[0,:lmax+1],color='darkblue',linestyle='--',alpha=0.5,label='90 GHz T, used in full-sky test')
#plt.plot(l,w_cib_null[0,:lmax+1],color='lightsteelblue',linestyle='-',label='90 GHz T, 1D, 19/20')
#plt.plot(l,w_cib_null_yuuki[1,:lmax+1],color='palegoldenrod',linestyle='-',label='150 GHz T, 2D sliced at m = 500, 19/20')
#plt.plot(l,w_cib_null_ana[1,:lmax+1],color='orange',linestyle='--',label='150 GHz T, Ana')
#plt.plot(l,w_cib_null_twoseds[1,:lmax+1],color='sandybrown',linestyle='--',alpha=0.5,label='150 GHz T, used in full-sky test')
#plt.plot(l,w_cib_null[1,:lmax+1],color='bisque',linestyle='-',label='150 GHz T, 1D, 19/20')
#plt.plot(l,w_cib_null_yuuki[2,:lmax+1],color='lightcoral',linestyle='-',label='220 GHz T, 2D sliced at m = 500, 19/20')
#plt.plot(l,w_cib_null_ana[2,:lmax+1],color='purple',linestyle='--',label='220 GHz T, Ana')
#plt.plot(l,w_cib_null_twoseds[2,:lmax+1],color='orchid',linestyle='--',alpha=0.5,label='220 GHz T, used in full-sky test')
#plt.plot(l,w_cib_null[2,:lmax+1],color='pink',linestyle='-',label='220 GHz T, 1D, 19/20')
plt.ylabel("$W_{\ell}$")
plt.xlabel('$\ell$')
plt.title(f'tSZ-null T ILC Weights',pad=10)
plt.legend(loc='upper right', fontsize='x-small')
plt.xlim(500,lmax)
#plt.ylim(-0.3,1.2)
#plt.ylim(-4,4)
plt.ylim(-5,10)
plt.tight_layout()
plt.savefig(dir_out+f'/figs/tszn_ilc_weights_spt3g_20192020.png',bbox_inches='tight')
