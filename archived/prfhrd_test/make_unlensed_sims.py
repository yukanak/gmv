#!/usr/bin/env python3
import sys
import numpy as np
import healpy as hp
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import utils
import camb
from pathlib import Path
import os
import matplotlib.pyplot as plt
import wignerd
import resp

####################################
lmax = 4096 #5000
nside = 2048 #8192
cambini = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_params.ini'
dir_out = '/scratch/users/yukanaka/full_res_maps/unl_from_lensed_cls/'
clfile = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat'
####################################
ell,sltt,slee,slbb,slte = utils.get_lensedcls(clfile,lmax=lmax)
i = int(sys.argv[1]) 
print(f'Doing sim {i}...')
np.random.seed(hash('tora')%2**32+i)
t,q,u = hp.synfast((sltt,slte,slee,slbb),nside,lmax)
file_loc = dir_out + f'unl_from_lensed_cls_seed{i}_lmax{lmax}_nside{nside}_20230905.fits'
hp.fitsfunc.write_map(file_loc,[t,q,u])
