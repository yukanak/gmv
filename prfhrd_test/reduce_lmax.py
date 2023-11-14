#!/usr/bin/env python3
import healpy as hp
import sys
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import utils

sim = int(sys.argv[1])

lmax=4096
input_plm = hp.read_alm(f'/scratch/users/yukanaka/lensing19-20/inputcmb/phi/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_phi1_seed{sim}.alm')
input_plm = utils.reduce_lmax(input_plm, lmax)
hp.fitsfunc.write_alm(f'/scratch/users/yukanaka/lensing19-20/inputcmb/phi/phi_lmax_4096//planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_phi1_seed{sim}_lmax{lmax}.alm',input_plm)
