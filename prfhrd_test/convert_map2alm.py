#!/usr/bin/env python3
import healpy as hp
import sys

sim = int(sys.argv[1])

lmax=4096
#input_phi = hp.read_map(f'/scratch/users/yukanaka/input_phi1/phi_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_seed{sim}.fits')
#input_phi = hp.pixelfunc.ud_grade(input_phi,8192)
t,q,u = hp.read_map(f'/scratch/users/yukanaka/lensing19-20/inputcmb/tqu2/len/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb2_seed{sim}_lmax17000_nside8192_interp1.6_method1_pol_1_lensedmap.fits',field=[0,1,2])
tlm,elm,blm = hp.map2alm([t,q,u],lmax=lmax)
#hp.fitsfunc.write_alm(f'/scratch/users/yukanaka/input_phi1/phi_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_seed{sim}_alm_lmax{lmax}.fits',input_plm)
hp.fitsfunc.write_alm(f'/scratch/users/yukanaka/lensing19-20/inputcmb/tqu2/len/alms/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb2_seed{sim}_alm_lmax{lmax}.fits',[tlm,elm,blm])
