#!/usr/bin/env python3
import healpy as hp
import sys

lmax = 4096
sim = int(sys.argv[1])

#t1,q1,u1 = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/tqu1/len/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb1_seed{sim}_lmax17000_nside8192_interp1.6_method1_pol_1_lensedmap.fits',field=[0,1,2])
#tlm1,elm1,blm1 = hp.map2alm([t1,q1,u1],lmax=lmax)
#hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/tqu1/len/alms/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb1_seed{sim}_alm_lmax{lmax}.fits',[tlm1,elm1,blm1])
#t2,q2,u2 = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/tqu2/len/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb2_seed{sim}_lmax17000_nside8192_interp1.6_method1_pol_1_lensedmap.fits',field=[0,1,2])
#tlm2,elm2,blm2 = hp.map2alm([t2,q2,u2],lmax=lmax)
#hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/tqu2/len/alms/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb2_seed{sim}_alm_lmax{lmax}.fits',[tlm2,elm2,blm2])

#phi = hp.read_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/phi/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_phi1_seed{sim}.alm')
#hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/phi/phi_lmax_{lmax}/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_phi1_seed{sim}_lmax{lmax}.alm', phi, lmax=lmax)

t150,q150,u150 = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_150ghz_seed{sim}.fits',field=[0,1,2])
tlm150,elm150,blm150 = hp.map2alm([t150,q150,u150],lmax=lmax)
hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_150ghz_seed{sim}_alm_lmax{lmax}.fits',[tlm150,elm150,blm150])
t220,q220,u220 = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_220ghz_seed{sim}.fits',field=[0,1,2])
tlm220,elm220,blm220 = hp.map2alm([t220,q220,u220],lmax=lmax)
hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_220ghz_seed{sim}_alm_lmax{lmax}.fits',[tlm220,elm220,blm220])
t95,q95,u95 = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_95ghz_seed{sim}.fits',field=[0,1,2])
tlm95,elm95,blm95 = hp.map2alm([t95,q95,u95],lmax=lmax)
hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_95ghz_seed{sim}_alm_lmax{lmax}.fits',[tlm95,elm95,blm95])
