#!/usr/bin/env python3
import healpy as hp
import sys

sim = int(sys.argv[1])

lmax = 4000
input_plm = hp.read_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/phi/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_phi1_seed{sim}.alm')
hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/phi/phi_lmax_{lmax}/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_phi1_seed{sim}_lmax{lmax}.alm', input_plm, lmax=lmax)

t,q,u = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/tqu1/len/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb1_seed{sim}_lmax17000_nside8192_interp1.6_method1_pol_1_lensedmap.fits',field=[0,1,2])
tlm,elm,blm = hp.map2alm([t,q,u],lmax=lmax)
hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/tqu1/len/alms/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb1_seed{sim}_alm_lmax{lmax}.fits',[tlm,elm,blm])

t,q,u = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/tqu2/len/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb2_seed{sim}_lmax17000_nside8192_interp1.6_method1_pol_1_lensedmap.fits',field=[0,1,2])
tlm,elm,blm = hp.map2alm([t,q,u],lmax=lmax)
hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/tqu2/len/alms/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb2_seed{sim}_alm_lmax{lmax}.fits',[tlm,elm,blm])
