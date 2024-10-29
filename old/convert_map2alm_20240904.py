#!/usr/bin/env python3
import healpy as hp
import sys

lmax = 4096
# Mask
mask = hp.read_map('/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mask8192_mdpl2_v0.7_spt3g_150ghz_lenmag_cibmap_radmap_fluxcut6.0mjy_singlepix.fits')

#t95_lcmb,q95_lcmb,u95_lcmb = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_95ghz_lcmbNG_uk.fits',field=[0,1,2])
#t95_lcib,q95_lcib,u95_lcib = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_95ghz_lcibNG_uk.fits',field=[0,1,2])
##t95_lksz,q95_lksz,u95_lksz = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_95ghz_lkszNGbahamas80_uk.fits',field=[0,1,2])
#t95_ltsz = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_95ghz_ltszNGbahamas80_uk_diffusioninp.fits')
##t95 = t95_lcmb + t95_lcib + t95_lksz + t95_ltsz; q95 = q95_lcmb + q95_lcib + q95_lksz; u95 = u95_lcmb + u95_lcib + u95_lksz
#t95 = t95_lcmb + t95_lcib + t95_ltsz; q95 = q95_lcmb + q95_lcib; u95 = u95_lcmb + u95_lcib
#t95 *= mask; q95 *= mask; u95 *= mask
##hp.fitsfunc.write_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_95ghz_map_nolrad_masked.fits',[t95,q95,u95])
#tlm95,elm95,blm95 = hp.map2alm([t95,q95,u95],lmax=lmax)
##hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_95ghz_alm_lmax{lmax}_nolrad.fits',[tlm95,elm95,blm95])
#hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_95ghz_alm_lmax{lmax}_nolradlksz.fits',[tlm95,elm95,blm95])
#
#t150_lcmb,q150_lcmb,u150_lcmb = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_150ghz_lcmbNG_uk.fits',field=[0,1,2])
#t150_lcib,q150_lcib,u150_lcib = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_150ghz_lcibNG_uk.fits',field=[0,1,2])
##t150_lksz,q150_lksz,u150_lksz = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_150ghz_lkszNGbahamas80_uk.fits',field=[0,1,2])
#t150_ltsz = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_150ghz_ltszNGbahamas80_uk_diffusioninp.fits')
##t150 = t150_lcmb + t150_lcib + t150_lksz + t150_ltsz; q150 = q150_lcmb + q150_lcib + q150_lksz; u150 = u150_lcmb + u150_lcib + u150_lksz
#t150 = t150_lcmb + t150_lcib + t150_ltsz; q150 = q150_lcmb + q150_lcib; u150 = u150_lcmb + u150_lcib
#t150 *= mask; q150 *= mask; u150 *= mask
##hp.fitsfunc.write_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_150ghz_map_nolrad_masked.fits',[t150,q150,u150])
#tlm150,elm150,blm150 = hp.map2alm([t150,q150,u150],lmax=lmax)
##hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_150ghz_alm_lmax{lmax}_nolrad.fits',[tlm150,elm150,blm150])
#hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_150ghz_alm_lmax{lmax}_nolradlksz.fits',[tlm150,elm150,blm150])

t220_lcmb,q220_lcmb,u220_lcmb = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_220ghz_lcmbNG_uk.fits',field=[0,1,2])
t220_lcib,q220_lcib,u220_lcib = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_220ghz_lcibNG_uk.fits',field=[0,1,2])
#t220_lksz,q220_lksz,u220_lksz = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_220ghz_lkszNGbahamas80_uk.fits',field=[0,1,2])
t220_ltsz = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_220ghz_ltszNGbahamas80_uk_diffusioninp.fits')
#t220 = t220_lcmb + t220_lcib + t220_lksz + t220_ltsz; q220 = q220_lcmb + q220_lcib + q220_lksz; u220 = u220_lcmb + u220_lcib + u220_lksz
t220 = t220_lcmb + t220_lcib + t220_ltsz; q220 = q220_lcmb + q220_lcib; u220 = u220_lcmb + u220_lcib
t220 *= mask; q220 *= mask; u220 *= mask
#hp.fitsfunc.write_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_220ghz_map_nolrad_masked.fits',[t220,q220,u220])
tlm220,elm220,blm220 = hp.map2alm([t220,q220,u220],lmax=lmax)
#hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_220ghz_alm_lmax{lmax}_nolrad.fits',[tlm220,elm220,blm220])
hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_220ghz_alm_lmax{lmax}_nolradlksz.fits',[tlm220,elm220,blm220])

'''
t95_lcmb,q95_lcmb,u95_lcmb = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_95ghz_lcmbNG_uk.fits',field=[0,1,2])
t95_lcib_lksz_lrad,q95_lcib_lksz_lrad,u95_lcib_lksz_lrad = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_95ghz_lcibNG_lkszNGbahamas80_lradNG_uk.fits',field=[0,1,2])
t95_ltsz = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_95ghz_ltszNGbahamas80_uk_diffusioninp.fits')
t95 = t95_lcmb + t95_lcib_lksz_lrad + t95_ltsz; q95 = q95_lcmb + q95_lcib_lksz_lrad; u95 = u95_lcmb + u95_lcib_lksz_lrad
t95 *= mask; q95 *= mask; u95 *= mask
hp.fitsfunc.write_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_95ghz_map_total_masked.fits',[t95,q95,u95])
#tlm95,elm95,blm95 = hp.map2alm([t95,q95,u95],lmax=lmax)
#hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_95ghz_alm_lmax{lmax}.fits',[tlm95,elm95,blm95])

t150_lcmb,q150_lcmb,u150_lcmb = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_150ghz_lcmbNG_uk.fits',field=[0,1,2])
t150_lcib_lksz_lrad,q150_lcib_lksz_lrad,u150_lcib_lksz_lrad = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_150ghz_lcibNG_lkszNGbahamas80_lradNG_uk.fits',field=[0,1,2])
t150_ltsz = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_150ghz_ltszNGbahamas80_uk_diffusioninp.fits')
t150 = t150_lcmb + t150_lcib_lksz_lrad + t150_ltsz; q150 = q150_lcmb + q150_lcib_lksz_lrad; u150 = u150_lcmb + u150_lcib_lksz_lrad
t150 *= mask; q150 *= mask; u150 *= mask
hp.fitsfunc.write_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_150ghz_map_total_masked.fits',[t150,q150,u150])
#tlm150,elm150,blm150 = hp.map2alm([t150,q150,u150],lmax=lmax)
#hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_150ghz_alm_lmax{lmax}.fits',[tlm150,elm150,blm150])

t220_lcmb,q220_lcmb,u220_lcmb = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_220ghz_lcmbNG_uk.fits',field=[0,1,2])
t220_lcib_lksz_lrad,q220_lcib_lksz_lrad,u220_lcib_lksz_lrad = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_220ghz_lcibNG_lkszNGbahamas80_lradNG_uk.fits',field=[0,1,2])
t220_ltsz = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_220ghz_ltszNGbahamas80_uk_diffusioninp.fits')
t220 = t220_lcmb + t220_lcib_lksz_lrad + t220_ltsz; q220 = q220_lcmb + q220_lcib_lksz_lrad; u220 = u220_lcmb + u220_lcib_lksz_lrad
t220 *= mask; q220 *= mask; u220 *= mask
hp.fitsfunc.write_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_220ghz_map_total_masked.fits',[t220,q220,u220])
#tlm220,elm220,blm220 = hp.map2alm([t220,q220,u220],lmax=lmax)
#hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_220ghz_alm_lmax{lmax}.fits',[tlm220,elm220,blm220])
'''
