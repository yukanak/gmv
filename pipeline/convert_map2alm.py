#!/usr/bin/env python3
import healpy as hp
import sys

# https://github.com/yomori/healqest/blob/master/pipeline/spt3g_20192020/emulator/v052425_v2/generate_maps.py is maybe interesting

#lmax = 4096
lmax = 9999
# Point source mask
mask = hp.read_map('/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mask8192_mdpl2_v0.7_spt3g_150ghz_lenmag_cibmap_radmap_fluxcut6.0mjy_singlepix.fits')

# Save foregrounds-only Agora
t95_lcib_lksz_lrad,q95_lcib_lksz_lrad,u95_lcib_lksz_lrad = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_95ghz_lcibNG_lkszNGbahamas80_lradNG_uk.fits',field=[0,1,2])
t95_ltsz = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_95ghz_ltszNGbahamas80_uk_diffusioninp.fits')
t95_lcib_lksz_lrad *= mask; q95_lcib_lksz_lrad *= mask; u95_lcib_lksz_lrad *= mask
t95 = t95_lcib_lksz_lrad + t95_ltsz; q95 = q95_lcib_lksz_lrad; u95 = u95_lcib_lksz_lrad
#t95 *= mask; q95 *= mask; u95 *= mask
tlm95,elm95,blm95 = hp.map2alm([t95,q95,u95],lmax=lmax)
hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lcib_lksz_lrad_ltsz_spt3g_95ghz_alm_lmax{lmax}.fits',[tlm95,elm95,blm95],overwrite=True)

t150_lcib_lksz_lrad,q150_lcib_lksz_lrad,u150_lcib_lksz_lrad = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_150ghz_lcibNG_lkszNGbahamas80_lradNG_uk.fits',field=[0,1,2])
t150_ltsz = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_150ghz_ltszNGbahamas80_uk_diffusioninp.fits')
t150_lcib_lksz_lrad *= mask; q150_lcib_lksz_lrad *= mask; u150_lcib_lksz_lrad *= mask
t150 = t150_lcib_lksz_lrad + t150_ltsz; q150 = q150_lcib_lksz_lrad; u150 = u150_lcib_lksz_lrad
#t150 *= mask; q150 *= mask; u150 *= mask
tlm150,elm150,blm150 = hp.map2alm([t150,q150,u150],lmax=lmax)
hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lcib_lksz_lrad_ltsz_spt3g_150ghz_alm_lmax{lmax}.fits',[tlm150,elm150,blm150],overwrite=True)

t220_lcib_lksz_lrad,q220_lcib_lksz_lrad,u220_lcib_lksz_lrad = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_220ghz_lcibNG_lkszNGbahamas80_lradNG_uk.fits',field=[0,1,2])
t220_ltsz = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_220ghz_ltszNGbahamas80_uk_diffusioninp.fits')
t220_lcib_lksz_lrad *= mask; q220_lcib_lksz_lrad *= mask; u220_lcib_lksz_lrad *= mask
t220 = t220_lcib_lksz_lrad + t220_ltsz; q220 = q220_lcib_lksz_lrad; u220 = u220_lcib_lksz_lrad
#t220 *= mask; q220 *= mask; u220 *= mask
tlm220,elm220,blm220 = hp.map2alm([t220,q220,u220],lmax=lmax)
hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lcib_lksz_lrad_ltsz_spt3g_220ghz_alm_lmax{lmax}.fits',[tlm220,elm220,blm220],overwrite=True)

'''
# Save lensed CMB-only Agora!!!
t_lcmb,q_lcmb,u_lcmb = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_agoraphiNG_scalep18_teb1_seed1001_v2_lmax17000_nside8192_interp1.6_method1_pol_1_lensedmap.fits',field=[0,1,2])
tlm_lcmb,elm_lcmb,blm_lcmb = hp.map2alm([t_lcmb,q_lcmb,u_lcmb],lmax=lmax) 
hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_lcmb_spt3g_95ghz_alm_lmax4096.fits',[tlm_lcmb,elm_lcmb,blm_lcmb],overwrite=True)
hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_lcmb_spt3g_150ghz_alm_lmax4096.fits',[tlm_lcmb,elm_lcmb,blm_lcmb],overwrite=True)
hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_lcmb_spt3g_220ghz_alm_lmax4096.fits',[tlm_lcmb,elm_lcmb,blm_lcmb],overwrite=True)
'''

'''
# Save lensed CMB + individual foreground components
t_lcmb,q_lcmb,u_lcmb = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_agoraphiNG_scalep18_teb1_seed1001_v2_lmax17000_nside8192_interp1.6_method1_pol_1_lensedmap.fits',field=[0,1,2])
# Lensed CMB + tSZ
t95_ltsz = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_95ghz_ltszNGbahamas80_uk_diffusioninp.fits')
t150_ltsz = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_150ghz_ltszNGbahamas80_uk_diffusioninp.fits')
t220_ltsz = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_220ghz_ltszNGbahamas80_uk_diffusioninp.fits')
#t95 = t_lcmb + t95_ltsz; q95 = q_lcmb; u95 = u_lcmb
#t150 = t_lcmb + t150_ltsz; q150 = q_lcmb; u150 = u_lcmb
#t220 = t_lcmb + t220_ltsz; q220 = q_lcmb; u220 = u_lcmb
#tlm95,elm95,blm95 = hp.map2alm([t95,q95,u95],lmax=lmax); tlm150,elm150,blm150 = hp.map2alm([t150,q150,u150],lmax=lmax); tlm220,elm220,blm220 = hp.map2alm([t220,q220,u220],lmax=lmax)
#hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lcmb_ltsz_spt3g_95ghz_alm_lmax{lmax}.fits',[tlm95,elm95,blm95],overwrite=True)
#hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lcmb_ltsz_spt3g_150ghz_alm_lmax{lmax}.fits',[tlm150,elm150,blm150],overwrite=True)
#hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lcmb_ltsz_spt3g_220ghz_alm_lmax{lmax}.fits',[tlm220,elm220,blm220],overwrite=True)
# Lensed tSZ
t95 = t95_ltsz; t150 = t150_ltsz; t220 = t220_ltsz
tlm95 = hp.map2alm(t95,lmax=lmax); tlm150 = hp.map2alm(t150,lmax=lmax); tlm220 = hp.map2alm(t220,lmax=lmax)
hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_ltsz_spt3g_95ghz_alm_lmax{lmax}.fits',tlm95,overwrite=True)
hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_ltsz_spt3g_150ghz_alm_lmax{lmax}.fits',tlm150,overwrite=True)
hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_ltsz_spt3g_220ghz_alm_lmax{lmax}.fits',tlm220,overwrite=True)
# Lensed CMB + CIB
t95_lcib,q95_lcib,u95_lcib = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_95ghz_lcibNG_uk.fits',field=[0,1,2])
t150_lcib,q150_lcib,u150_lcib = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_150ghz_lcibNG_uk.fits',field=[0,1,2])
t220_lcib,q220_lcib,u220_lcib = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_220ghz_lcibNG_uk.fits',field=[0,1,2])
t95_lcib *= mask; q95_lcib *= mask; u95_lcib *= mask; t150_lcib *= mask; q150_lcib *= mask; u150_lcib *= mask; t220_lcib *= mask; q220_lcib *= mask; u220_lcib *= mask;
#t95 = t_lcmb + t95_lcib; q95 = q_lcmb + q95_lcib; u95 = u_lcmb + u95_lcib
#t150 = t_lcmb + t150_lcib; q150 = q_lcmb + q150_lcib; u150 = u_lcmb + u150_lcib
#t220 = t_lcmb + t220_lcib; q220 = q_lcmb + q220_lcib; u220 = u_lcmb + u220_lcib
#tlm95,elm95,blm95 = hp.map2alm([t95,q95,u95],lmax=lmax); tlm150,elm150,blm150 = hp.map2alm([t150,q150,u150],lmax=lmax); tlm220,elm220,blm220 = hp.map2alm([t220,q220,u220],lmax=lmax)
#hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lcmb_lcib_spt3g_95ghz_alm_lmax{lmax}.fits',[tlm95,elm95,blm95],overwrite=True)
#hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lcmb_lcib_spt3g_150ghz_alm_lmax{lmax}.fits',[tlm150,elm150,blm150],overwrite=True)
#hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lcmb_lcib_spt3g_220ghz_alm_lmax{lmax}.fits',[tlm220,elm220,blm220],overwrite=True)
# Lensed CIB
t95 = t95_lcib; q95 = q95_lcib; u95 = u95_lcib
t150 = t150_lcib; q150 = q150_lcib; u150 = u150_lcib
t220 = t220_lcib; q220 = q220_lcib; u220 = u220_lcib
tlm95,elm95,blm95 = hp.map2alm([t95,q95,u95],lmax=lmax); tlm150,elm150,blm150 = hp.map2alm([t150,q150,u150],lmax=lmax); tlm220,elm220,blm220 = hp.map2alm([t220,q220,u220],lmax=lmax)
hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lcib_spt3g_95ghz_alm_lmax{lmax}.fits',[tlm95,elm95,blm95],overwrite=True)
hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lcib_spt3g_150ghz_alm_lmax{lmax}.fits',[tlm150,elm150,blm150],overwrite=True)
hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lcib_spt3g_220ghz_alm_lmax{lmax}.fits',[tlm220,elm220,blm220],overwrite=True)
# Lensed CMB + kSZ
t95_lksz,q95_lksz,u95_lksz = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_95ghz_lkszNGbahamas80_uk.fits',field=[0,1,2])
t150_lksz,q150_lksz,u150_lksz = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_150ghz_lkszNGbahamas80_uk.fits',field=[0,1,2])
t220_lksz,q220_lksz,u220_lksz = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_220ghz_lkszNGbahamas80_uk.fits',field=[0,1,2])
t95_lksz *= mask; q95_lksz *= mask; u95_lksz *= mask; t150_lksz *= mask; q150_lksz *= mask; u150_lksz *= mask; t220_lksz *= mask; q220_lksz *= mask; u220_lksz *= mask;
#t95 = t_lcmb + t95_lksz; q95 = q_lcmb + q95_lksz; u95 = u_lcmb + u95_lksz
#t150 = t_lcmb + t150_lksz; q150 = q_lcmb + q150_lksz; u150 = u_lcmb + u150_lksz
#t220 = t_lcmb + t220_lksz; q220 = q_lcmb + q220_lksz; u220 = u_lcmb + u220_lksz
#tlm95,elm95,blm95 = hp.map2alm([t95,q95,u95],lmax=lmax); tlm150,elm150,blm150 = hp.map2alm([t150,q150,u150],lmax=lmax); tlm220,elm220,blm220 = hp.map2alm([t220,q220,u220],lmax=lmax)
#hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lcmb_lksz_spt3g_95ghz_alm_lmax{lmax}.fits',[tlm95,elm95,blm95],overwrite=True)
#hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lcmb_lksz_spt3g_150ghz_alm_lmax{lmax}.fits',[tlm150,elm150,blm150],overwrite=True)
#hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lcmb_lksz_spt3g_220ghz_alm_lmax{lmax}.fits',[tlm220,elm220,blm220],overwrite=True)
# Lensed kSZ
t95 = t95_lksz; q95 = q95_lksz; u95 = u95_lksz
t150 = t150_lksz; q150 = q150_lksz; u150 = u150_lksz
t220 = t220_lksz; q220 = q220_lksz; u220 = u220_lksz
tlm95,elm95,blm95 = hp.map2alm([t95,q95,u95],lmax=lmax); tlm150,elm150,blm150 = hp.map2alm([t150,q150,u150],lmax=lmax); tlm220,elm220,blm220 = hp.map2alm([t220,q220,u220],lmax=lmax)
hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lksz_spt3g_95ghz_alm_lmax{lmax}.fits',[tlm95,elm95,blm95],overwrite=True)
hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lksz_spt3g_150ghz_alm_lmax{lmax}.fits',[tlm150,elm150,blm150],overwrite=True)
hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lksz_spt3g_220ghz_alm_lmax{lmax}.fits',[tlm220,elm220,blm220],overwrite=True)
# Lensed CMB + radio
t95_lrad,q95_lrad,u95_lrad = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_95ghz_lradNG_uk.fits',field=[0,1,2])
t150_lrad,q150_lrad,u150_lrad = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_150ghz_lradNG_uk.fits',field=[0,1,2])
t220_lrad,q220_lrad,u220_lrad = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_220ghz_lradNG_uk.fits',field=[0,1,2])
t95_lrad *= mask; q95_lrad *= mask; u95_lrad *= mask; t150_lrad *= mask; q150_lrad *= mask; u150_lrad *= mask; t220_lrad *= mask; q220_lrad *= mask; u220_lrad *= mask;
#t95 = t_lcmb + t95_lrad; q95 = q_lcmb + q95_lrad; u95 = u_lcmb + u95_lrad
#t150 = t_lcmb + t150_lrad; q150 = q_lcmb + q150_lrad; u150 = u_lcmb + u150_lrad
#t220 = t_lcmb + t220_lrad; q220 = q_lcmb + q220_lrad; u220 = u_lcmb + u220_lrad
#tlm95,elm95,blm95 = hp.map2alm([t95,q95,u95],lmax=lmax); tlm150,elm150,blm150 = hp.map2alm([t150,q150,u150],lmax=lmax); tlm220,elm220,blm220 = hp.map2alm([t220,q220,u220],lmax=lmax)
#hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lcmb_lrad_spt3g_95ghz_alm_lmax{lmax}.fits',[tlm95,elm95,blm95],overwrite=True)
#hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lcmb_lrad_spt3g_150ghz_alm_lmax{lmax}.fits',[tlm150,elm150,blm150],overwrite=True)
#hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lcmb_lrad_spt3g_220ghz_alm_lmax{lmax}.fits',[tlm220,elm220,blm220],overwrite=True)
# Lensed radio
t95 = t95_lrad; q95 = q95_lrad; u95 = u95_lrad
t150 = t150_lrad; q150 = q150_lrad; u150 = u150_lrad
t220 = t220_lrad; q220 = q220_lrad; u220 = u220_lrad
tlm95,elm95,blm95 = hp.map2alm([t95,q95,u95],lmax=lmax); tlm150,elm150,blm150 = hp.map2alm([t150,q150,u150],lmax=lmax); tlm220,elm220,blm220 = hp.map2alm([t220,q220,u220],lmax=lmax)
hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lrad_spt3g_95ghz_alm_lmax{lmax}.fits',[tlm95,elm95,blm95],overwrite=True)
hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lrad_spt3g_150ghz_alm_lmax{lmax}.fits',[tlm150,elm150,blm150],overwrite=True)
hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lrad_spt3g_220ghz_alm_lmax{lmax}.fits',[tlm220,elm220,blm220],overwrite=True)
# Lensed CMB + tSZ + CIB
t95_ltsz = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_95ghz_ltszNGbahamas80_uk_diffusioninp.fits')
t150_ltsz = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_150ghz_ltszNGbahamas80_uk_diffusioninp.fits')
t220_ltsz = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_220ghz_ltszNGbahamas80_uk_diffusioninp.fits')
t95_lcib,q95_lcib,u95_lcib = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_95ghz_lcibNG_uk.fits',field=[0,1,2])
t150_lcib,q150_lcib,u150_lcib = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_150ghz_lcibNG_uk.fits',field=[0,1,2])
t220_lcib,q220_lcib,u220_lcib = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_220ghz_lcibNG_uk.fits',field=[0,1,2])
t95_lcib *= mask; q95_lcib *= mask; u95_lcib *= mask; t150_lcib *= mask; q150_lcib *= mask; u150_lcib *= mask; t220_lcib *= mask; q220_lcib *= mask; u220_lcib *= mask;
t95 = t_lcmb + t95_lcib + t95_ltsz; q95 = q_lcmb + q95_lcib; u95 = u_lcmb + u95_lcib
t150 = t_lcmb + t150_lcib + t150_ltsz; q150 = q_lcmb + q150_lcib; u150 = u_lcmb + u150_lcib
t220 = t_lcmb + t220_lcib + t220_ltsz; q220 = q_lcmb + q220_lcib; u220 = u_lcmb + u220_lcib
tlm95,elm95,blm95 = hp.map2alm([t95,q95,u95],lmax=lmax); tlm150,elm150,blm150 = hp.map2alm([t150,q150,u150],lmax=lmax); tlm220,elm220,blm220 = hp.map2alm([t220,q220,u220],lmax=lmax)
hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lcmb_ltsz_lcib_spt3g_95ghz_alm_lmax{lmax}.fits',[tlm95,elm95,blm95],overwrite=True)
hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lcmb_ltsz_lcib_spt3g_150ghz_alm_lmax{lmax}.fits',[tlm150,elm150,blm150],overwrite=True)
hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lcmb_ltsz_lcib_spt3g_220ghz_alm_lmax{lmax}.fits',[tlm220,elm220,blm220],overwrite=True)
# Lensed tSZ + CIB
t95 = t95_lcib + t95_ltsz; q95 = q95_lcib; u95 = u95_lcib
t150 = t150_lcib + t150_ltsz; q150 = q150_lcib; u150 = u150_lcib
t220 = t220_lcib + t220_ltsz; q220 = q220_lcib; u220 = u220_lcib
tlm95,elm95,blm95 = hp.map2alm([t95,q95,u95],lmax=lmax); tlm150,elm150,blm150 = hp.map2alm([t150,q150,u150],lmax=lmax); tlm220,elm220,blm220 = hp.map2alm([t220,q220,u220],lmax=lmax)
hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_ltsz_lcib_spt3g_95ghz_alm_lmax{lmax}.fits',[tlm95,elm95,blm95],overwrite=True)
hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_ltsz_lcib_spt3g_150ghz_alm_lmax{lmax}.fits',[tlm150,elm150,blm150],overwrite=True)
hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_ltsz_lcib_spt3g_220ghz_alm_lmax{lmax}.fits',[tlm220,elm220,blm220],overwrite=True)
'''

'''
# Save total Agora
t95_lcmb,q95_lcmb,u95_lcmb = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_agoraphiNG_scalep18_teb1_seed1001_v2_lmax17000_nside8192_interp1.6_method1_pol_1_lensedmap.fits',field=[0,1,2])
t95_lcib_lksz_lrad,q95_lcib_lksz_lrad,u95_lcib_lksz_lrad = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_95ghz_lcibNG_lkszNGbahamas80_lradNG_uk.fits',field=[0,1,2])
t95_ltsz = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_95ghz_ltszNGbahamas80_uk_diffusioninp.fits')
t95_lcib_lksz_lrad *= mask; q95_lcib_lksz_lrad *= mask; u95_lcib_lksz_lrad *= mask
t95 = t95_lcmb + t95_lcib_lksz_lrad + t95_ltsz; q95 = q95_lcmb + q95_lcib_lksz_lrad; u95 = u95_lcmb + u95_lcib_lksz_lrad
#t95 *= mask; q95 *= mask; u95 *= mask
#hp.fitsfunc.write_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_95ghz_map_total_masked.fits',[t95,q95,u95])
tlm95,elm95,blm95 = hp.map2alm([t95,q95,u95],lmax=lmax)
hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_95ghz_alm_lmax{lmax}.fits',[tlm95,elm95,blm95],overwrite=True)

t150_lcmb,q150_lcmb,u150_lcmb = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_agoraphiNG_scalep18_teb1_seed1001_v2_lmax17000_nside8192_interp1.6_method1_pol_1_lensedmap.fits',field=[0,1,2])
t150_lcib_lksz_lrad,q150_lcib_lksz_lrad,u150_lcib_lksz_lrad = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_150ghz_lcibNG_lkszNGbahamas80_lradNG_uk.fits',field=[0,1,2])
t150_ltsz = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_150ghz_ltszNGbahamas80_uk_diffusioninp.fits')
t150_lcib_lksz_lrad *= mask; q150_lcib_lksz_lrad *= mask; u150_lcib_lksz_lrad *= mask
t150 = t150_lcmb + t150_lcib_lksz_lrad + t150_ltsz; q150 = q150_lcmb + q150_lcib_lksz_lrad; u150 = u150_lcmb + u150_lcib_lksz_lrad
#t150 *= mask; q150 *= mask; u150 *= mask
#hp.fitsfunc.write_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_150ghz_map_total_masked.fits',[t150,q150,u150])
tlm150,elm150,blm150 = hp.map2alm([t150,q150,u150],lmax=lmax)
hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_150ghz_alm_lmax{lmax}.fits',[tlm150,elm150,blm150],overwrite=True)

t220_lcmb,q220_lcmb,u220_lcmb = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_agoraphiNG_scalep18_teb1_seed1001_v2_lmax17000_nside8192_interp1.6_method1_pol_1_lensedmap.fits',field=[0,1,2])
t220_lcib_lksz_lrad,q220_lcib_lksz_lrad,u220_lcib_lksz_lrad = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_220ghz_lcibNG_lkszNGbahamas80_lradNG_uk.fits',field=[0,1,2])
t220_ltsz = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_220ghz_ltszNGbahamas80_uk_diffusioninp.fits')
t220_lcib_lksz_lrad *= mask; q220_lcib_lksz_lrad *= mask; u220_lcib_lksz_lrad *= mask
t220 = t220_lcmb + t220_lcib_lksz_lrad + t220_ltsz; q220 = q220_lcmb + q220_lcib_lksz_lrad; u220 = u220_lcmb + u220_lcib_lksz_lrad
#t220 *= mask; q220 *= mask; u220 *= mask
#hp.fitsfunc.write_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_220ghz_map_total_masked.fits',[t220,q220,u220])
tlm220,elm220,blm220 = hp.map2alm([t220,q220,u220],lmax=lmax)
hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_220ghz_alm_lmax{lmax}.fits',[tlm220,elm220,blm220],overwrite=True)
'''

