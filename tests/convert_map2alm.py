#!/usr/bin/env python3
import healpy as hp
import sys

lmax = 4096

#t95,q95,u95 = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_spt3g_95ghz_lcmbNG_lcibNG_ltszNGbahamas80_lkszNGbahamas80_lradNG_tszdiffinp_ptsrcsinglepixmasked_uk.fits',field=[0,1,2])
t95,q95,u95 = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/mdpl2_spt3g_95ghz_lcmbNG_uk.fits',field=[0,1,2])
tlm95,elm95,blm95 = hp.map2alm([t95,q95,u95],lmax=lmax)
#hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_spt3g_95ghz_alm_lmax{lmax}.fits',[tlm95,elm95,blm95])
hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_lcmb_spt3g_95ghz_alm_lmax{lmax}.fits',[tlm95,elm95,blm95])

#t150,q150,u150 = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_spt3g_150ghz_lcmbNG_lcibNG_ltszNGbahamas80_lkszNGbahamas80_lradNG_tszdiffinp_ptsrcsinglepixmasked_uk.fits',field=[0,1,2])
t150,q150,u150 = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/mdpl2_spt3g_150ghz_lcmbNG_uk.fits',field=[0,1,2])
tlm150,elm150,blm150 = hp.map2alm([t150,q150,u150],lmax=lmax)
#hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_spt3g_150ghz_alm_lmax{lmax}.fits',[tlm150,elm150,blm150])
hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_lcmb_spt3g_150ghz_alm_lmax{lmax}.fits',[tlm150,elm150,blm150])

#t220,q220,u220 = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_spt3g_220ghz_lcmbNG_lcibNG_ltszNGbahamas80_lkszNGbahamas80_lradNG_tszdiffinp_ptsrcsinglepixmasked_uk.fits',field=[0,1,2])
t220,q220,u220 = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/mdpl2_spt3g_220ghz_lcmbNG_uk.fits',field=[0,1,2])
tlm220,elm220,blm220 = hp.map2alm([t220,q220,u220],lmax=lmax)
#hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_spt3g_220ghz_alm_lmax{lmax}.fits',[tlm220,elm220,blm220])
hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_lcmb_spt3g_220ghz_alm_lmax{lmax}.fits',[tlm220,elm220,blm220])

t,q,u = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/mdpl2_spt3g_150ghz_lcibNG_uk.fits',field=[0,1,2])
tlm,elm,blm = hp.map2alm([t,q,u],lmax=lmax)
hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_lcib_spt3g_150ghz_alm_lmax{lmax}.fits',[tlm,elm,blm])

t,q,u = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/mdpl2_spt3g_150ghz_ltszNGbahamas80_uk.fits',field=[0,1,2])
tlm,elm,blm = hp.map2alm([t,q,u],lmax=lmax)
hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_ltsz_spt3g_150ghz_alm_lmax{lmax}.fits',[tlm,elm,blm])

#input_kappa = hp.read_map('/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_raytrace16384_cmbkappa_highzadded_lowLcorrected.fits')
#klm = hp.map2alm(input_kappa,lmax=lmax)
#hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_spt3g_input_klm_lmax{lmax}.fits',klm)
