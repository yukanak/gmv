#!/usr/bin/env python3
import healpy as hp
import sys

sim = int(sys.argv[1])
lmax = 4096

t,q,u = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_95ghz_seed{sim}.fits',field=[0,1,2])
tlm,elm,blm = hp.map2alm([t,q,u],lmax=lmax)
hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_95ghz_seed{sim}_alm_lmax{lmax}.fits',[tlm,elm,blm])

t150,q150,u150 = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_150ghz_seed{sim}.fits',field=[0,1,2])
tlm150,elm150,blm150 = hp.map2alm([t150,q150,u150],lmax=lmax)
hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_150ghz_seed{sim}_alm_lmax{lmax}.fits',[tlm150,elm150,blm150])

t220,q220,u220 = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_220ghz_seed{sim}.fits',field=[0,1,2])
tlm220,elm220,blm220 = hp.map2alm([t220,q220,u220],lmax=lmax)
hp.fitsfunc.write_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_220ghz_seed{sim}_alm_lmax{lmax}.fits',[tlm220,elm220,blm220])
