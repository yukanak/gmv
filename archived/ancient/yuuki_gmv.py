#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import qest_original
import healpy as hp
import sys
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')                  
import utils

clfile = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat'
lmax = 4096
nside = 8192
fwhm = 1
nlev_t = 5
nlev_p = 5
sim = 1
file_map = f'/scratch/users/yukanaka/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb1_seed{sim}_lmax17000_nside8192_interp1.6_method1_pol_1_lensedmap.fits'
dir_out = '/scratch/users/yukanaka/gmv/'
append = 'gmv_yuuki'

ell,sltt,slee,slbb,slte = utils.get_lensedcls(clfile,lmax=lmax)
bl = hp.gauss_beam(fwhm=fwhm*0.00029088,lmax=lmax)                              
nltt = (np.pi/180./60.*nlev_t)**2 / bl**2                                       
nlee = (np.pi/180./60.*nlev_p)**2 / bl**2                                       
nlbb = (np.pi/180./60.*nlev_p)**2 / bl**2                                       
nlte = 0
tlm1 = tlm2 = np.load(f'/scratch/users/yukanaka/gmv/almbar_pre_cinv_filt/tlm_seed{sim}_lmax{lmax}_nside{nside}_20230123.npy')
elm1 = elm2 = np.load(f'/scratch/users/yukanaka/gmv/almbar_pre_cinv_filt/elm_seed{sim}_lmax{lmax}_nside{nside}_20230123.npy')
blm1 = blm2 = np.load(f'/scratch/users/yukanaka/gmv/almbar_pre_cinv_filt/blm_seed{sim}_lmax{lmax}_nside{nside}_20230123.npy')
cltt = sltt + nltt
clee = slee + nlee
clbb = slbb + nlbb
clte = slte + nlte
dl = cltt*clee - clte**2

glmgmv = 0
glmtt  = 0
glmpp  = 0

glmtt1,clmtt1 = qest_original.qest('TT', lmax, clfile, hp.almxfl(tlm1,clee/dl), hp.almxfl(tlm2,clee/dl) )
glmtt2,clmtt2 = qest_original.qest('EE', lmax, clfile, hp.almxfl(tlm1,clte/dl), hp.almxfl(tlm2,clte/dl) )
glmtt3,clmtt3 = qest_original.qest('TE', lmax, clfile, hp.almxfl(tlm1,clee/dl), hp.almxfl(tlm2,clte/dl) )
glmtt4,clmtt4 = qest_original.qest('ET', lmax, clfile, hp.almxfl(tlm1,clte/dl), hp.almxfl(tlm2,clee/dl) )
glmgmv += 0.5*(glmtt1+glmtt2-glmtt3-glmtt4)
glmtt  += 0.5*(glmtt1+glmtt2-glmtt3-glmtt4)

glmee1,clmee1 = qest_original.qest('TT', lmax, clfile, hp.almxfl(elm1,clte/dl), hp.almxfl(elm2,clte/dl) )
glmee2,clmee2 = qest_original.qest('EE', lmax, clfile, hp.almxfl(elm1,cltt/dl), hp.almxfl(elm2,cltt/dl) )
glmee3,clmee3 = qest_original.qest('TE', lmax, clfile, hp.almxfl(elm1,clte/dl), hp.almxfl(elm2,cltt/dl) )
glmee4,clmee4 = qest_original.qest('ET', lmax, clfile, hp.almxfl(elm1,cltt/dl), hp.almxfl(elm2,clte/dl) )
glmgmv += 0.5*(glmee1+glmee2-glmee3-glmee4)
glmtt  += 0.5*(glmee1+glmee2-glmee3-glmee4)

#glmte1,clmte1 = qest_original.qest('TT', lmax, clfile, hp.almxfl(tlm,slte/dl), hp.almxfl(elm,slee/dl) ) # TYPO slte<-> slee
glmte1,clmte1 = qest_original.qest('TT', lmax, clfile, hp.almxfl(tlm1,clee/dl), hp.almxfl(elm2,clte/dl) ) # FIXED
glmte2,clmte2 = qest_original.qest('EE', lmax, clfile, hp.almxfl(tlm1,clte/dl), hp.almxfl(elm2,cltt/dl) )
glmte3,clmte3 = qest_original.qest('TE', lmax, clfile, hp.almxfl(tlm1,clee/dl), hp.almxfl(elm2,cltt/dl) )
glmte4,clmte4 = qest_original.qest('ET', lmax, clfile, hp.almxfl(tlm1,clte/dl), hp.almxfl(elm2,clte/dl) )
glmte5,clmte5 = qest_original.qest('TT', lmax, clfile, hp.almxfl(elm1,clte/dl), hp.almxfl(tlm2,clee/dl) ) # FIXED
glmte6,clmte6 = qest_original.qest('EE', lmax, clfile, hp.almxfl(elm1,cltt/dl), hp.almxfl(tlm2,clte/dl) )
glmte7,clmte7 = qest_original.qest('TE', lmax, clfile, hp.almxfl(elm1,clte/dl), hp.almxfl(tlm2,clte/dl) )
glmte8,clmte8 = qest_original.qest('ET', lmax, clfile, hp.almxfl(elm1,cltt/dl), hp.almxfl(tlm2,clee/dl) )
glmgmv += 0.5*(-glmte1-glmte2+glmte3+glmte4-glmte5-glmte6+glmte7+glmte8)
glmtt  += 0.5*(-glmte1-glmte2+glmte3+glmte4-glmte5-glmte6+glmte7+glmte8)

glmtb1,clmtb1 = qest_original.qest('TB', lmax, clfile, hp.almxfl(tlm1,clee/dl), hp.almxfl(blm2,1/clbb) )
glmtb2,clmtb2 = qest_original.qest('EB', lmax, clfile, hp.almxfl(tlm1,clte/dl), hp.almxfl(blm2,1/clbb) )
glmtb3,clmtb3 = qest_original.qest('BT', lmax, clfile, hp.almxfl(blm1,1/clbb) , hp.almxfl(tlm2,clee/dl) )
glmtb4,clmtb4 = qest_original.qest('BE', lmax, clfile, hp.almxfl(blm1,1/clbb) , hp.almxfl(tlm2,clte/dl) )
glmgmv += 0.5*(glmtb1-glmtb2+glmtb3-glmtb4)
glmpp  += 0.5*(glmtb1-glmtb2+glmtb3-glmtb4)

glmeb1,clmeb1 = qest_original.qest('TB', lmax, clfile, hp.almxfl(elm1,clte/dl), hp.almxfl(blm2,1/clbb) )
glmeb2,clmeb2 = qest_original.qest('EB', lmax, clfile, hp.almxfl(elm1,cltt/dl), hp.almxfl(blm2,1/clbb) )
glmeb3,clmeb3 = qest_original.qest('BT', lmax, clfile, hp.almxfl(blm1,1/clbb) , hp.almxfl(elm2,clte/dl) )
glmeb4,clmeb4 = qest_original.qest('BE', lmax, clfile, hp.almxfl(blm1,1/clbb) , hp.almxfl(elm2,cltt/dl) )
glmgmv += 0.5*(-glmeb1+glmeb2-glmeb3+glmeb4)
glmpp  += 0.5*(-glmeb1+glmeb2-glmeb3+glmeb4)

np.save(dir_out+f'/plm_healqest_seed{sim}_lmax{lmax}_nside{nside}_{append}.npy',glmgmv)
np.save(dir_out+f'/plm_healqest_seed{sim}_lmax{lmax}_nside{nside}_{append}_tteete.npy',glmtt)
np.save(dir_out+f'/plm_healqest_seed{sim}_lmax{lmax}_nside{nside}_{append}_tbeb.npy',glmpp)
