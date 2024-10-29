import os, sys
import numpy as np
import healpy as hp
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import healqest_utils as utils

config = utils.parse_yaml('test_yuka.yaml')
lmax = config['lensrec']['lmax']
nside = config['lensrec']['nside']
dir_out = config['dir_out']
lmaxT = config['lensrec']['lmaxT']
lmaxP = config['lensrec']['lmaxP']
lmin = config['lensrec']['lminT']
cltype = config['lensrec']['cltype']
cls = config['cls']
sl = {ee:config['cls'][cltype][ee] for ee in config['cls'][cltype].keys()}
l = np.arange(0,lmax+1)
klm = hp.read_alm('/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_spt3g_input_klm_lmax4096.fits')
input_clkk = hp.alm2cl(klm,klm,lmax=lmax)
clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
clkk = slpp * (l*(l+1))**2/4
# Agora sims (TOTAL, CMB + foregrounds)
#agora_095 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_spt3g_95ghz_alm_lmax4096.fits'
#agora_150 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_spt3g_150ghz_alm_lmax4096.fits'
#agora_220 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_spt3g_220ghz_alm_lmax4096.fits'
agora_095 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_95ghz_alm_lmax4096.fits'
agora_150 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_150ghz_alm_lmax4096.fits'
agora_220 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_220ghz_alm_lmax4096.fits'
# Lensed CMB-only Agora sims
#lcmb_095 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_lcmb_spt3g_95ghz_alm_lmax4096.fits'
#lcmb_150 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_lcmb_spt3g_150ghz_alm_lmax4096.fits'
#lcmb_220 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_lcmb_spt3g_220ghz_alm_lmax4096.fits'
lcmb_095 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lcmb_spt3g_95ghz_alm_lmax4096.fits'
lcmb_150 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lcmb_spt3g_150ghz_alm_lmax4096.fits'
lcmb_220 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lcmb_spt3g_220ghz_alm_lmax4096.fits'
# Gaussian full sky single frequency foreground sims
flm_95ghz_sim1 = f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_95ghz_seed1_alm_lmax{lmax}.fits'
flm_150ghz_sim1 = f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_150ghz_seed1_alm_lmax{lmax}.fits'
flm_220ghz_sim1 = f'/oak/stanford/orgs/kipac/users/yukanaka/fg/totfg_220ghz_seed1_alm_lmax{lmax}.fits'
# Gaussian CMB maps
alm_cmb_sim1 = f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/tqu1/len/alms/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb1_seed1_alm_lmax{lmax}.fits'
# ILC weights
# Dimension (3, 6001) for 90, 150, 220 GHz respectively
w_tsz_null = np.loadtxt('ilc_weights/weights1d_TT_spt3g_cmbynull.dat')
w_Tmv = np.loadtxt('ilc_weights/weights1d_TT_spt3g_cmbmv.dat')
w_Emv = np.loadtxt('ilc_weights/weights1d_EE_spt3g_cmbmv.dat')
w_Bmv = np.loadtxt('ilc_weights/weights1d_BB_spt3g_cmbmv.dat')
# If lmaxT != lmaxP, we add artificial noise in TT for ell > lmaxT
artificial_noise = np.zeros(lmax+1)
artificial_noise[lmaxT+2:] = 1.e10

'''
# Debug RDN0
rdn0_filename = dir_out+f'/n0/rdn0_98simpairs_healqest_gmv_cinv_lmaxT3000_lmaxP4096_nside2048_standard_resp_from_sims.pkl'
rdn0 = pickle.load(open(rdn0_filename,'rb'))
rdn0_total = rdn0 * (l*(l+1))**2/4
n0_filename = dir_out+f'/n0/n0_98simpairs_healqest_gmv_cinv_lmaxT3000_lmaxP4096_nside2048_standard_resp_from_sims.pkl'
n0 = pickle.load(open(n0_filename,'rb'))
n0_total = n0['total'] * (l*(l+1))**2/4
n1_filename = dir_out+f'/n1/n1_98simpairs_healqest_gmv_cinv_lmaxT3000_lmaxP4096_nside2048_standard_resp_from_sims.pkl'
n1 = pickle.load(open(n1_filename,'rb'))
n1_total = n1['total'] * (l*(l+1))**2/4

# Plot
plt.clf()
plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')

plt.plot(l, rdn0_total, color='firebrick', alpha=0.8, linestyle='-',label='Standard GMV, RDN0')
plt.plot(l, n0_total, color='darkblue', alpha=0.8, linestyle='-',label='Standard GMV, Sim-Based N0')
plt.plot(l, n1_total, color='forestgreen', alpha=0.8, linestyle='-',label='Standard GMV, Sim-Based N1')

plt.ylabel("$[\ell(\ell+1)]^2$$N_0$ / 4 $[\mu K^2]$")
plt.xlabel('$\ell$')
plt.title(f'GMV Reconstruction Noise Comparison')
plt.legend(loc='upper left', fontsize='small')
plt.xscale('log')
plt.yscale('log')
plt.xlim(10,lmax)
#plt.xlim(10,2000)
#plt.ylim(1e-8,1e-6)
#plt.ylim(1e-8,2e-7)
plt.savefig(dir_out+f'/figs/n0_comparison_gmv_rdn0.png',bbox_inches='tight')

plt.clf()
ratio_n0_total = rdn0_total/n0_total
# Ratios with error bars
plt.axhline(y=1, color='k', linestyle='--')
plt.plot(l, ratio_n0_total, color='firebrick', alpha=0.8, linestyle='-',label='Ratio Standard GMV: RDN0 / Sim-Based')
plt.xlabel('$\ell$')
plt.title(f'GMV Reconstruction Noise Comparison')
plt.legend(loc='upper left', fontsize='small')
plt.xscale('log')
#plt.ylim(0.9,1.4)
plt.xlim(10,lmax)
plt.savefig(dir_out+f'/figs/n0_comparison_gmv_ratio_rdn0.png',bbox_inches='tight')
'''

'''
# Plot maps and show colorbar, see if the CMB realizations match for the "total" Agora map vs "lensed CMB only"
# Check because it seems like there is still CMB in my "foreground only" sims obtained from subtracting the "lensed CMB only" sim from the total
# Which will happen if the CMB realizations for the two don't match

# TOTAL
#t95,q95,u95 = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_spt3g_95ghz_lcmbNG_lcibNG_ltszNGbahamas80_lkszNGbahamas80_lradNG_tszdiffinp_ptsrcsinglepixmasked_uk.fits',field=[0,1,2])
#t150,q150,u150 = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_spt3g_150ghz_lcmbNG_lcibNG_ltszNGbahamas80_lkszNGbahamas80_lradNG_tszdiffinp_ptsrcsinglepixmasked_uk.fits',field=[0,1,2])
#t220,q220,u220 = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_spt3g_220ghz_lcmbNG_lcibNG_ltszNGbahamas80_lkszNGbahamas80_lradNG_tszdiffinp_ptsrcsinglepixmasked_uk.fits',field=[0,1,2])
#t95,q95,u95 = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_95ghz_lcmbNG_lcibNG_ltszNGbahamas80_lkszNGbahamas80_lradNG_uk.fits',field=[0,1,2])
#t150,q150,u150 = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_150ghz_lcmbNG_lcibNG_ltszNGbahamas80_lkszNGbahamas80_lradNG_uk.fits',field=[0,1,2])
#t220,q220,u220 = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_220ghz_lcmbNG_lcibNG_ltszNGbahamas80_lkszNGbahamas80_lradNG_uk.fits',field=[0,1,2])
t95,q95,u95 = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_95ghz_map_total_masked.fits',field=[0,1,2])
t150,q150,u150 = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_150ghz_map_total_masked.fits',field=[0,1,2])
t220,q220,u220 = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_220ghz_map_total_masked.fits',field=[0,1,2])

# LENSED CMB ONLY
#t95_lcmb,q95_lcmb,u95_lcmb = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/mdpl2_spt3g_95ghz_lcmbNG_uk.fits',field=[0,1,2])
#t150_lcmb,q150_lcmb,u150_lcmb = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/mdpl2_spt3g_150ghz_lcmbNG_uk.fits',field=[0,1,2])
#t220_lcmb,q220_lcmb,u220_lcmb = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/mdpl2_spt3g_220ghz_lcmbNG_uk.fits',field=[0,1,2])
t95_lcmb,q95_lcmb,u95_lcmb = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_95ghz_lcmbNG_uk.fits',field=[0,1,2])
t150_lcmb,q150_lcmb,u150_lcmb = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_150ghz_lcmbNG_uk.fits',field=[0,1,2])
t220_lcmb,q220_lcmb,u220_lcmb = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_220ghz_lcmbNG_uk.fits',field=[0,1,2])

# Plot
scale = 700
plt.figure(0)
plt.clf()
#hp.gnomview(phi,title='Reconstructed Kappa Map',rot=[0,-60],notext=True,xsize=300,ysize=300,reso=3,min=-1*scale*3,max=scale*3,cmap='RdBu_r')#,unit="uK")
hp.gnomview(t150,title='Agora Total 150 GHz T Map',rot=[0,-60],notext=True,xsize=300,ysize=300,reso=3,cmap='RdBu_r')#,unit="uK")
plt.savefig(dir_out+f'/figs/agora_t150_map.png',bbox_inches='tight')

plt.clf()
#hp.gnomview(source,title='Reconstructed Source Map',rot=[0,-60],notext=True,xsize=300,ysize=300,reso=3,min=-1*scale*3,max=scale*3,cmap='RdBu_r')
hp.gnomview(t150_lcmb,title='Agora Lensed CMB-Only 150 GHz T Map',rot=[0,-60],notext=True,xsize=300,ysize=300,reso=3,cmap='RdBu_r')
plt.savefig(dir_out+f'/figs/agora_t150_lcmb_map.png',bbox_inches='tight')

plt.clf()
#hp.gnomview(phi,title='Reconstructed Kappa Map',rot=[0,-60],notext=True,xsize=300,ysize=300,reso=3,min=-1*scale*3,max=scale*3,cmap='RdBu_r')#,unit="uK")
hp.gnomview(t150,title='Agora Total 150 GHz T Map',rot=[0,-60],notext=True,xsize=300,ysize=300,reso=3,min=-1*scale,max=scale,cmap='RdBu_r')#,unit="uK")
plt.savefig(dir_out+f'/figs/agora_t150_map_constrained.png',bbox_inches='tight')

plt.clf()
#hp.gnomview(source,title='Reconstructed Source Map',rot=[0,-60],notext=True,xsize=300,ysize=300,reso=3,min=-1*scale*3,max=scale*3,cmap='RdBu_r')
hp.gnomview(t150_lcmb,title='Agora Lensed CMB-Only 150 GHz T Map',rot=[0,-60],notext=True,xsize=300,ysize=300,reso=3,min=-1*scale,max=scale,cmap='RdBu_r')
plt.savefig(dir_out+f'/figs/agora_t150_lcmb_map_constrained.png',bbox_inches='tight')
'''

# Compare signal-only and foregrounds-only spectra
# Get Agora sim (signal + foregrounds)
print('Getting alms...')
tlm_95, elm_95, blm_95 = hp.read_alm(agora_095,hdu=[1,2,3])
tlm_150, elm_150, blm_150 = hp.read_alm(agora_150,hdu=[1,2,3])
tlm_220, elm_220, blm_220 = hp.read_alm(agora_220,hdu=[1,2,3])
# 10/16/2024: Adding noise too to make it comparable with total Gaussian spectra
nlm1_090_filename = dir_out + f'nlm/nlm_090_lmax{lmax}_seed1.alm'
nlm1_150_filename = dir_out + f'nlm/nlm_150_lmax{lmax}_seed1.alm'
nlm1_220_filename = dir_out + f'nlm/nlm_220_lmax{lmax}_seed1.alm'
nlmt_090,nlme_090,nlmb_090 = hp.read_alm(nlm1_090_filename,hdu=[1,2,3])
nlmt_150,nlme_150,nlmb_150 = hp.read_alm(nlm1_150_filename,hdu=[1,2,3])
nlmt_220,nlme_220,nlmb_220 = hp.read_alm(nlm1_220_filename,hdu=[1,2,3])
tlm_150 += nlmt_150; tlm_220 += nlmt_220; tlm_95 += nlmt_090
elm_150 += nlme_150; elm_220 += nlme_220; elm_95 += nlme_090
blm_150 += nlmb_150; blm_220 += nlmb_220; blm_95 += nlmb_090
tlm = hp.almxfl(tlm_95,w_Tmv[0][:lmax+1]) + hp.almxfl(tlm_150,w_Tmv[1][:lmax+1]) + hp.almxfl(tlm_220,w_Tmv[2][:lmax+1])
elm = hp.almxfl(elm_95,w_Emv[0][:lmax+1]) + hp.almxfl(elm_150,w_Emv[1][:lmax+1]) + hp.almxfl(elm_220,w_Emv[2][:lmax+1])
blm = hp.almxfl(blm_95,w_Bmv[0][:lmax+1]) + hp.almxfl(blm_150,w_Bmv[1][:lmax+1]) + hp.almxfl(blm_220,w_Bmv[2][:lmax+1])
# NOTE: AGORA TOTAL
#cltt_agora_tot = hp.alm2cl(tlm,tlm)
cltt_agora_tot = hp.alm2cl(tlm,tlm) + artificial_noise
clee_agora_tot = hp.alm2cl(elm,elm)
clbb_agora_tot = hp.alm2cl(blm,blm)
clte_agora_tot = hp.alm2cl(tlm,elm)

# Get Agora lensed CMB-only
tlm_lcmb_95, elm_lcmb_95, blm_lcmb_95 = hp.read_alm(lcmb_095,hdu=[1,2,3])
tlm_lcmb_150, elm_lcmb_150, blm_lcmb_150 = hp.read_alm(lcmb_150,hdu=[1,2,3])
tlm_lcmb_220, elm_lcmb_220, blm_lcmb_220 = hp.read_alm(lcmb_220,hdu=[1,2,3])
tlm_lcmb = hp.almxfl(tlm_lcmb_95,w_Tmv[0][:lmax+1]) + hp.almxfl(tlm_lcmb_150,w_Tmv[1][:lmax+1]) + hp.almxfl(tlm_lcmb_220,w_Tmv[2][:lmax+1])
elm_lcmb = hp.almxfl(elm_lcmb_95,w_Emv[0][:lmax+1]) + hp.almxfl(elm_lcmb_150,w_Emv[1][:lmax+1]) + hp.almxfl(elm_lcmb_220,w_Emv[2][:lmax+1])
blm_lcmb = hp.almxfl(blm_lcmb_95,w_Bmv[0][:lmax+1]) + hp.almxfl(blm_lcmb_150,w_Bmv[1][:lmax+1]) + hp.almxfl(blm_lcmb_220,w_Bmv[2][:lmax+1])
# NOTE: AGORA TOTAL x AGORA CMB
cltt_agora_tot_lcmb_cross = hp.alm2cl(tlm,tlm_lcmb)
clee_agora_tot_lcmb_cross = hp.alm2cl(elm,elm_lcmb)
clbb_agora_tot_lcmb_cross = hp.alm2cl(blm,blm_lcmb)
clte_agora_tot_lcmb_cross = hp.alm2cl(tlm,elm_lcmb)

tlm = hp.almxfl(tlm_lcmb_95,w_Tmv[0][:lmax+1]) + hp.almxfl(tlm_lcmb_150,w_Tmv[1][:lmax+1]) + hp.almxfl(tlm_lcmb_220,w_Tmv[2][:lmax+1])
elm = hp.almxfl(elm_lcmb_95,w_Emv[0][:lmax+1]) + hp.almxfl(elm_lcmb_150,w_Emv[1][:lmax+1]) + hp.almxfl(elm_lcmb_220,w_Emv[2][:lmax+1])
blm = hp.almxfl(blm_lcmb_95,w_Bmv[0][:lmax+1]) + hp.almxfl(blm_lcmb_150,w_Bmv[1][:lmax+1]) + hp.almxfl(blm_lcmb_220,w_Bmv[2][:lmax+1])
# NOTE: AGORA CMB
sltt_agora = hp.alm2cl(tlm,tlm) + artificial_noise
slee_agora = hp.alm2cl(elm,elm)
slbb_agora = hp.alm2cl(blm,blm)
slte_agora = hp.alm2cl(tlm,elm)

# Get Agora foreground-only
tlm_fg_95 = tlm_95 - tlm_lcmb_95; elm_fg_95 = elm_95 - elm_lcmb_95; blm_fg_95 = blm_95 - blm_lcmb_95
tlm_fg_150 = tlm_150 - tlm_lcmb_150; elm_fg_150 = elm_150 - elm_lcmb_150; blm_fg_150 = blm_150 - blm_lcmb_150
tlm_fg_220 = tlm_220 - tlm_lcmb_220; elm_fg_220 = elm_220 - elm_lcmb_220; blm_fg_220 = blm_220 - blm_lcmb_220
tlm = hp.almxfl(tlm_fg_95,w_Tmv[0][:lmax+1]) + hp.almxfl(tlm_fg_150,w_Tmv[1][:lmax+1]) + hp.almxfl(tlm_fg_220,w_Tmv[2][:lmax+1])
elm = hp.almxfl(elm_fg_95,w_Emv[0][:lmax+1]) + hp.almxfl(elm_fg_150,w_Emv[1][:lmax+1]) + hp.almxfl(elm_fg_220,w_Emv[2][:lmax+1])
blm = hp.almxfl(blm_fg_95,w_Bmv[0][:lmax+1]) + hp.almxfl(blm_fg_150,w_Bmv[1][:lmax+1]) + hp.almxfl(blm_fg_220,w_Bmv[2][:lmax+1])
# NOTE: AGORA FG
fltt_agora = hp.alm2cl(tlm,tlm)
flee_agora = hp.alm2cl(elm,elm)
flbb_agora = hp.alm2cl(blm,blm)
flte_agora = hp.alm2cl(tlm,elm)

# Rotate Agora CMB map
r = hp.Rotator(np.array([np.pi/2,np.pi/2,0]))
tlm_lcmb_95, elm_lcmb_95, blm_lcmb_95 = r.rotate_alm([tlm_lcmb_95, elm_lcmb_95, blm_lcmb_95])
tlm_lcmb_150, elm_lcmb_150, blm_lcmb_150 = r.rotate_alm([tlm_lcmb_150, elm_lcmb_150, blm_lcmb_150])
tlm_lcmb_220, elm_lcmb_220, blm_lcmb_220 = r.rotate_alm([tlm_lcmb_220, elm_lcmb_220, blm_lcmb_220])
tlm = hp.almxfl(tlm_lcmb_95,w_Tmv[0][:lmax+1]) + hp.almxfl(tlm_lcmb_150,w_Tmv[1][:lmax+1]) + hp.almxfl(tlm_lcmb_220,w_Tmv[2][:lmax+1])
elm = hp.almxfl(elm_lcmb_95,w_Emv[0][:lmax+1]) + hp.almxfl(elm_lcmb_150,w_Emv[1][:lmax+1]) + hp.almxfl(elm_lcmb_220,w_Emv[2][:lmax+1])
blm = hp.almxfl(blm_lcmb_95,w_Bmv[0][:lmax+1]) + hp.almxfl(blm_lcmb_150,w_Bmv[1][:lmax+1]) + hp.almxfl(blm_lcmb_220,w_Bmv[2][:lmax+1])
# NOTE: ROTATED AGORA CMB
sltt_agora_rot = hp.alm2cl(tlm,tlm) + artificial_noise
slee_agora_rot = hp.alm2cl(elm,elm)
slbb_agora_rot = hp.alm2cl(blm,blm)
slte_agora_rot = hp.alm2cl(tlm,elm)

# Unlensed Gaussian CMB maps with the same power spectra as the rotated Agora maps
[cltt_090_090,cltt_150_150,cltt_220_220,cltt_090_150,cltt_150_220,cltt_090_220] = hp.alm2cl([tlm_lcmb_95,tlm_lcmb_150,tlm_lcmb_220])
[clee_090_090,clee_150_150,clee_220_220,clee_090_150,clee_150_220,clee_090_220] = hp.alm2cl([elm_lcmb_95,elm_lcmb_150,elm_lcmb_220])
[clbb_090_090,clbb_150_150,clbb_220_220,clbb_090_150,clbb_150_220,clbb_090_220] = hp.alm2cl([blm_lcmb_95,blm_lcmb_150,blm_lcmb_220])
# Seed "A"
np.random.seed(4190002645)
tlm_lcmb_95,elm_lcmb_95,blm_lcmb_95 = hp.synalm([cltt_090_090,clee_090_090,clbb_090_090,cltt_090_090*0],new=True,lmax=lmax)
# Seed "A"
np.random.seed(4190002645)
cltt_T2a = np.nan_to_num((cltt_090_150)**2 / cltt_090_090); clee_T2a = np.nan_to_num((clee_090_150)**2 / clee_090_090); clbb_T2a = np.nan_to_num((clbb_090_150)**2 / clbb_090_090)
tlm_T2a,elm_T2a,blm_T2a = hp.synalm([cltt_T2a,clee_T2a,clbb_T2a,cltt_T2a*0],new=True,lmax=lmax)
# Seed "B"
np.random.seed(89052206)
cltt_T2b = cltt_150_150 - cltt_T2a; clee_T2b = clee_150_150 - clee_T2a; clbb_T2b = clbb_150_150 - clbb_T2a
tlm_T2b,elm_T2b,blm_T2b = hp.synalm([cltt_T2b,clee_T2b,clbb_T2b,cltt_T2b*0],new=True,lmax=lmax)
tlm_lcmb_150 = tlm_T2a + tlm_T2b; elm_lcmb_150 = elm_T2a + elm_T2b; blm_lcmb_150 = blm_T2a + blm_T2b
# Seed "A"
np.random.seed(4190002645)
cltt_T3a = np.nan_to_num((cltt_090_220)**2 / cltt_090_090); clee_T3a = np.nan_to_num((clee_090_220)**2 / clee_090_090); clbb_T3a = np.nan_to_num((clbb_090_220)**2 / clbb_090_090)
tlm_T3a,elm_T3a,blm_T3a = hp.synalm([cltt_T3a,clee_T3a,clbb_T3a,cltt_T3a*0],new=True,lmax=lmax)
# Seed "B"
np.random.seed(89052206)
cltt_T3b = np.nan_to_num((cltt_150_220 - cltt_090_150*cltt_090_220/cltt_090_090)**2 / cltt_T2b)
clee_T3b = np.nan_to_num((clee_150_220 - clee_090_150*clee_090_220/clee_090_090)**2 / clee_T2b)
clbb_T3b = np.nan_to_num((clbb_150_220 - clbb_090_150*clbb_090_220/clbb_090_090)**2 / clbb_T2b)
tlm_T3b,elm_T3b,blm_T3b = hp.synalm([cltt_T3b,clee_T3b,clbb_T3b,cltt_T3b*0],new=True,lmax=lmax)
# Seed "C"
np.random.seed(978540195)
cltt_T3c = cltt_220_220 - cltt_T3a - cltt_T3b; clee_T3c = clee_220_220 - clee_T3a - clee_T3b; clbb_T3c = clbb_220_220 - clbb_T3a - clbb_T3b
tlm_T3c,elm_T3c,blm_T3c = hp.synalm([cltt_T3c,clee_T3c,clbb_T3c,cltt_T3c*0],new=True,lmax=lmax)
tlm_lcmb_220 = tlm_T3a + tlm_T3b + tlm_T3c; elm_lcmb_220 = elm_T3a + elm_T3b + elm_T3c; blm_lcmb_220 = blm_T3a + blm_T3b + blm_T3c
# Combine frequencies
tlm = hp.almxfl(tlm_lcmb_95,w_Tmv[0][:lmax+1]) + hp.almxfl(tlm_lcmb_150,w_Tmv[1][:lmax+1]) + hp.almxfl(tlm_lcmb_220,w_Tmv[2][:lmax+1])
elm = hp.almxfl(elm_lcmb_95,w_Emv[0][:lmax+1]) + hp.almxfl(elm_lcmb_150,w_Emv[1][:lmax+1]) + hp.almxfl(elm_lcmb_220,w_Emv[2][:lmax+1])
blm = hp.almxfl(blm_lcmb_95,w_Bmv[0][:lmax+1]) + hp.almxfl(blm_lcmb_150,w_Bmv[1][:lmax+1]) + hp.almxfl(blm_lcmb_220,w_Bmv[2][:lmax+1])
# NOTE: UNLENSED GAUSSIAN CMB FROM ROTATED AGORA CMB
sltt_agora_rot_gaussian_unl = hp.alm2cl(tlm,tlm) + artificial_noise
slee_agora_rot_gaussian_unl = hp.alm2cl(elm,elm)
slbb_agora_rot_gaussian_unl = hp.alm2cl(blm,blm)
slte_agora_rot_gaussian_unl = hp.alm2cl(tlm,elm)

# Gaussian CMB
tlm1,elm1,blm1 = hp.read_alm(alm_cmb_sim1,hdu=[1,2,3])
# NOTE: GAUSSIAN CMB
sltt_gaussian = hp.alm2cl(tlm1,tlm1) + artificial_noise
slee_gaussian = hp.alm2cl(elm1,elm1)
slbb_gaussian = hp.alm2cl(blm1,blm1)
slte_gaussian = hp.alm2cl(tlm1,elm1)

# Gaussian foregrounds
tflm1_95, eflm1_95, bflm1_95 = hp.read_alm(flm_95ghz_sim1,hdu=[1,2,3])
tflm1_150, eflm1_150, bflm1_150 = hp.read_alm(flm_150ghz_sim1,hdu=[1,2,3])
tflm1_220, eflm1_220, bflm1_220 = hp.read_alm(flm_220ghz_sim1,hdu=[1,2,3])
tlm = hp.almxfl(tflm1_95,w_Tmv[0][:lmax+1]) + hp.almxfl(tflm1_150,w_Tmv[1][:lmax+1]) + hp.almxfl(tflm1_220,w_Tmv[2][:lmax+1])
elm = hp.almxfl(eflm1_95,w_Emv[0][:lmax+1]) + hp.almxfl(eflm1_150,w_Emv[1][:lmax+1]) + hp.almxfl(eflm1_220,w_Emv[2][:lmax+1])
blm = hp.almxfl(bflm1_95,w_Bmv[0][:lmax+1]) + hp.almxfl(bflm1_150,w_Bmv[1][:lmax+1]) + hp.almxfl(bflm1_220,w_Bmv[2][:lmax+1])
# NOTE: GAUSSIAN FG
fltt_gaussian = hp.alm2cl(tlm,tlm)
flee_gaussian = hp.alm2cl(elm,elm)
flbb_gaussian = hp.alm2cl(blm,blm)
flte_gaussian = hp.alm2cl(tlm,elm)

#plt.figure(0)
#plt.clf()
#plt.plot(l, sltt, color='firebrick', linestyle='-', label='sltt, theory')
#plt.plot(l, slee, color='forestgreen', linestyle='-', label='slee, theory')
#plt.plot(l, slbb, color='darkblue', linestyle='-', label='slbb, theory')
#plt.plot(l, slte, color='gold', linestyle='-', label='slte, theory')
#plt.plot(l, sltt_agora, color='darksalmon', linestyle='--', label='sltt, Agora')
#plt.plot(l, slee_agora, color='lightgreen', linestyle='--', label='slee, Agora')
#plt.plot(l, slbb_agora, color='powderblue', linestyle='--', label='slbb, Agora')
#plt.plot(l, slte_agora, color='palegoldenrod', linestyle='--', label='slte, Agora')
#plt.plot(l, sltt_agora_rot, color='rosybrown', linestyle='--', label='sltt, Agora, rotated')
#plt.plot(l, slee_agora_rot, color='darkseagreen', linestyle='--', label='slee, Agora, rotated')
#plt.plot(l, slbb_agora_rot, color='cornflowerblue', linestyle='--', label='slbb, Agora, rotated')
#plt.plot(l, slte_agora_rot, color='orange', linestyle='--', label='slte, Agora, rotated')
#plt.plot(l, sltt_agora_rot_gaussian_unl, color='lightpink', linestyle=':', label='sltt, unlensed Gaussian CMB from rotated Agora')
#plt.plot(l, slee_agora_rot_gaussian_unl, color='mediumaquamarine', linestyle=':', label='slee, unlensed Gaussian CMB from rotated Agora')
#plt.plot(l, slbb_agora_rot_gaussian_unl, color='cadetblue', linestyle=':', label='slbb, unlensed Gaussian CMB from rotated Agora')
#plt.plot(l, slte_agora_rot_gaussian_unl, color='bisque', linestyle=':', label='slte, unlensed Gaussian CMB from rotated Agora')
#plt.plot(l, sltt_gaussian, color='violet', linestyle=':', label='sltt, Gaussian CMB')
#plt.plot(l, slee_gaussian, color='olive', linestyle=':', label='slee, Gaussian CMB')
#plt.plot(l, slbb_gaussian, color='royalblue', linestyle=':', label='slbb, Gaussian CMB')
#plt.plot(l, slte_gaussian, color='peru', linestyle=':', label='slte, Gaussian CMB')
#plt.xscale('log')
#plt.yscale('log')
#plt.xlim(300,lmax)
#plt.ylim(1e-9,1e0)
#plt.legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize='x-small')
#plt.title(f'spectra')
#plt.ylabel("$C_\ell$")
#plt.xlabel('$\ell$')
#plt.savefig(dir_out+f'/figs/agora_alt_spectra_signal.png',bbox_inches='tight')
#
#plt.clf()
#plt.plot(l, fltt_agora, color='darksalmon', linestyle='--', label='fltt, Agora')
#plt.plot(l, flee_agora, color='lightgreen', linestyle='--', label='flee, Agora')
#plt.plot(l, flbb_agora, color='powderblue', linestyle='--', label='flbb, Agora')
#plt.plot(l, flte_agora, color='palegoldenrod', linestyle='--', label='flte, Agora')
#plt.plot(l, fltt_gaussian, color='rosybrown', linestyle='--', label='fltt, Gaussian')
#plt.plot(l, flee_gaussian, color='darkseagreen', linestyle='--', label='flee, Gaussian')
#plt.plot(l, flbb_gaussian, color='cornflowerblue', linestyle='--', label='flbb, Gaussian')
#plt.plot(l, flte_gaussian, color='orange', linestyle='--', label='flte, Gaussian')
#plt.xscale('log')
#plt.yscale('log')
#plt.xlim(300,lmax)
#plt.ylim(1e-10,1e0)
#plt.legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize='x-small')
#plt.title(f'spectra')
#plt.ylabel("$C_\ell$")
#plt.xlabel('$\ell$')
#plt.savefig(dir_out+f'/figs/agora_alt_spectra_fg.png',bbox_inches='tight')
#
#plt.clf()
#plt.plot(l, cltt_agora_tot, color='darksalmon', linestyle='--', label='total cltt, Agora')
#plt.plot(l, clee_agora_tot, color='lightgreen', linestyle='--', label='total clee, Agora')
#plt.plot(l, clbb_agora_tot, color='powderblue', linestyle='--', label='total clbb, Agora')
#plt.plot(l, clte_agora_tot, color='palegoldenrod', linestyle='--', label='total clte, Agora')
#plt.plot(l, cltt_agora_tot_lcmb_cross, color='rosybrown', linestyle='--', label='cltt, Agora total x lensed CMB')
#plt.plot(l, clee_agora_tot_lcmb_cross, color='darkseagreen', linestyle='--', label='clee, Agora total x lensed CMB')
#plt.plot(l, clbb_agora_tot_lcmb_cross, color='cornflowerblue', linestyle='--', label='clbb, Agora total x lensed CMB')
#plt.plot(l, clte_agora_tot_lcmb_cross, color='orange', linestyle='--', label='clte, Agora total x lensed CMB')
#plt.xscale('log')
#plt.yscale('log')
#plt.xlim(300,lmax)
#plt.ylim(1e-9,1e0)
#plt.legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize='x-small')
#plt.title(f'spectra')
#plt.ylabel("$C_\ell$")
#plt.xlabel('$\ell$')
#plt.savefig(dir_out+f'/figs/agora_alt_spectra_total.png',bbox_inches='tight')

totalcls_gaussian_filename = dir_out+f'totalcls/totalcls_average_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_standard.npy'
totalcls = np.load(totalcls_gaussian_filename)
cltt_gaussian_tot = totalcls[:,0]; clee_gaussian_tot = totalcls[:,1]; clbb_gaussian_tot = totalcls[:,2]; clte_gaussian_tot = totalcls[:,3]
plt.figure(0)
plt.clf()
plt.axhline(y=1, color='k', linestyle='--')
#plt.plot(l, sltt_agora/sltt_gaussian, color='firebrick', alpha=0.8, linestyle='-', label='sltt ratio, Agora/Gaussian')
#plt.plot(l, slee_agora/slee_gaussian, color='forestgreen', alpha=0.8, linestyle='-', label='slee ratio, Agora/Gaussian')
#plt.plot(l, slbb_agora/slbb_gaussian, color='darkblue', alpha=0.8, linestyle='-', label='slbb ratio, Agora/Gaussian')
##plt.plot(l, slte_agora/slte_gaussian, color='gold', alpha=0.8, linestyle='-', label='slte ratio, Agora/Gaussian')
#plt.plot(l, fltt_agora/fltt_gaussian, color='darksalmon', alpha=0.8, linestyle='--', label='fltt ratio, Agora/Gaussian')
#plt.plot(l, flee_agora/flee_gaussian, color='lightgreen', alpha=0.8, linestyle='--', label='flee ratio, Agora/Gaussian')
#plt.plot(l, flbb_agora/flbb_gaussian, color='powderblue', alpha=0.8, linestyle='--', label='flbb ratio, Agora/Gaussian')
##plt.plot(l, flte_agora/flte_gaussian, color='palegoldenrod', alpha=0.8, linestyle='--', label='flte ratio, Agora/Gaussian')
plt.plot(l, cltt_agora_tot/cltt_gaussian_tot, color='rosybrown', alpha=0.8, linestyle=':', label='cltt ratio, Agora/Gaussian')
plt.plot(l, clee_agora_tot/clee_gaussian_tot, color='darkseagreen', alpha=0.8, linestyle=':', label='clee ratio, Agora/Gaussian')
plt.plot(l, clbb_agora_tot/clbb_gaussian_tot, color='cornflowerblue', alpha=0.8, linestyle=':', label='clbb ratio, Agora/Gaussian')
##plt.plot(l, clte_agora_tot/clte_gaussian_tot, color='orange', alpha=0.8, linestyle=':', label='clte ratio, Agora/Gaussian')
plt.xlabel('$\ell$')
plt.title(f'Spectra Comparison, Agora/Gaussian')
plt.legend(fontsize='x-small')
plt.xscale('log')
plt.xlim(300,lmax)
plt.ylim(0.5,1.5)
plt.savefig(dir_out+f'/figs/agora_spectra_ratio_comparison.png',bbox_inches='tight')

'''
# Compare total spectra

append='agora_standard'
totalcls_filename = dir_out+f'totalcls/totalcls_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy'
totalcls = np.load(totalcls_filename)
cltt = totalcls[:,0]; clee = totalcls[:,1]; clbb = totalcls[:,2]; clte = totalcls[:,3]

append='agora_standard_rotatedcmb'
totalcls_filename = dir_out+f'totalcls/totalcls_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy'
totalcls = np.load(totalcls_filename)
cltt_rotatedcmb = totalcls[:,0]; clee_rotatedcmb = totalcls[:,1]; clbb_rotatedcmb = totalcls[:,2]; clte_rotatedcmb = totalcls[:,3]

append = 'agora_standard_gaussiancmb'
totalcls_filename = dir_out+f'totalcls/totalcls_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy'
totalcls = np.load(totalcls_filename)
cltt_gaussiancmb = totalcls[:,0]; clee_gaussiancmb = totalcls[:,1]; clbb_gaussiancmb = totalcls[:,2]; clte_gaussiancmb = totalcls[:,3]

append = 'agora_standard_rotatedcmb_gaussianfg'
totalcls_filename = dir_out+f'totalcls/totalcls_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy'
totalcls = np.load(totalcls_filename)
cltt_rotatedcmb_gaussianfg = totalcls[:,0]; clee_rotatedcmb_gaussianfg = totalcls[:,1]; clbb_rotatedcmb_gaussianfg = totalcls[:,2]; clte_rotatedcmb_gaussianfg = totalcls[:,3]

append = 'agora_standard_rotatedgaussiancmb'
totalcls_filename = dir_out+f'totalcls/totalcls_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy'
totalcls = np.load(totalcls_filename)
cltt_rotatedgaussiancmb = totalcls[:,0]; clee_rotatedgaussiancmb = totalcls[:,1]; clbb_rotatedgaussiancmb = totalcls[:,2]; clte_rotatedgaussiancmb = totalcls[:,3]

append = 'agora_standard_gaussianfg'
totalcls_filename = dir_out+f'totalcls/totalcls_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy'
totalcls = np.load(totalcls_filename)
cltt_gaussianfg = totalcls[:,0]; clee_gaussianfg = totalcls[:,1]; clbb_gaussianfg = totalcls[:,2]; clte_gaussianfg = totalcls[:,3]

totalcls_filename = dir_out+f'totalcls/totalcls_average_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_standard.npy'
totalcls = np.load(totalcls_filename)
cltt_standard_notagora = totalcls[:,0]; clee_standard_notagora = totalcls[:,1]; clbb_standard_notagora = totalcls[:,2]; clte_standard_notagora = totalcls[:,3]

# Unlensed CMB alms sampled from lensed theory spectra
unl_map_sim1 = f'/oak/stanford/orgs/kipac/users/yukanaka/unl_from_lensed_cls/unl_from_lensed_cls_seed1_lmax{lmax}_nside{nside}_20230905.fits'
t1,q1,u1 = hp.read_map(unl_map_sim1,field=[0,1,2])
tlm1,elm1,blm1 = hp.map2alm([t1,q1,u1],lmax=lmax)
tlm1_150 = tlm1.copy(); tlm1_220 = tlm1.copy(); tlm1_95 = tlm1.copy()
elm1_150 = elm1.copy(); elm1_220 = elm1.copy(); elm1_95 = elm1.copy()
blm1_150 = blm1.copy(); blm1_220 = blm1.copy(); blm1_95 = blm1.copy()
tflm1_150, eflm1_150, bflm1_150 = hp.read_alm(flm_150ghz_sim1,hdu=[1,2,3])
tflm1_220, eflm1_220, bflm1_220 = hp.read_alm(flm_220ghz_sim1,hdu=[1,2,3])
tflm1_95, eflm1_95, bflm1_95 = hp.read_alm(flm_95ghz_sim1,hdu=[1,2,3])
tlm1_150 += tflm1_150; tlm1_220 += tflm1_220; tlm1_95 += tflm1_95
elm1_150 += eflm1_150; elm1_220 += eflm1_220; elm1_95 += eflm1_95
blm1_150 += bflm1_150; blm1_220 += bflm1_220; blm1_95 += bflm1_95
nlm1_090_filename = dir_out + f'nlm/nlm_090_lmax{lmax}_seed1.alm'
nlm1_150_filename = dir_out + f'nlm/nlm_150_lmax{lmax}_seed1.alm'
nlm1_220_filename = dir_out + f'nlm/nlm_220_lmax{lmax}_seed1.alm'
nlmt1_090,nlme1_090,nlmb1_090 = hp.read_alm(nlm1_090_filename,hdu=[1,2,3])
nlmt1_150,nlme1_150,nlmb1_150 = hp.read_alm(nlm1_150_filename,hdu=[1,2,3])
nlmt1_220,nlme1_220,nlmb1_220 = hp.read_alm(nlm1_220_filename,hdu=[1,2,3])
tlm1_150 += nlmt1_150; tlm1_220 += nlmt1_220; tlm1_95 += nlmt1_090
elm1_150 += nlme1_150; elm1_220 += nlme1_220; elm1_95 += nlme1_090
blm1_150 += nlmb1_150; blm1_220 += nlmb1_220; blm1_95 += nlmb1_090
tlm1 = hp.almxfl(tlm1_95,w_Tmv[0][:lmax+1]) + hp.almxfl(tlm1_150,w_Tmv[1][:lmax+1]) + hp.almxfl(tlm1_220,w_Tmv[2][:lmax+1])
elm1 = hp.almxfl(elm1_95,w_Emv[0][:lmax+1]) + hp.almxfl(elm1_150,w_Emv[1][:lmax+1]) + hp.almxfl(elm1_220,w_Emv[2][:lmax+1])
blm1 = hp.almxfl(blm1_95,w_Bmv[0][:lmax+1]) + hp.almxfl(blm1_150,w_Bmv[1][:lmax+1]) + hp.almxfl(blm1_220,w_Bmv[2][:lmax+1])
cltt_unlnotagora = hp.alm2cl(tlm1,tlm1) + artificial_noise
clee_unlnotagora = hp.alm2cl(elm1,elm1)
clbb_unlnotagora = hp.alm2cl(blm1,blm1)
clte_unlnotagora = hp.alm2cl(tlm1,elm1)

plt.figure(0)
plt.clf()
plt.plot(l, cltt, color='firebrick', linestyle='-', label='cltt, standard')
plt.plot(l, clee, color='forestgreen', linestyle='-', label='clee, standard')
plt.plot(l, clbb, color='darkblue', linestyle='-', label='clbb, standard')
plt.plot(l, clte, color='gold', linestyle='-', label='clte, standard')
plt.plot(l, cltt_rotatedcmb, color='darksalmon', linestyle='--', label='cltt, rotatedcmb')
plt.plot(l, clee_rotatedcmb, color='lightgreen', linestyle='--', label='clee, rotatedcmb')
plt.plot(l, clbb_rotatedcmb, color='powderblue', linestyle='--', label='clbb, rotatedcmb')
plt.plot(l, clte_rotatedcmb, color='palegoldenrod', linestyle='--', label='clte, rotatedcmb')
plt.plot(l, cltt_gaussiancmb, color='rosybrown', linestyle='--', label='cltt, gaussiancmb')
plt.plot(l, clee_gaussiancmb, color='darkseagreen', linestyle='--', label='clee, gaussiancmb')
plt.plot(l, clbb_gaussiancmb, color='cornflowerblue', linestyle='--', label='clbb, gaussiancmb')
plt.plot(l, clte_gaussiancmb, color='orange', linestyle='--', label='clte, gaussiancmb')
plt.plot(l, cltt_rotatedcmb_gaussianfg, color='lightpink', linestyle=':', label='cltt, rotatedcmb_gaussianfg')
plt.plot(l, clee_rotatedcmb_gaussianfg, color='mediumaquamarine', linestyle=':', label='clee, rotatedcmb_gaussianfg')
plt.plot(l, clbb_rotatedcmb_gaussianfg, color='cadetblue', linestyle=':', label='clbb, rotatedcmb_gaussianfg')
plt.plot(l, clte_rotatedcmb_gaussianfg, color='bisque', linestyle=':', label='clte, rotatedcmb_gaussianfg')
plt.plot(l, cltt_rotatedgaussiancmb, color='violet', linestyle=':', label='cltt, rotatedgaussiancmb (UNLENSED)')
plt.plot(l, clee_rotatedgaussiancmb, color='olive', linestyle=':', label='clee, rotatedgaussiancmb (UNLENSED)')
plt.plot(l, clbb_rotatedgaussiancmb, color='royalblue', linestyle=':', label='clbb, rotatedgaussiancmb (UNLENSED)')
plt.plot(l, clte_rotatedgaussiancmb, color='peru', linestyle=':', label='clte, rotatedgaussiancmb (UNLENSED)')
#plt.plot(l, cltt_gaussianfg, color='violet', linestyle=':', label='cltt, gaussianfg')
#plt.plot(l, clee_gaussianfg, color='olive', linestyle=':', label='clee, gaussianfg')
#plt.plot(l, clbb_gaussianfg, color='royalblue', linestyle=':', label='clbb, gaussianfg')
#plt.plot(l, clte_gaussianfg, color='peru', linestyle=':', label='clte, gaussianfg')
#plt.plot(l, cltt_standard_notagora, color='violet', linestyle=':', label='cltt, standard, not Agora')
#plt.plot(l, clee_standard_notagora, color='olive', linestyle=':', label='clee, standard, not Agora')
#plt.plot(l, clbb_standard_notagora, color='royalblue', linestyle=':', label='clbb, standard, not Agora')
#plt.plot(l, clte_standard_notagora, color='peru', linestyle=':', label='clte, standard, not Agora')
#plt.plot(l, cltt_unlnotagora, color='violet', linestyle=':', label='cltt, unl, not Agora')
#plt.plot(l, clee_unlnotagora, color='olive', linestyle=':', label='clee, unl, not Agora')
#plt.plot(l, clbb_unlnotagora, color='royalblue', linestyle=':', label='clbb, unl, not Agora')
#plt.plot(l, clte_unlnotagora, color='peru', linestyle=':', label='clte, unl, not Agora')
plt.xscale('log')
plt.yscale('log')
plt.xlim(300,lmax)
plt.ylim(1e-9,1e1)
plt.legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize='x-small')
plt.title(f'spectra')
plt.ylabel("$C_\ell$")
plt.xlabel('$\ell$')
plt.savefig(dir_out+f'/figs/agora_alt_spectra.png',bbox_inches='tight')
'''
