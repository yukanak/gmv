import pickle
import sys, os
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import numpy as np
import matplotlib.pyplot as plt
import healqest_utils as utils
import healpy as hp

def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

config_file = 'test_yuka_lmaxT3500.yaml'
append = 'standard'
fg_model = 'websky'
config = utils.parse_yaml(config_file)
lmax = config['lensrec']['Lmax']
lmaxT = config['lensrec']['lmaxT']
lmaxP = config['lensrec']['lmaxP']
lmin = config['lensrec']['lminT']
nside = config['lensrec']['nside']
dir_out = config['dir_out']
cltype = config['lensrec']['cltype']
l = np.arange(0,lmax+1)
sl = {ee:config['cls'][cltype][ee] for ee in config['cls'][cltype].keys()}
#totalcls_avg = np.load(dir_out+f'totalcls/totalcls_average_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy')

alm_cmb_sim1 = f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/tqu1/len/alms/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb1_seed1_alm_lmax{lmax}.fits'
tlm1,elm1,blm1 = hp.read_alm(alm_cmb_sim1,hdu=[1,2,3])
sltt = hp.alm2cl(tlm1,tlm1); slee = hp.alm2cl(elm1,elm1); slbb = hp.alm2cl(blm1,blm1); slte = hp.alm2cl(tlm1,elm1)
# ILC weights
# These are dimensions (4097, 3) initially; then transpose to make it (3, 4097)
w_tsz_null = np.load('/oak/stanford/orgs/kipac/users/yukanaka/websky/weights_websky_cmbrec_tsznull_lmax4096.npy').T
w_Tmv = np.load('/oak/stanford/orgs/kipac/users/yukanaka/websky/weights_websky_cmbrec_mv_lmax4096.npy').T
w_cib_null = np.load('/oak/stanford/orgs/kipac/users/yukanaka/websky/weights_websky_cmbrec_cibnull_lmax4096.npy').T
# Dimension (3, 6001) for 90, 150, 220 GHz respectively
w_Emv = np.loadtxt('ilc_weights/weights1d_EE_spt3g_cmbmv.dat')
w_Bmv = np.loadtxt('ilc_weights/weights1d_BB_spt3g_cmbmv.dat')
# If lmaxT != lmaxP, we add artificial noise in TT for ell > lmaxT
artificial_noise = np.zeros(lmax+1)
artificial_noise[lmaxT+2:] = 1.e10

# Websky sims, use foreground-less E and B lensed alms from their website
websky_095_T = '/oak/stanford/orgs/kipac/users/yukanaka/websky/websky_spt_95ghz_lcmb_tsz_cib_ksz_kszpatchy_muk_alm_lmax4096.fits'
websky_150_T = '/oak/stanford/orgs/kipac/users/yukanaka/websky/websky_spt_150ghz_lcmb_tsz_cib_ksz_kszpatchy_muk_alm_lmax4096.fits'
websky_220_T = '/oak/stanford/orgs/kipac/users/yukanaka/websky/websky_spt_220ghz_lcmb_tsz_cib_ksz_kszpatchy_muk_alm_lmax4096.fits'
# From https://lambda.gsfc.nasa.gov/simulation/mocks_data.html but they claim it's "T,Q,U alms", but I think they mean T,E,B...
websky_nofg = '/oak/stanford/orgs/kipac/users/yukanaka/websky/lensed_alm.fits'
# Load
tlm_95 = hp.read_alm(websky_095_T)
tlm_150 = hp.read_alm(websky_150_T)
tlm_220 = hp.read_alm(websky_220_T)
tlm_lcmb, elm_lcmb, blm_lcmb = hp.read_alm(websky_nofg,hdu=[1,2,3])
tlm_95 = utils.reduce_lmax(tlm_95,lmax=lmax); tlm_150 = utils.reduce_lmax(tlm_150,lmax=lmax); tlm_220 = utils.reduce_lmax(tlm_220,lmax=lmax);
tlm_lcmb = utils.reduce_lmax(tlm_lcmb,lmax=lmax)
elm_lcmb = utils.reduce_lmax(elm_lcmb,lmax=lmax)
blm_lcmb = utils.reduce_lmax(blm_lcmb,lmax=lmax)
# ILC the WebSky alms
if append == 'standard':
    tlm = hp.almxfl(tlm_95,w_Tmv[0][:lmax+1]) + hp.almxfl(tlm_150,w_Tmv[1][:lmax+1]) + hp.almxfl(tlm_220,w_Tmv[2][:lmax+1])
elif append == 'mh':
    tlm = hp.almxfl(tlm_95,w_tsz_null[0][:lmax+1]) + hp.almxfl(tlm_150,w_tsz_null[1][:lmax+1]) + hp.almxfl(tlm_220,w_tsz_null[2][:lmax+1])
elm = hp.almxfl(elm_lcmb,w_Emv[0][:lmax+1]) + hp.almxfl(elm_lcmb,w_Emv[1][:lmax+1]) + hp.almxfl(elm_lcmb,w_Emv[2][:lmax+1])
blm = hp.almxfl(blm_lcmb,w_Bmv[0][:lmax+1]) + hp.almxfl(blm_lcmb,w_Bmv[1][:lmax+1]) + hp.almxfl(blm_lcmb,w_Bmv[2][:lmax+1])

##### WEBSKY LENSED CMB ONLY #####
sltt_websky = hp.alm2cl(tlm_lcmb,tlm_lcmb)
slee_websky = hp.alm2cl(elm_lcmb,elm_lcmb)
slbb_websky = hp.alm2cl(blm_lcmb,blm_lcmb)
slte_websky = hp.alm2cl(tlm_lcmb,elm_lcmb)
#################################

tlm_fg_95 = tlm_95 - tlm_lcmb;
tlm_fg_150 = tlm_150 - tlm_lcmb;
tlm_fg_220 = tlm_220 - tlm_lcmb;
if append == 'standard':
    tlm_fg = hp.almxfl(tlm_fg_95,w_Tmv[0][:lmax+1]) + hp.almxfl(tlm_fg_150,w_Tmv[1][:lmax+1]) + hp.almxfl(tlm_fg_220,w_Tmv[2][:lmax+1])
elif append == 'mh':
    tlm_fg = hp.almxfl(tlm_fg_95,w_tsz_null[0][:lmax+1]) + hp.almxfl(tlm_fg_150,w_tsz_null[1][:lmax+1]) + hp.almxfl(tlm_fg_220,w_tsz_null[2][:lmax+1])

##### WEBSKY FOREGROUNDS ONLY #####
fltt_websky = hp.alm2cl(tlm_fg,tlm_fg)
##################################

nlm1_090_filename = dir_out + f'nlm/nlm_090_lmax{lmax}_seed999.alm'
nlm1_150_filename = dir_out + f'nlm/nlm_150_lmax{lmax}_seed999.alm'
nlm1_220_filename = dir_out + f'nlm/nlm_220_lmax{lmax}_seed999.alm'
nlmt_090,nlme_090,nlmb_090 = hp.read_alm(nlm1_090_filename,hdu=[1,2,3])
nlmt_150,nlme_150,nlmb_150 = hp.read_alm(nlm1_150_filename,hdu=[1,2,3])
nlmt_220,nlme_220,nlmb_220 = hp.read_alm(nlm1_220_filename,hdu=[1,2,3])
if append == 'standard':
    tlm_n = hp.almxfl(nlmt_090,w_Tmv[0][:lmax+1]) + hp.almxfl(nlmt_150,w_Tmv[1][:lmax+1]) + hp.almxfl(nlmt_220,w_Tmv[2][:lmax+1])
elif append == 'mh':
    tlm_n = hp.almxfl(nlmt_090,w_tsz_null[0][:lmax+1]) + hp.almxfl(nlmt_150,w_tsz_null[1][:lmax+1]) + hp.almxfl(nlmt_220,w_tsz_null[2][:lmax+1])
elm_n = hp.almxfl(nlme_090,w_Emv[0][:lmax+1]) + hp.almxfl(nlme_150,w_Emv[1][:lmax+1]) + hp.almxfl(nlme_220,w_Emv[2][:lmax+1])
blm_n = hp.almxfl(nlmb_090,w_Bmv[0][:lmax+1]) + hp.almxfl(nlmb_150,w_Bmv[1][:lmax+1]) + hp.almxfl(nlmb_220,w_Bmv[2][:lmax+1])

##### WEBSKY NOISE ONLY #####
nltt_websky = hp.alm2cl(tlm_n,tlm_n) + artificial_noise
nlee_websky = hp.alm2cl(elm_n,elm_n)
nlbb_websky = hp.alm2cl(blm_n,blm_n)
nlte_websky = hp.alm2cl(tlm_n,elm_n)
############################

# Adding noise too to make it comparable with total Gaussian spectra
tlm_150 += nlmt_150; tlm_220 += nlmt_220; tlm_95 += nlmt_090
elm_150 = elm_lcmb + nlme_150; elm_220 = elm_lcmb + nlme_220; elm_95 = elm_lcmb + nlme_090
blm_150 = blm_lcmb + nlmb_150; blm_220 = blm_lcmb + nlmb_220; blm_95 = blm_lcmb + nlmb_090
if append == 'standard':
    tlm = hp.almxfl(tlm_95,w_Tmv[0][:lmax+1]) + hp.almxfl(tlm_150,w_Tmv[1][:lmax+1]) + hp.almxfl(tlm_220,w_Tmv[2][:lmax+1])
elif append == 'mh':
    tlm = hp.almxfl(tlm_95,w_tsz_null[0][:lmax+1]) + hp.almxfl(tlm_150,w_tsz_null[1][:lmax+1]) + hp.almxfl(tlm_220,w_tsz_null[2][:lmax+1])
elm = hp.almxfl(elm_95,w_Emv[0][:lmax+1]) + hp.almxfl(elm_150,w_Emv[1][:lmax+1]) + hp.almxfl(elm_220,w_Emv[2][:lmax+1])
blm = hp.almxfl(blm_95,w_Bmv[0][:lmax+1]) + hp.almxfl(blm_150,w_Bmv[1][:lmax+1]) + hp.almxfl(blm_220,w_Bmv[2][:lmax+1])

##### WEBSKY TOTAL (LCMB + FG + NOISE) #####
cltt_websky_tot = hp.alm2cl(tlm,tlm) + artificial_noise
clee_websky_tot = hp.alm2cl(elm,elm)
clbb_websky_tot = hp.alm2cl(blm,blm)
clte_websky_tot = hp.alm2cl(tlm,elm)
###########################################

# Get foreground + noise only
tlm_fg_95 = tlm_95 - tlm_lcmb;
tlm_fg_150 = tlm_150 - tlm_lcmb;
tlm_fg_220 = tlm_220 - tlm_lcmb;
if append == 'standard':
    tlm = hp.almxfl(tlm_fg_95,w_Tmv[0][:lmax+1]) + hp.almxfl(tlm_fg_150,w_Tmv[1][:lmax+1]) + hp.almxfl(tlm_fg_220,w_Tmv[2][:lmax+1])
elif append == 'mh':
    tlm = hp.almxfl(tlm_fg_95,w_tsz_null[0][:lmax+1]) + hp.almxfl(tlm_fg_150,w_tsz_null[1][:lmax+1]) + hp.almxfl(tlm_fg_220,w_tsz_null[2][:lmax+1])

##### WEBSKY FG + NOISE #####
fnltt_websky = hp.alm2cl(tlm,tlm) + artificial_noise
fnlee_websky = nlee_websky
fnlbb_websky = nlbb_websky
fnlte_websky = nlte_websky
############################

# Get noise-only spectra
noise_curves_090_090 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_090_090.txt'))
noise_curves_150_150 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_150_150.txt'))
noise_curves_220_220 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_220_220.txt'))
noise_curves_090_150 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_090_150.txt'))
noise_curves_090_220 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_090_220.txt'))
noise_curves_150_220 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_150_220.txt'))
# Combine cross frequency spectra with ILC weights
ret = np.zeros((lmax+1,3))
for a in range(3):
    if append == 'standard':
        if a == 0: b='tt'; c=1; w1=w_Tmv; w2=w_Tmv
        else: pass
    elif append == 'mh':
        if a == 0: b='tt'; c=1; w1=w_tsz_null; w2=w_tsz_null
        else: pass
    if a == 1: b='ee'; c=2; w1=w_Emv; w2=w_Emv
    if a == 2: b='bb'; c=3; w1=w_Bmv; w2=w_Bmv
    for ll in l:
        # At each ell, have 3x3 matrix with each block containing Cl for different frequency combinations
        clmat = np.zeros((3,3))
        clmat[0,0] = noise_curves_090_090[ll,c]
        clmat[1,1] = noise_curves_150_150[ll,c]
        clmat[2,2] = noise_curves_220_220[ll,c]
        clmat[0,1] = clmat[1,0] = noise_curves_090_150[ll,c]
        clmat[0,2] = clmat[2,0] = noise_curves_090_220[ll,c]
        clmat[1,2] = clmat[2,1] = noise_curves_150_220[ll,c]
        ret[ll,a]=np.dot(w1[:,ll], np.dot(clmat, w2[:,ll].T))

# Get foregrounds-only spectrum for sim 1
fnlm_sim1 = f'/oak/stanford/orgs/kipac/users/yukanaka/gmv/inputs/spt3g_2019_2020_websky/websky_fg_plus_spt3g_20192020_noise_lmax{lmax}_seed1_mv.fits'
tlm1, elm1, blm1 = hp.read_alm(fnlm_sim1,hdu=[1,2,3])
# Load noise
nlm_090_filename = dir_out + f'nlm/nlm_090_lmax{lmax}_seed1.alm'
nlm_150_filename = dir_out + f'nlm/nlm_150_lmax{lmax}_seed1.alm'
nlm_220_filename = dir_out + f'nlm/nlm_220_lmax{lmax}_seed1.alm'
nlmt_090,nlme_090,nlmb_090 = hp.read_alm(nlm_090_filename,hdu=[1,2,3])
nlmt_150,nlme_150,nlmb_150 = hp.read_alm(nlm_150_filename,hdu=[1,2,3])
nlmt_220,nlme_220,nlmb_220 = hp.read_alm(nlm_220_filename,hdu=[1,2,3])
# ILC combine frequencies
nlmt_mv = hp.almxfl(nlmt_090,w_Tmv[0][:lmax+1]) + hp.almxfl(nlmt_150,w_Tmv[1][:lmax+1]) + hp.almxfl(nlmt_220,w_Tmv[2][:lmax+1])
nlme_mv = hp.almxfl(nlme_090,w_Emv[0][:lmax+1]) + hp.almxfl(nlme_150,w_Emv[1][:lmax+1]) + hp.almxfl(nlme_220,w_Emv[2][:lmax+1])
nlmb_mv = hp.almxfl(nlmb_090,w_Bmv[0][:lmax+1]) + hp.almxfl(nlmb_150,w_Bmv[1][:lmax+1]) + hp.almxfl(nlmb_220,w_Bmv[2][:lmax+1])
tlm1 -= nlmt_mv
elm1 -= nlme_mv
blm1 -= nlmb_mv
fltt = hp.alm2cl(tlm1,tlm1); flee = hp.alm2cl(elm1,elm1); flbb = hp.alm2cl(blm1,blm1); flte = hp.alm2cl(tlm1,elm1)

'''
plt.figure(1)
plt.clf()
plt.plot(l, moving_average(clte_websky_tot/(totalcls_avg[:,3]),window_size=15), color='palegoldenrod', alpha=0.8, linestyle='--', label='clte ratio, WebSky/Gaussian')
plt.plot(l, moving_average(clbb_websky_tot/(totalcls_avg[:,2]),window_size=15), color='powderblue', alpha=0.8, linestyle='--', label='clbb ratio, WebSky/Gaussian')
plt.plot(l, moving_average(clee_websky_tot/(totalcls_avg[:,1]),window_size=15), color='lightgreen', alpha=0.8, linestyle='--', label='clee ratio, WebSky/Gaussian')
if append == 'standard':
    plt.plot(l, moving_average(cltt_websky_tot/(totalcls_avg[:,0]),window_size=15), color='darksalmon', alpha=0.8, linestyle='--', label='cltt ratio, WebSky/Gaussian')
else:
    plt.plot(l, moving_average(cltt_websky_tot/(totalcls_avg[:,5]),window_size=15), color='darksalmon', alpha=0.8, linestyle='--', label='cltt2 ratio, WebSky/Gaussian')
plt.axhline(y=1, color='k', linestyle='--')
plt.xlabel('$\ell$')
plt.title(f'Total Spectra Comparison, WebSky/Gaussian')
plt.legend(fontsize='x-small')
plt.xscale('log')
plt.xlim(300,lmax)
plt.ylim(0.95,1.05)
plt.savefig(dir_out+f'/figs/websky_spectra_ratio_comparison_vs_websky_{append}.png',bbox_inches='tight')
'''

plt.figure(1)
plt.clf()
plt.axhline(y=1, color='k', linestyle='--')
plt.plot(l, moving_average(slte_websky/(slte),window_size=15), color='palegoldenrod', alpha=0.8, linestyle='--', label='slte ratio, WebSky/Gaussian')
plt.plot(l, moving_average(slbb_websky/(slbb),window_size=15), color='powderblue', alpha=0.8, linestyle='--', label='slbb ratio, WebSky/Gaussian')
plt.plot(l, moving_average(slee_websky/(slee),window_size=15), color='lightgreen', alpha=0.8, linestyle='--', label='slee ratio, WebSky/Gaussian')
if append == 'standard':
    plt.plot(l, moving_average(sltt_websky/(sltt),window_size=15), color='darksalmon', alpha=0.8, linestyle='--', label='sltt ratio, WebSky/Gaussian')
else:
    plt.plot(l, moving_average(sltt_websky/(sltt),window_size=15), color='darksalmon', alpha=0.8, linestyle='--', label='sltt ratio, WebSky/Gaussian')
plt.xlabel('$\ell$')
plt.title(f'Lensed CMB Spectra Comparison, WebSky/Gaussian')
plt.legend(fontsize='x-small')
plt.xscale('log')
plt.xlim(300,lmax)
plt.ylim(0.95,1.05)
plt.savefig(dir_out+f'/figs/websky_lcmb_spectra_ratio_comparison_vs_websky_{append}.png',bbox_inches='tight')

plt.figure(1)
plt.clf()
plt.axhline(y=1, color='k', linestyle='--')
plt.plot(l, moving_average(nlbb_websky/(ret[:,2]),window_size=15), color='powderblue', alpha=0.8, linestyle='--', label='nlbb ratio, Gaussian/From ILC-ed Spectra')
plt.plot(l, moving_average(nlee_websky/(ret[:,1]),window_size=15), color='lightgreen', alpha=0.8, linestyle='--', label='nlee ratio, Gaussian/From ILC-ed Spectra')
if append == 'standard':
    plt.plot(l, moving_average(nltt_websky/(ret[:,0]),window_size=15), color='darksalmon', alpha=0.8, linestyle='--', label='nltt ratio, Gaussian/From ILC-ed Spectra')
else:
    plt.plot(l, moving_average(nltt_websky/(ret[:,0]),window_size=15), color='darksalmon', alpha=0.8, linestyle='--', label='nltt2 ratio, Gaussian/From ILC-ed Spectra')
plt.xlabel('$\ell$')
plt.title(f'Noise Spectra Comparison, Gaussian/From ILC-ed Spectra')
plt.legend(fontsize='x-small')
plt.xscale('log')
plt.xlim(300,lmax)
plt.ylim(0.95,1.05)
plt.savefig(dir_out+f'/figs/websky_n_spectra_ratio_comparison_vs_websky_{append}.png',bbox_inches='tight')

'''
plt.figure(1)
plt.clf()
plt.axhline(y=1, color='k', linestyle='--')
if append == 'standard':
    ell,sltt,slee,slbb,slte = utils.get_lensedcls('/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat',lmax=lmax)
    plt.plot(l, moving_average(fltt_websky/(totalcls_avg[:,0]-sltt-ret[:,0]-artificial_noise),window_size=15), color='darksalmon', alpha=0.8, linestyle='--', label='fltt ratio, WebSky/Gaussian')
else:
    ell,sltt,slee,slbb,slte = utils.get_lensedcls('/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat',lmax=lmax)
    plt.plot(l, moving_average(fltt_websky/(totalcls_avg[:,5]-sltt-ret[:,0]-artificial_noise),window_size=15), color='darksalmon', alpha=0.8, linestyle='--', label='fltt2 ratio, WebSky/Gaussian')
plt.xlabel('$\ell$')
plt.title(f'Foreground Spectra Comparison, WebSky/Gaussian')
plt.legend(fontsize='x-small')
plt.xscale('log')
plt.xlim(300,lmax)
plt.ylim(0.95,1.05)
plt.savefig(dir_out+f'/figs/websky_fg_spectra_ratio_comparison_vs_websky_{append}.png',bbox_inches='tight')

plt.figure(1)
plt.clf()
plt.axhline(y=1, color='k', linestyle='--')
plt.plot(l, moving_average(fnlte_websky/(totalcls_avg[:,3]-slte),window_size=15), color='palegoldenrod', alpha=0.8, linestyle='--', label='flte + nlte ratio, WebSky/Gaussian')
plt.plot(l, moving_average(fnlbb_websky/(totalcls_avg[:,2]-slbb),window_size=15), color='powderblue', alpha=0.8, linestyle='--', label='flbb + nlbb ratio, WebSky/Gaussian')
plt.plot(l, moving_average(fnlee_websky/(totalcls_avg[:,1]-slee),window_size=15), color='lightgreen', alpha=0.8, linestyle='--', label='flee + nlee ratio, WebSky/Gaussian')
if append == 'standard':
    plt.plot(l, moving_average(fnltt_websky/(totalcls_avg[:,0]-sltt),window_size=15), color='darksalmon', alpha=0.8, linestyle='--', label='fltt + nltt ratio, WebSky/Gaussian')
else:
    plt.plot(l, moving_average(fnltt_websky/(totalcls_avg[:,5]-sltt),window_size=15), color='darksalmon', alpha=0.8, linestyle='--', label='fltt2 + nltt ratio, WebSky/Gaussian')
plt.xlabel('$\ell$')
plt.title(f'Foreground + Noise Spectra Comparison, WebSky/Gaussian')
plt.legend(fontsize='x-small')
plt.xscale('log')
plt.xlim(300,lmax)
plt.ylim(0.95,1.05)
plt.savefig(dir_out+f'/figs/websky_nfg_spectra_ratio_comparison_vs_websky_{append}.png',bbox_inches='tight')

plt.figure(1)
plt.clf()
plt.plot(l, fltt_websky, color='darksalmon', alpha=0.8, linestyle='--', label='fltt, WebSky')
plt.plot(l, totalcls_avg[:,0]-sltt-ret[:,0]-artificial_noise, color='cornflowerblue', alpha=0.8, linestyle='--', label='cltt - sltt - nltt, Gaussian')
plt.plot(l, fltt, color='lightgreen', alpha=0.8, linestyle='--', label='fltt, Gaussian sim 1')
plt.grid(True, linestyle="--", alpha=0.5)
plt.xlabel('$L$')
plt.title(f'Input fltt',pad=10)
plt.legend(loc='upper right', fontsize='small')
plt.xscale('log')
plt.yscale('log')
plt.xlim(10,lmax)
plt.tight_layout()
plt.savefig(dir_out+'/figs/temp.png',bbox_inches='tight')
'''
