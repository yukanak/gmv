import pickle
import sys, os
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import numpy as np
import matplotlib.pyplot as plt
import healqest_utils as utils
import healpy as hp

config_file = 'test_yuka.yaml'
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
ell,sltt,slee,slbb,slte = utils.get_lensedcls('/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat' ,lmax=lmax)
n = 99
# Dimension (3, 6001) for 90, 150, 220 GHz respectively
weights_mv_ilc_T = 'ilc_weights/weights1d_TT_spt3g_cmbmv.dat'
weights_mv_ilc_E = 'ilc_weights/weights1d_EE_spt3g_cmbmv.dat'
weights_mv_ilc_B = 'ilc_weights/weights1d_BB_spt3g_cmbmv.dat'
w_Tmv = np.loadtxt(weights_mv_ilc_T)
w_Emv = np.loadtxt(weights_mv_ilc_E)
w_Bmv = np.loadtxt(weights_mv_ilc_B)
w_srini = np.load('ilc_weights/weights_cmb_mv_cmbfree_cibfree_spt3g1920.npy',allow_pickle=True)
w_Tmv_srini_95 = w_srini.item()['cmbmv'][95][1]
w_Tmv_srini_150 = w_srini.item()['cmbmv'][150][1]
w_Tmv_srini_220 = w_srini.item()['cmbmv'][220][1]
w_Tmv_srini = np.vstack((w_Tmv_srini_95,w_Tmv_srini_150,w_Tmv_srini_220))
noise_curves_090_090 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_090_090.txt'))
noise_curves_150_150 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_150_150.txt'))
noise_curves_220_220 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_220_220.txt'))
noise_curves_090_150 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_090_150.txt'))
noise_curves_090_220 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_090_220.txt'))
noise_curves_150_220 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_150_220.txt'))
# Foreground curves
fg_curves = pickle.load(open('fg_curves/agora_tsz_spec.pk','rb'))
tsz_curve_095_095 = fg_curves['masked']['95x95']
tsz_curve_150_150 = fg_curves['masked']['150x150']
tsz_curve_220_220 = fg_curves['masked']['220x220']
tsz_curve_095_150 = fg_curves['masked']['95x150']
tsz_curve_095_220 = fg_curves['masked']['95x220']
tsz_curve_150_220 = fg_curves['masked']['150x220']
# Agora sims
agora_095 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_spt3g_95ghz_alm_lmax4096.fits'
agora_150 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_spt3g_150ghz_alm_lmax4096.fits'
agora_220 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims/agora_spt3g_220ghz_alm_lmax4096.fits'
tlm_095, elm_095, blm_095 = hp.read_alm(agora_095,hdu=[1,2,3])
tlm_150, elm_150, blm_150 = hp.read_alm(agora_150,hdu=[1,2,3])
tlm_220, elm_220, blm_220 = hp.read_alm(agora_220,hdu=[1,2,3])
nlm_090_filename = dir_out + f'nlm/nlm_090_lmax{lmax}_seed1.alm'
nlm_150_filename = dir_out + f'nlm/nlm_150_lmax{lmax}_seed1.alm'
nlm_220_filename = dir_out + f'nlm/nlm_220_lmax{lmax}_seed1.alm'
nlmt_090,nlme_090,nlmb_090 = hp.read_alm(nlm_090_filename,hdu=[1,2,3])
nlmt_150,nlme_150,nlmb_150 = hp.read_alm(nlm_150_filename,hdu=[1,2,3])
nlmt_220,nlme_220,nlmb_220 = hp.read_alm(nlm_220_filename,hdu=[1,2,3])
tlm_095 += nlmt_090; elm_095 += nlme_090; blm_095 += nlmb_090
tlm_150 += nlmt_150; elm_150 += nlme_150; blm_150 += nlmb_150
tlm_220 += nlmt_220; elm_220 += nlme_220; blm_220 += nlmb_220
tlm = hp.almxfl(tlm_095,w_Tmv[0][:lmax+1]) + hp.almxfl(tlm_150,w_Tmv[1][:lmax+1]) + hp.almxfl(tlm_220,w_Tmv[2][:lmax+1])
elm = hp.almxfl(elm_095,w_Emv[0][:lmax+1]) + hp.almxfl(elm_150,w_Emv[1][:lmax+1]) + hp.almxfl(elm_220,w_Emv[2][:lmax+1])
blm = hp.almxfl(blm_095,w_Bmv[0][:lmax+1]) + hp.almxfl(blm_150,w_Bmv[1][:lmax+1]) + hp.almxfl(blm_220,w_Bmv[2][:lmax+1])
# totalcls for standard case
totalcls = np.load(dir_out+f'totalcls/totalcls_average_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_standard.npy')
cltt_standard = totalcls[:,0]; clee_standard = totalcls[:,1]; clbb_standard = totalcls[:,2]; clte_standard = totalcls[:,3]
# Get spectra from Agora sims
artificial_noise = np.zeros(lmax+1)
artificial_noise[lmaxT+2:] = 1.e10
cltt_agora = hp.alm2cl(tlm,tlm) + artificial_noise
clee_agora = hp.alm2cl(elm,elm)
clbb_agora = hp.alm2cl(blm,blm)
clte_agora = hp.alm2cl(tlm,elm)

plt.figure(0)
plt.clf()
plt.plot(ell, cltt_agora, color='firebrick', linestyle='-', label='total TT, Agora')
plt.plot(ell, clee_agora, color='forestgreen', linestyle='-', label='total EE, Agora')
plt.plot(ell, clbb_agora, color='darkblue', linestyle='-', label='total BB, Agora')
plt.plot(ell, clte_agora, color='gold', linestyle='-', label='total TE, Agora')
plt.plot(ell, totalcls[:,0], color='darksalmon', linestyle='--', label='total TT, standard')
plt.plot(ell, totalcls[:,1], color='lightgreen', linestyle='--', label='total EE, standard')
plt.plot(ell, totalcls[:,2], color='powderblue', linestyle='--', label='total BB, standard')
plt.plot(ell, totalcls[:,3], color='palegoldenrod', linestyle='--', label='total TE, standard')
plt.xscale('log')
plt.yscale('log')
plt.xlim(10,lmax)
plt.ylim(1e-9,1e2)
plt.legend(loc='center left', bbox_to_anchor=(1,0.5))
plt.title(f'Agora spectra check')
plt.ylabel("$C_\ell$")
plt.xlabel('$\ell$')
plt.savefig(dir_out+f'/figs/totalcls_agora_check.png',bbox_inches='tight')

