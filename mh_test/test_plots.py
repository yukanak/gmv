import sys, os
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import numpy as np
import matplotlib.pyplot as plt
import utils
import healpy as hp

config_file = 'mh_yuka.yaml'
config = utils.parse_yaml(config_file)
lmax = config['lmax']
nside = config['nside']
dir_out = config['dir_out']
lmaxT = config['lmaxT']
lmaxP = config['lmaxP']
cltype = config['cltype']
append = f'mh'
l = np.arange(0,lmax+1)
sl = {ee:config['cls'][cltype][ee] for ee in config['cls'][cltype].keys()}

ell,sltt,slee,slbb,slte = utils.get_lensedcls('/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat' ,lmax=lmax)
sim1 = np.load(dir_out+f'totalcls/totalcls_seed1_1_seed2_1_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
sim2 = np.load(dir_out+f'totalcls/totalcls_seed1_2_seed2_2_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
old_fsky_corr=25.308939726920805
old_noise_file='/home/users/yukanaka/gmv/prfhrd_test/nl_cmbmv_20192020.dat'
old_noise_curves = np.loadtxt(old_noise_file)
old_nltt = old_fsky_corr * old_noise_curves[:,1]; old_nlee = old_fsky_corr * old_noise_curves[:,2]; old_nlbb = old_fsky_corr * old_noise_curves[:,2]
# If lmaxT != lmaxP, we add artificial noise in TT for ell > lmaxT
artificial_noise = np.zeros(lmax+1)
artificial_noise[lmaxT+2:] = 1.e10
# Resulting old spectra
cltt_old = sl['tt'][:lmax+1] + old_nltt[:lmax+1] + artificial_noise
clee_old = sl['ee'][:lmax+1] + old_nlee[:lmax+1]
clbb_old = sl['bb'][:lmax+1] + old_nlbb[:lmax+1]
clte_old = sl['te'][:lmax+1]

plt.figure(0)
plt.clf()
#plt.plot(ell, sl['tt'][:lmax+1] * (l*(l+1))/(2*np.pi), color='firebrick', linestyle='-', label='sltt')
#plt.plot(ell, sl['ee'][:lmax+1] * (l*(l+1))/(2*np.pi), color='forestgreen', linestyle='-', label='slee')
#plt.plot(ell, sl['bb'][:lmax+1] * (l*(l+1))/(2*np.pi), color='darkblue', linestyle='-', label='slbb')
#plt.plot(ell, sl['te'][:lmax+1] * (l*(l+1))/(2*np.pi), color='gold', linestyle='-', label='slte')
plt.plot(ell, cltt_old, color='firebrick', linestyle='-', label='total TT, old')
plt.plot(ell, clee_old, color='forestgreen', linestyle='-', label='total EE, old')
plt.plot(ell, clbb_old, color='darkblue', linestyle='-', label='total BB, old')
plt.plot(ell, clte_old, color='gold', linestyle='-', label='total TE, old')
plt.plot(ell, sim1[:,0], color='pink', linestyle='--', label='total TT')
plt.plot(ell, sim1[:,1], color='lightgreen', linestyle='--', label='total EE')
plt.plot(ell, sim1[:,2], color='powderblue', linestyle='--', label='total BB')
plt.plot(ell, sim1[:,3], color='palegoldenrod', linestyle='--', label='total TE')
plt.xscale('log')
plt.yscale('log')
plt.xlim(1000,lmax)
#plt.ylim(1e4,1e10)
plt.ylim(1e-9,1e-1)
plt.legend(loc='center left', bbox_to_anchor=(1,0.5))
plt.title('sim 1')
plt.ylabel("$C_\ell$")
plt.xlabel('$\ell$')
#plt.savefig(dir_out+f'/figs/totalcls_vs_signal_mh_sim1.png',bbox_inches='tight')
plt.savefig(dir_out+f'/figs/totalcls_mh_vs_old_sim1.png',bbox_inches='tight')

plt.clf()
plt.plot(ell, sltt * (l*(l+1))**2/4, color='firebrick', linestyle='-', label='sltt')
plt.plot(ell, slee * (l*(l+1))**2/4, color='forestgreen', linestyle='-', label='slee')
plt.plot(ell, slbb * (l*(l+1))**2/4, color='darkblue', linestyle='-', label='slbb')
plt.plot(ell, slte * (l*(l+1))**2/4, color='gold', linestyle='-', label='slte')
plt.plot(ell, sim2[:,0] * (l*(l+1))**2/4, color='pink', linestyle='--', label='total TT')
plt.plot(ell, sim2[:,1] * (l*(l+1))**2/4, color='lightgreen', linestyle='--', label='total EE')
plt.plot(ell, sim2[:,2] * (l*(l+1))**2/4, color='powderblue', linestyle='--', label='total BB')
plt.plot(ell, sim2[:,3] * (l*(l+1))**2/4, color='palegoldenrod', linestyle='--', label='total TE')
plt.xscale('log')
plt.yscale('log')
plt.xlim(10,lmax)
plt.ylim(1e4,1e10)
plt.legend(loc='center left', bbox_to_anchor=(1,0.5))
plt.title('sim 2')
plt.ylabel("$C_\ell^{\kappa\kappa}$")
plt.xlabel('$\ell$')
plt.savefig(dir_out+f'/figs/totalcls_vs_signal_mh_sim2.png',bbox_inches='tight')

noise_file_090_090='noise_curves/nl_fromstack_090_090.txt'
noise_file_150_150='noise_curves/nl_fromstack_150_150.txt'
noise_file_220_220='noise_curves/nl_fromstack_220_220.txt'
noise_file_090_150='noise_curves/nl_fromstack_090_150.txt'
noise_file_090_220='noise_curves/nl_fromstack_090_220.txt'
noise_file_150_220='noise_curves/nl_fromstack_150_220.txt'
fsky_corr=1
noise_curves_090_090 = np.nan_to_num(np.loadtxt(noise_file_090_090))
noise_curves_150_150 = np.nan_to_num(np.loadtxt(noise_file_150_150))
noise_curves_220_220 = np.nan_to_num(np.loadtxt(noise_file_220_220))
noise_curves_090_150 = np.nan_to_num(np.loadtxt(noise_file_090_150))
noise_curves_090_220 = np.nan_to_num(np.loadtxt(noise_file_090_220))
noise_curves_150_220 = np.nan_to_num(np.loadtxt(noise_file_150_220))
nltt_090_090 = fsky_corr * noise_curves_090_090[:,1]; nlee_090_090 = fsky_corr * noise_curves_090_090[:,2]; nlbb_090_090 = fsky_corr * noise_curves_090_090[:,3]
nltt_150_150 = fsky_corr * noise_curves_150_150[:,1]; nlee_150_150 = fsky_corr * noise_curves_150_150[:,2]; nlbb_150_150 = fsky_corr * noise_curves_150_150[:,3]
nltt_220_220 = fsky_corr * noise_curves_220_220[:,1]; nlee_220_220 = fsky_corr * noise_curves_220_220[:,2]; nlbb_220_220 = fsky_corr * noise_curves_220_220[:,3]
nltt_090_150 = fsky_corr * noise_curves_090_150[:,1]; nlee_090_150 = fsky_corr * noise_curves_090_150[:,2]; nlbb_090_150 = fsky_corr * noise_curves_090_150[:,3]
nltt_090_220 = fsky_corr * noise_curves_090_220[:,1]; nlee_090_220 = fsky_corr * noise_curves_090_220[:,2]; nlbb_090_220 = fsky_corr * noise_curves_090_220[:,3]
nltt_150_220 = fsky_corr * noise_curves_150_220[:,1]; nlee_150_220 = fsky_corr * noise_curves_150_220[:,2]; nlbb_150_220 = fsky_corr * noise_curves_150_220[:,3]
nlm1_090_filename = dir_out + f'nlm/nlm_090_lmax{lmax}_seed1.alm'
nlm1_150_filename = dir_out + f'nlm/nlm_150_lmax{lmax}_seed1.alm'
nlm1_220_filename = dir_out + f'nlm/nlm_220_lmax{lmax}_seed1.alm'
nlm2_090_filename = dir_out + f'nlm/nlm_090_lmax{lmax}_seed2.alm'
nlm2_150_filename = dir_out + f'nlm/nlm_150_lmax{lmax}_seed2.alm'
nlm2_220_filename = dir_out + f'nlm/nlm_220_lmax{lmax}_seed2.alm'
nlmt1_090,nlme1_090,nlmb1_090 = hp.read_alm(nlm1_090_filename,hdu=[1,2,3])
nlmt1_150,nlme1_150,nlmb1_150 = hp.read_alm(nlm1_150_filename,hdu=[1,2,3])
nlmt1_220,nlme1_220,nlmb1_220 = hp.read_alm(nlm1_220_filename,hdu=[1,2,3])
sim1_nltt_090_090 = hp.alm2cl(nlmt1_090)
sim1_nlee_090_090 = hp.alm2cl(nlme1_090)
sim1_nlbb_090_090 = hp.alm2cl(nlmb1_090)
sim1_nltt_150_150 = hp.alm2cl(nlmt1_150)
sim1_nlee_150_150 = hp.alm2cl(nlme1_150)
sim1_nlbb_150_150 = hp.alm2cl(nlmb1_150)
sim1_nltt_220_220 = hp.alm2cl(nlmt1_220)
sim1_nlee_220_220 = hp.alm2cl(nlme1_220)
sim1_nlbb_220_220 = hp.alm2cl(nlmb1_220)
sim1_nltt_090_150 = hp.alm2cl(nlmt1_090,nlmt1_150)
sim1_nlee_090_150 = hp.alm2cl(nlme1_090,nlmt1_150)
sim1_nlbb_090_150 = hp.alm2cl(nlmb1_090,nlmt1_150)
sim1_nltt_090_220 = hp.alm2cl(nlmt1_090,nlmt1_220)
sim1_nlee_090_220 = hp.alm2cl(nlme1_090,nlmt1_220)
sim1_nlbb_090_220 = hp.alm2cl(nlmb1_090,nlmt1_220)
sim1_nltt_150_220 = hp.alm2cl(nlmt1_150,nlmt1_220)
sim1_nlee_150_220 = hp.alm2cl(nlme1_150,nlmt1_220)
sim1_nlbb_150_220 = hp.alm2cl(nlmb1_150,nlmt1_220)
nlev_t = 5
nlev_p = 5
nltt = (np.pi/180./60.*nlev_t)**2
nlpp = (np.pi/180./60.*nlev_p)**2

plt.figure(1)
plt.clf()
plt.plot(ell, nltt_090_090[:lmax+1], color='firebrick', linestyle='-', label='nltt 90 x 90 from file')
plt.plot(ell, nltt_150_150[:lmax+1], color='sienna', linestyle='-', label='nltt 150 x 150 from file')
plt.plot(ell, nltt_220_220[:lmax+1], color='orange', linestyle='-', label='nltt 220 x 220 from file')
plt.plot(ell, sim1_nltt_090_090, color='pink', linestyle='--', label='nltt 90 x 90 from sims')
plt.plot(ell, sim1_nltt_150_150, color='sandybrown', linestyle='--', label='nltt 150 x 150 from sims')
plt.plot(ell, sim1_nltt_220_220, color='bisque', linestyle='--', label='nltt 220 x 220 from sims')
plt.plot(ell, nlee_090_090[:lmax+1], color='forestgreen', linestyle='-', label='nlee 90 x 90 from file')
plt.plot(ell, nlee_150_150[:lmax+1], color='lightseagreen', linestyle='-', label='nlee 150 x 150 from file')
plt.plot(ell, nlee_220_220[:lmax+1], color='olive', linestyle='-', label='nlee 220 x 220 from file')
plt.plot(ell, sim1_nlee_090_090, color='lightgreen', linestyle='--', label='nlee 90 x 90 from sims')
plt.plot(ell, sim1_nlee_150_150, color='mediumaquamarine', linestyle='--', label='nlee 150 x 150 from sims')
plt.plot(ell, sim1_nlee_220_220, color='darkseagreen', linestyle='--', label='nlee 220 x 220 from sims')
plt.plot(ell, nlbb_090_090[:lmax+1], color='darkblue', linestyle='-', label='nlbb 90 x 90 from file')
plt.plot(ell, nlbb_150_150[:lmax+1], color='rebeccapurple', linestyle='-', label='nlbb 150 x 150 from file')
plt.plot(ell, nlbb_220_220[:lmax+1], color='steelblue', linestyle='-', label='nlbb 220 x 220 from file')
plt.plot(ell, sim1_nlbb_090_090, color='cornflowerblue', linestyle='--', label='nlbb 90 x 90 from sims')
plt.plot(ell, sim1_nlbb_150_150, color='thistle', linestyle='--', label='nlbb 150 x 150 from sims')
plt.plot(ell, sim1_nlbb_220_220, color='lightsteelblue', linestyle='--', label='nlbb 220 x 220 from sims')
plt.axhline(y=nlpp, color='darkgray', linestyle='--', label='5 uK-arcmin')
plt.xscale('log')
plt.yscale('log')
plt.xlim(200,lmax)
#plt.ylim(1e3,1e10)
plt.ylim(1e-7,1e1)
plt.legend(loc='center left', bbox_to_anchor=(1,0.5))
plt.title('sim 1')
plt.ylabel("$N_\ell$")
plt.xlabel('$\ell$')
plt.savefig(dir_out+f'/figs/noise_spectra_mh_sim1.png',bbox_inches='tight')

plt.clf()
# Expect around 5 uK-arcmin
plt.plot(ell, nltt_090_150[:lmax+1], color='firebrick', linestyle='-', label='nltt 90 x 150 from file')
plt.plot(ell, nltt_090_220[:lmax+1], color='sienna', linestyle='-', label='nltt 90 x 220 from file')
plt.plot(ell, nltt_150_220[:lmax+1], color='orange', linestyle='-', label='nltt 150 x 220 from file')
plt.plot(ell, sim1_nltt_090_150, color='pink', linestyle='--', label='nltt 90 x 150 from sims')
plt.plot(ell, sim1_nltt_090_220, color='sandybrown', linestyle='--', label='nltt 90 x 220 from sims')
plt.plot(ell, sim1_nltt_150_220, color='bisque', linestyle='--', label='nltt 150 x 220 from sims')
plt.plot(ell, nlee_090_150[:lmax+1], color='forestgreen', linestyle='-', label='nlee 90 x 150 from file')
plt.plot(ell, nlee_090_220[:lmax+1], color='lightseagreen', linestyle='-', label='nlee 90 x 220 from file')
plt.plot(ell, nlee_150_220[:lmax+1], color='olive', linestyle='-', label='nlee 150 x 220 from file')
plt.plot(ell, sim1_nlee_090_150, color='lightgreen', linestyle='--', label='nlee 90 x 150 from sims')
plt.plot(ell, sim1_nlee_090_220, color='mediumaquamarine', linestyle='--', label='nlee 90 x 220 from sims')
plt.plot(ell, sim1_nlee_150_220, color='darkseagreen', linestyle='--', label='nlee 150 x 220 from sims')
plt.plot(ell, nlbb_090_150[:lmax+1], color='darkblue', linestyle='-', label='nlbb 90 x 150 from file')
plt.plot(ell, nlbb_090_220[:lmax+1], color='rebeccapurple', linestyle='-', label='nlbb 90 x 220 from file')
plt.plot(ell, nlbb_150_220[:lmax+1], color='steelblue', linestyle='-', label='nlbb 150 x 220 from file')
plt.plot(ell, sim1_nlbb_090_150, color='cornflowerblue', linestyle='--', label='nlbb 90 x 150 from sims')
plt.plot(ell, sim1_nlbb_090_220, color='thistle', linestyle='--', label='nlbb 90 x 220 from sims')
plt.plot(ell, sim1_nlbb_150_220, color='lightsteelblue', linestyle='--', label='nlbb 150 x 220 from sims')
plt.axhline(y=nlpp, color='darkgray', linestyle='--', label='5 uK-arcmin')
plt.xscale('log')
plt.yscale('log')
plt.xlim(200,lmax)
#plt.ylim(1e3,1e10)
plt.ylim(1e-7,1e1)
plt.legend(loc='center left', bbox_to_anchor=(1,0.5))
plt.title('sim 1')
plt.ylabel("$N_\ell$")
plt.xlabel('$\ell$')
plt.savefig(dir_out+f'/figs/noise_cross_spectra_mh_sim1.png',bbox_inches='tight')

