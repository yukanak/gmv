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
totalcls = np.load(dir_out+f'totalcls/totalcls_average_lmaxT3000_lmaxP4096_nside2048_mh.npy')
totalcls_mvTT = np.load(dir_out+f'totalcls/totalcls_mvTT_average_lmaxT3000_lmaxP4096_nside2048_mh.npy')
totalcls_tsznulledTT = np.load(dir_out+f'totalcls/totalcls_tsznulledTT_average_lmaxT3000_lmaxP4096_nside2048_mh.npy')
fsky_corr=1
noise_curves_090_090 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_090_090.txt'))
noise_curves_150_150 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_150_150.txt'))
noise_curves_220_220 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_220_220.txt'))
noise_curves_090_150 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_090_150.txt'))
noise_curves_090_220 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_090_220.txt'))
noise_curves_150_220 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_150_220.txt'))
nltt_090_090 = fsky_corr * noise_curves_090_090[:,1]; nlee_090_090 = fsky_corr * noise_curves_090_090[:,2]; nlbb_090_090 = fsky_corr * noise_curves_090_090[:,3]
nltt_150_150 = fsky_corr * noise_curves_150_150[:,1]; nlee_150_150 = fsky_corr * noise_curves_150_150[:,2]; nlbb_150_150 = fsky_corr * noise_curves_150_150[:,3]
nltt_220_220 = fsky_corr * noise_curves_220_220[:,1]; nlee_220_220 = fsky_corr * noise_curves_220_220[:,2]; nlbb_220_220 = fsky_corr * noise_curves_220_220[:,3]
nltt_090_150 = fsky_corr * noise_curves_090_150[:,1]; nlee_090_150 = fsky_corr * noise_curves_090_150[:,2]; nlbb_090_150 = fsky_corr * noise_curves_090_150[:,3]
nltt_090_220 = fsky_corr * noise_curves_090_220[:,1]; nlee_090_220 = fsky_corr * noise_curves_090_220[:,2]; nlbb_090_220 = fsky_corr * noise_curves_090_220[:,3]
nltt_150_220 = fsky_corr * noise_curves_150_220[:,1]; nlee_150_220 = fsky_corr * noise_curves_150_220[:,2]; nlbb_150_220 = fsky_corr * noise_curves_150_220[:,3]
# Shape (16501, 10); ells, TT, EE, BB, then cross spectra
fg_curves_090_090 = np.nan_to_num(np.loadtxt('fg_curves/cls_allfg90_allfg90.dat'))
fg_curves_150_150 = np.nan_to_num(np.loadtxt('fg_curves/cls_allfg150_allfg150.dat'))
fg_curves_220_220 = np.nan_to_num(np.loadtxt('fg_curves/cls_allfg220_allfg220.dat'))
fg_curves_090_150 = np.nan_to_num(np.loadtxt('fg_curves/cls_allfg90_allfg150.dat'))
fg_curves_090_220 = np.nan_to_num(np.loadtxt('fg_curves/cls_allfg90_allfg220.dat'))
fg_curves_150_220 = np.nan_to_num(np.loadtxt('fg_curves/cls_allfg150_allfg220.dat'))
# Dimension (3, 6001) for 90, 150, 220 GHz respectively
weights_tsz_null_T = 'ilc_weights/weights1d_TT_spt3g_cmbynull.dat'
weights_mv_ilc_T = 'ilc_weights/weights1d_TT_spt3g_cmbmv.dat'
weights_mv_ilc_E = 'ilc_weights/weights1d_EE_spt3g_cmbmv.dat'
weights_mv_ilc_B = 'ilc_weights/weights1d_BB_spt3g_cmbmv.dat'
w_tsz_null = np.loadtxt(weights_tsz_null_T)
w_Tmv = np.loadtxt(weights_mv_ilc_T)
w_Emv = np.loadtxt(weights_mv_ilc_E)
w_Bmv = np.loadtxt(weights_mv_ilc_B)

##########

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
plt.plot(ell, totalcls[:,0], color='pink', linestyle='--', label='total TT')
plt.plot(ell, totalcls[:,2], color='lightgreen', linestyle='--', label='total EE')
plt.plot(ell, totalcls[:,3], color='powderblue', linestyle='--', label='total BB')
plt.plot(ell, totalcls[:,4], color='palegoldenrod', linestyle='--', label='total TE')
plt.xscale('log')
plt.yscale('log')
plt.xlim(10,lmax)
#plt.ylim(1e4,1e10)
plt.ylim(1e-6,1e2)
plt.legend(loc='center left', bbox_to_anchor=(1,0.5))
plt.title('totalcls')
plt.ylabel("$C_\ell$")
plt.xlabel('$\ell$')
#plt.savefig(dir_out+f'/figs/totalcls_vs_signal_mh_sim1.png',bbox_inches='tight')
plt.savefig(dir_out+f'/figs/totalcls_mh_vs_old.png',bbox_inches='tight')

##########

# Combine cross frequency spectra with ILC weights
# Second dimension order TT tSZ-nulled x MV, TT MV x MV, TT tSZ-nulled x tSZ-nulled, EE, BB
ret = np.zeros((lmax+1,5))
for a in range(5):
    if a == 0: b='tt'; c=1; w1=w_Tmv; w2=w_tsz_null
    if a == 1: b='tt'; c=1; w1=w_Tmv; w2=w_Tmv
    if a == 2: b='tt'; c=1; w1=w_tsz_null; w2=w_tsz_null
    if a == 3: b='ee'; c=2; w1=w_Emv; w2=w_Emv
    if a == 4: b='bb'; c=3; w1=w_Bmv; w2=w_Bmv
    for ll in l:
        # At each ell, have 3x3 matrix with each block containing Cl for different frequency combinations
        clmat = np.zeros((3,3))
        clmat[0,0] = sl[b][ll] + noise_curves_090_090[ll,c] + fg_curves_090_090[ll,c]
        clmat[1,1] = sl[b][ll] + noise_curves_150_150[ll,c] + fg_curves_150_150[ll,c]
        clmat[2,2] = sl[b][ll] + noise_curves_220_220[ll,c] + fg_curves_220_220[ll,c]
        clmat[0,1] = clmat[1,0] = sl[b][ll] + noise_curves_090_150[ll,c] + fg_curves_090_150[ll,c]
        clmat[0,2] = clmat[2,0] = sl[b][ll] + noise_curves_090_220[ll,c] + fg_curves_090_220[ll,c]
        clmat[1,2] = clmat[2,1] = sl[b][ll] + noise_curves_150_220[ll,c] + fg_curves_150_220[ll,c]
        ret[ll,a]=np.dot(w1[:,ll], np.dot(clmat, w2[:,ll].T))

plt.figure(1)
plt.clf()
plt.plot(ell, ret[:,0], color='firebrick', linestyle='-', label='total TT, T1 cross T2')
plt.plot(ell, ret[:,1], color='forestgreen', linestyle='-', label='total TT, T1 auto (MV)')
plt.plot(ell, ret[:,2], color='darkblue', linestyle='-', label='total TT, T2 auto (tSZ-nulled)')
plt.plot(ell, ret[:,3], color='gold', linestyle='-', label='total EE')
plt.plot(ell, ret[:,4], color='rebeccapurple', linestyle='-', label='total BB')
plt.plot(ell, totalcls[:,0], color='lightgreen', linestyle='--', label='total TT, T1 auto from sims')
plt.plot(ell, totalcls[:,1], color='powderblue', linestyle='--', label='total TT, T2 auto from sims')
plt.plot(ell, totalcls[:,2], color='palegoldenrod', linestyle='--', label='total EE from sims')
plt.plot(ell, totalcls[:,3], color='thistle', linestyle='--', label='total BB from sims')
plt.xscale('log')
plt.yscale('log')
plt.xlim(1000,lmax)
plt.ylim(1e-6,1e-2)
plt.legend(loc='center left', bbox_to_anchor=(1,0.5))
plt.title('totalcls TT')
plt.ylabel("$C_\ell$")
plt.xlabel('$\ell$')
plt.savefig(dir_out+f'/figs/totalcls_mh_TT.png',bbox_inches='tight')

##########

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
a = (np.pi/180./60.*8.2)**2
b = (np.pi/180./60.*6.5)**2
c = (np.pi/180./60.*25)**2

plt.figure(2)
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
#plt.axhline(y=nlpp, color='darkgray', linestyle='--', label='5 uK-arcmin')
plt.axhline(y=a, color='gray', linestyle='--', label='8.2 uK-arcmin')
plt.axhline(y=b, color='darkgray', linestyle='--', label='6.5 uK-arcmin')
plt.axhline(y=c, color='dimgray', linestyle='--', label='25 uK-arcmin')
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

##########

# Make simplified "MV ILC" weights for BB residuals
w_inv_residuals_BB = np.zeros((3,lmax+1))
e = np.ones(3)
for ll in l:
    # At each ell, have 3x3 matrix with each block containing Cl for different frequency combinations
    clmat = np.zeros((3,3))
    clmat[0,0] = noise_curves_090_090[ll,3] + fg_curves_090_090[ll,3]
    clmat[1,1] = noise_curves_150_150[ll,3] + fg_curves_150_150[ll,3]
    clmat[2,2] = noise_curves_220_220[ll,3] + fg_curves_220_220[ll,3]
    clmat[0,1] = clmat[1,0] = noise_curves_090_150[ll,3] + fg_curves_090_150[ll,3]
    clmat[0,2] = clmat[2,0] = noise_curves_090_220[ll,3] + fg_curves_090_220[ll,3]
    clmat[1,2] = clmat[2,1] = noise_curves_150_220[ll,3] + fg_curves_150_220[ll,3]
    w_inv_residuals_BB[:,ll]=(e.T@np.linalg.pinv(clmat))/(e.T@np.linalg.pinv(clmat)@e)

# Get MV ILC spectrum from Yuuki's weights and "MV ILC" spectrum using the 1/noise+fg weights
ret = np.zeros((lmax+1,2))
for a in range(2):
    if a == 0: b='bb'; c=3; w1=w_Bmv; w2=w_Bmv
    if a == 1: b='bb'; c=3; w1=w_inv_residuals_BB; w2=w_inv_residuals_BB
    for ll in l:
        # At each ell, have 3x3 matrix with each block containing Cl for different frequency combinations
        clmat = np.zeros((3,3))
        clmat[0,0] = noise_curves_090_090[ll,c] + fg_curves_090_090[ll,c]
        clmat[1,1] = noise_curves_150_150[ll,c] + fg_curves_150_150[ll,c]
        clmat[2,2] = noise_curves_220_220[ll,c] + fg_curves_220_220[ll,c]
        clmat[0,1] = clmat[1,0] = noise_curves_090_150[ll,c] + fg_curves_090_150[ll,c]
        clmat[0,2] = clmat[2,0] = noise_curves_090_220[ll,c] + fg_curves_090_220[ll,c]
        clmat[1,2] = clmat[2,1] = noise_curves_150_220[ll,c] + fg_curves_150_220[ll,c]
        ret[ll,a]=np.dot(w1[:,ll], np.dot(clmat, w2[:,ll].T))

plt.clf()
plt.plot(ell, nlbb_090_090[:lmax+1]+fg_curves_090_090[:lmax+1,3], color='darkblue', linestyle='-', label='nlbb+flbb 90 x 90 from file')
plt.plot(ell, nlbb_150_150[:lmax+1]+fg_curves_150_150[:lmax+1,3], color='rebeccapurple', linestyle='-', label='nlbb+flbb 150 x 150 from file')
plt.plot(ell, nlbb_220_220[:lmax+1]+fg_curves_220_220[:lmax+1,3], color='steelblue', linestyle='-', label='nlbb+flbb 220 x 220 from file')
plt.plot(ell, nlbb_090_150[:lmax+1]+fg_curves_090_150[:lmax+1,3], color='cornflowerblue', linestyle='-', label='nlbb+flbb 90 x 150 from file')
plt.plot(ell, nlbb_090_220[:lmax+1]+fg_curves_090_220[:lmax+1,3], color='thistle', linestyle='-', label='nlbb+flbb 90 x 220 from file')
plt.plot(ell, nlbb_150_220[:lmax+1]+fg_curves_150_220[:lmax+1,3], color='lightsteelblue', linestyle='-', label='nlbb+flbb 150 x 220 from file')
plt.plot(ell, ret[:,0], color='gray', linestyle='-', label='nlbb+flbb MV ILC')
plt.plot(ell, ret[:,1], color='dimgray', linestyle='--', label='nlbb+flbb 1/residuals weights')
#plt.axhline(y=(np.pi/180./60.*8.2)**2, color='gray', linestyle='--', label='8.2 uK-arcmin')
#plt.axhline(y=(np.pi/180./60.*6.5)**2, color='darkgray', linestyle='--', label='6.5 uK-arcmin')
#plt.axhline(y=(np.pi/180./60.*25)**2, color='dimgray', linestyle='--', label='25 uK-arcmin')
plt.xscale('log')
plt.yscale('log')
plt.xlim(200,lmax)
plt.ylim(1e-7,1e-3)
plt.legend(loc='center left', bbox_to_anchor=(1,0.5))
plt.title('BB fg+noise spectra')
plt.ylabel("$C_\ell$")
plt.xlabel('$\ell$')
plt.savefig(dir_out+f'/figs/bb_residuals_spectra_mh.png',bbox_inches='tight')

