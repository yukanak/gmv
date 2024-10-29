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
lmin = config['lensrec']['lminT']
lmaxT = config['lensrec']['lmaxT']
lmaxP = config['lensrec']['lmaxP']
nside = config['lensrec']['nside']
dir_out = config['dir_out']
cltype = config['lensrec']['cltype']
append = f'profhrd_tszfg'
l = np.arange(0,lmax+1)
sl = {ee:config['cls'][cltype][ee] for ee in config['cls'][cltype].keys()}
ell,sltt,slee,slbb,slte = utils.get_lensedcls('/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat' ,lmax=lmax)
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
# Foreground curves (tSZ)
fg_curves = pickle.load(open('fg_curves/agora_tsz_spec.pk','rb'))
tsz_curve_095_095 = fg_curves['masked']['95x95']
tsz_curve_150_150 = fg_curves['masked']['150x150']
tsz_curve_220_220 = fg_curves['masked']['220x220']
tsz_curve_095_150 = fg_curves['masked']['95x150']
tsz_curve_095_220 = fg_curves['masked']['95x220']
tsz_curve_150_220 = fg_curves['masked']['150x220']
# Foreground curves (Agora spectra)
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
# These are from Srini... Weird format. Assumes either one or two spectral energy distributions for CIB
w_cib_null = np.load('ilc_weights/weights_cmb_mv_cmbfree_cibfree_spt3g1920.npy',allow_pickle=True)
w_cib_null_95 = w_cib_null.item()['cmbcibfree'][95][1]
w_cib_null_150 = w_cib_null.item()['cmbcibfree'][150][1]
w_cib_null_220 = w_cib_null.item()['cmbcibfree'][220][1]
w_cib_null_srini = np.vstack((w_cib_null_95,w_cib_null_150,w_cib_null_220))
w_cib_null_2 = np.load('ilc_weights/weights_cmb_mv_cmbfree_cibfreetwoSEDs_spt3g1920.npy',allow_pickle=True)
w_cib_null_2_95 = w_cib_null_2.item()['cmbcibfree'][95][1]
w_cib_null_2_150 = w_cib_null_2.item()['cmbcibfree'][150][1]
w_cib_null_2_220 = w_cib_null_2.item()['cmbcibfree'][220][1]
w_cib_null_2_srini = np.vstack((w_cib_null_2_95,w_cib_null_2_150,w_cib_null_2_220))
w_Tmv_srini_95 = w_cib_null.item()['cmbmv'][95][1]
w_Tmv_srini_150 = w_cib_null.item()['cmbmv'][150][1]
w_Tmv_srini_220 = w_cib_null.item()['cmbmv'][220][1]
w_Tmv_srini = np.vstack((w_Tmv_srini_95,w_Tmv_srini_150,w_Tmv_srini_220))
w_tsz_null_srini_95 = w_cib_null.item()['cmbtszfree'][95][1]
w_tsz_null_srini_150 = w_cib_null.item()['cmbtszfree'][150][1]
w_tsz_null_srini_220 = w_cib_null.item()['cmbtszfree'][220][1]
w_tsz_null_srini = np.vstack((w_tsz_null_srini_95,w_tsz_null_srini_150,w_tsz_null_srini_220))

profile_filename = 'fg_profiles/TT_srini_mvilc_foreground_residuals.pkl'    
if os.path.isfile(profile_filename):                                        
    u = pickle.load(open(profile_filename,'rb'))                            
else:                                                                       
    # Combine Agora TT cross frequency tSZ spectra with MV ILC weights to get ILC-ed foreground residuals
    ret = np.zeros((lmax+1))                                                
    b='tt'; c=1; w1=w_Tmv_srini; w2=w_Tmv_srini                             
    for ll in l:                                                            
        # At each ell, get 3x3 matrix with each block containing Cl for different freq combinations
        clmat = np.zeros((3,3))                                             
        clmat[0,0] = tsz_curve_095_095[ll]                                  
        clmat[1,1] = tsz_curve_150_150[ll]                                  
        clmat[2,2] = tsz_curve_220_220[ll]                                  
        clmat[0,1] = clmat[1,0] = tsz_curve_095_150[ll]                     
        clmat[0,2] = clmat[2,0] = tsz_curve_095_220[ll]                     
        clmat[1,2] = clmat[2,1] = tsz_curve_150_220[ll]                     
        ret[ll] = np.dot(w1[:,ll], np.dot(clmat, w2[:,ll].T))               
    # Use the TT ILC-ed foreground residuals as the profile                 
    u = ret                                                                 
    with open(profile_filename,'wb') as f:                                  
        pickle.dump(u,f)

'''
plt.figure(0)
plt.clf()
plt.plot(ell, nltt_090_090[:lmax+1], color='firebrick', linestyle='-', label='nltt 90 x 90 from file')
plt.plot(ell, nltt_150_150[:lmax+1], color='sienna', linestyle='-', label='nltt 150 x 150 from file')
plt.plot(ell, nltt_220_220[:lmax+1], color='orange', linestyle='-', label='nltt 220 x 220 from file')
plt.plot(ell, tsz_curve_095_095[:lmax+1], color='pink', linestyle='-', label='tsz 95 x 95 from file')
plt.plot(ell, tsz_curve_150_150[:lmax+1], color='sandybrown', linestyle='-', label='tsz 150 x 150 from file')
plt.plot(ell, tsz_curve_220_220[:lmax+1], color='bisque', linestyle='-', label='tsz 220 x 220 from file')
plt.xscale('log')
plt.yscale('log')
plt.xlim(10,lmax)
#plt.ylim(1e4,1e10)
#plt.ylim(1e-6,1e2)
plt.legend(loc='center left', bbox_to_anchor=(1,0.5))
plt.ylabel("$C_\ell$")
plt.xlabel('$\ell$')
plt.savefig(dir_out+f'/figs/test.png',bbox_inches='tight')

# If lmaxT != lmaxP, we add artificial noise in TT for ell > lmaxT
artificial_noise = np.zeros(lmax+1)
artificial_noise[lmaxT+2:] = 1.e10
# Combine cross frequency spectra with ILC weights
ret = np.zeros((lmax+1,4))
for a in range(4):
    if a == 0: b='tt'; c=1; w1=w_Tmv; w2=w_Tmv
    if a == 1: b='tt'; c=1; w1=w_Tmv_srini; w2=w_Tmv_srini
    if a == 2: b='tt'; c=1; w1=w_tsz_null; w2=w_tsz_null
    if a == 3: b='tt'; c=1; w1=w_tsz_null_srini; w2=w_tsz_null_srini
    for ll in l:
        # At each ell, have 3x3 matrix with each block containing Cl for different frequency combinations
        clmat = np.zeros((3,3))
        clmat[0,0] = sl[b][ll] + noise_curves_090_090[ll,c] + tsz_curve_095_095[ll]
        clmat[1,1] = sl[b][ll] + noise_curves_150_150[ll,c] + tsz_curve_150_150[ll]
        clmat[2,2] = sl[b][ll] + noise_curves_220_220[ll,c] + tsz_curve_220_220[ll]
        clmat[0,1] = clmat[1,0] = sl[b][ll] + noise_curves_090_150[ll,c] + tsz_curve_095_150[ll]
        clmat[0,2] = clmat[2,0] = sl[b][ll] + noise_curves_090_220[ll,c] + tsz_curve_095_220[ll]
        clmat[1,2] = clmat[2,1] = sl[b][ll] + noise_curves_150_220[ll,c] + tsz_curve_150_220[ll]
        ret[ll,a]=np.dot(w1[:,ll], np.dot(clmat, w2[:,ll].T))

plt.figure(1)
plt.clf()
plt.plot(ell, ret[:,0], color='firebrick', linestyle='-', label='total TT, Yuuki MV-ILC')
plt.plot(ell, ret[:,1], color='pink', linestyle='--', label='total TT, Srini MV-ILC')
plt.plot(ell, ret[:,2], color='rebeccapurple', linestyle='-', label='total TT, Yuuki tSZ-nulled')
plt.plot(ell, ret[:,3], color='thistle', linestyle='--', label='total TT, Srini tSZ-nulled')
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-6,1e0)
plt.xlim(10,lmax)
plt.legend(loc='center left', bbox_to_anchor=(1,0.5))
plt.title('ILC-weighted TT autospectra')
plt.ylabel("$C_\ell$")
plt.xlabel('$\ell$')
plt.savefig(dir_out+f'/figs/ilc_weights_srini_yuuki_comparison.png',bbox_inches='tight')
'''

# If lmaxT != lmaxP, we add artificial noise in TT for ell > lmaxT
artificial_noise = np.zeros(lmax+1)
artificial_noise[lmaxT+2:] = 1.e10
# Combine cross frequency spectra with ILC weights
ret = np.zeros((lmax+1,3))
for a in range(3):
    if a == 0: b='tt'; c=1; w1=w_Tmv; w2=w_Tmv
    if a == 1: b='tt'; c=1; w1=w_cib_null_srini; w2=w_cib_null_srini
    if a == 2: b='tt'; c=1; w1=w_cib_null_2_srini; w2=w_cib_null_2_srini
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

plt.figure(2)
plt.clf()
plt.plot(ell, ret[:,0], color='firebrick', linestyle='-', label='TT Residuals, Yuuki MV-ILC')
plt.plot(ell, ret[:,1], color='pink', linestyle='--', label='TT Residuals, Srini CIB-nulled (One SED)')
plt.plot(ell, ret[:,2], color='rebeccapurple', linestyle='--', label='TT Residuals, Srini CIB-nulled (Two SEDs)')
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-6,1e0)
plt.xlim(10,lmax)
plt.legend(loc='upper left')
plt.title('ILC-weighted TT Residuals')
plt.ylabel("$C_\ell$")
plt.xlabel('$\ell$')
plt.savefig(dir_out+f'/figs/ilc_weights_mvilc_cib_residuals_comparison.png',bbox_inches='tight')
