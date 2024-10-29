import pickle
import sys, os
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import numpy as np
import matplotlib.pyplot as plt
import healqest_utils as utils
import healpy as hp

append_list = ['mh', 'crossilc_twoseds', 'crossilc_onesed']
config_file_list = ['test_yuka_lmaxT3500.yaml', 'test_yuka_lmaxT4000.yaml']
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

for config_file in config_file_list:
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
    ell,sltt,slee,slbb,slte = utils.get_lensedcls('/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat',lmax=lmax)
    n = 250

    for append in append_list:
        cltt1 = np.zeros(lmax+1)
        cltt2 = np.zeros(lmax+1)
        clttx = np.zeros(lmax+1)
        cltt3 = np.zeros(lmax+1)
        clee = np.zeros(lmax+1)
        clbb = np.zeros(lmax+1)
        clte = np.zeros(lmax+1)
        clt1e = np.zeros(lmax+1)
        clt2e = np.zeros(lmax+1)
        clt1t3 = np.zeros(lmax+1)
        clt2t3 = np.zeros(lmax+1)
        for i in np.arange(n)+1:
            # totalcls: T3T3, EE, BB, T3E, T1T1, T2T2, T1T2, T1T3, T2T3, T1E, T2E
            totalcls = np.load(dir_out+f'totalcls/totalcls_seed1_{i}_seed2_{i}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
            cltt3 += totalcls[:,0]
            clee += totalcls[:,1]
            clbb += totalcls[:,2]
            clte += totalcls[:,3]
            cltt1 += totalcls[:,4]
            cltt2 += totalcls[:,5]
            clttx += totalcls[:,6]
            clt1t3 += totalcls[:,7]
            clt2t3 += totalcls[:,8]
            clt1e += totalcls[:,9]
            clt2e += totalcls[:,10]
        cltt1 /= n
        cltt2 /= n
        clttx /= n
        cltt3 /= n
        clee /= n
        clbb /= n
        clte /= n
        clt1t3 /= n
        clt2t3 /= n
        clt1e /= n
        clt2e /= n
        totalcls_avg = np.vstack((cltt3,clee,clbb,clte,cltt1,cltt2,clttx,clt1t3,clt2t3,clt1e,clt2e)).T
        #totalcls_avg = np.vstack((cltt3,clee,clbb,clte)).T
        np.save(dir_out+f'totalcls/totalcls_average_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy',totalcls_avg)

        plt.figure(0)
        plt.clf()
        plt.plot(ell, sltt, color='firebrick', linestyle='-', label='sltt')
        plt.plot(ell, slee, color='forestgreen', linestyle='-', label='slee')
        plt.plot(ell, slbb, color='darkblue', linestyle='-', label='slbb')
        plt.plot(ell, slte, color='gold', linestyle='-', label='slte')
        plt.plot(ell, totalcls_avg[:,4], color='pink', linestyle='--', label='total T1T1')
        plt.plot(ell, totalcls_avg[:,5], color='darkorchid', linestyle='--', label='total T2T2')
        plt.plot(ell, totalcls_avg[:,0], color='darksalmon', linestyle='--', label='total T3T3')
        #plt.plot(ell, totalcls_avg[:,0], color='darksalmon', linestyle='--', label='total TT')
        plt.plot(ell, totalcls_avg[:,1], color='lightgreen', linestyle='--', label='total EE')
        plt.plot(ell, totalcls_avg[:,2], color='powderblue', linestyle='--', label='total BB')
        plt.plot(ell, totalcls_avg[:,3], color='palegoldenrod', linestyle='--', label='total TE')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(10,lmax)
        plt.ylim(1e-9,1e2)
        plt.legend(loc='center left', bbox_to_anchor=(1,0.5))
        plt.title(f'Average of Sims 1 Through {n}')
        plt.ylabel("$C_\ell$")
        plt.xlabel('$\ell$')
        plt.savefig(dir_out+f'/figs/totalcls_vs_signal_{append}_average_lmaxT{lmaxT}.png',bbox_inches='tight')
