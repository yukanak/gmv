import pickle
import sys, os
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import numpy as np
import matplotlib.pyplot as plt
import healqest_utils as utils
import healpy as hp

#append_list = ['standard']
#append_list = ['mh', 'crossilc_twoseds', 'crossilc_onesed']
append_list = ['standard']
#config_file_list = ['test_yuka.yaml', 'test_yuka_lmaxT3500.yaml', 'test_yuka_lmaxT4000.yaml']
config_file_list = ['test_yuka_lmaxT3500.yaml']
#fg_models_list = ['agora', 'websky']
fg_models_list = ['agora']

def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

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
        for fg_model in fg_models_list:
            if append == 'crossilc_twoseds' and fg_model == 'websky':
                continue
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
                totalcls = np.load(dir_out+f'totalcls/totalcls_seed1_{i}_seed2_{i}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy')
                cltt3 += totalcls[:,0]
                clee += totalcls[:,1]
                clbb += totalcls[:,2]
                clte += totalcls[:,3]
                if append != 'standard':
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
            if append != 'standard':
                totalcls_avg = np.vstack((cltt3,clee,clbb,clte,cltt1,cltt2,clttx,clt1t3,clt2t3,clt1e,clt2e)).T
            else:
                totalcls_avg = np.vstack((cltt3,clee,clbb,clte)).T
            if not os.path.isfile(dir_out+f'totalcls/totalcls_average_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy'):
                np.save(dir_out+f'totalcls/totalcls_average_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{fg_model}_{append}.npy',totalcls_avg)

            plt.figure(0)
            plt.clf()
            plt.plot(ell, sltt, color='firebrick', linestyle='-', label='sltt')
            plt.plot(ell, slee, color='forestgreen', linestyle='-', label='slee')
            plt.plot(ell, slbb, color='darkblue', linestyle='-', label='slbb')
            plt.plot(ell, slte, color='gold', linestyle='-', label='slte')
            if append != 'standard':
                # totalcls: T3T3, EE, BB, T3E, T1T1, T2T2, T1T2, T1T3, T2T3, T1E, T2E
                plt.plot(ell, totalcls_avg[:,4], color='pink', linestyle='--', label='total T1T1')
                plt.plot(ell, totalcls_avg[:,5], color='darkorchid', linestyle='--', label='total T2T2')
                plt.plot(ell, totalcls_avg[:,0], color='darksalmon', linestyle='--', label='total T3T3')
            else:
                plt.plot(ell, totalcls_avg[:,0], color='darksalmon', linestyle='--', label='total TT')
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
            plt.savefig(dir_out+f'/figs/totalcls_vs_signal_{fg_model}_{append}_average_lmaxT{lmaxT}.png',bbox_inches='tight')

            if fg_model == 'agora' and (append == 'standard' or append == 'mh'):
                totalcls_old_filename = '/oak/stanford/orgs/kipac/users/yukanaka/outputs_with_frequency_separated_inputs/'+f'totalcls/totalcls_average_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy'
                totalcls_old = np.load(totalcls_old_filename)
                cltt_old_tot = totalcls_old[:,0]; clee_old_tot = totalcls_old[:,1]; clbb_old_tot = totalcls_old[:,2]; clte_old_tot = totalcls_old[:,3]
                if append == 'mh':
                    cltt2_old_tot = totalcls_old[:,5]

                alm_cmb_sim1 = f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/tqu1/len/alms/lensed_planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_teb1_seed1_alm_lmax{lmax}.fits'
                tlm1,elm1,blm1 = hp.read_alm(alm_cmb_sim1,hdu=[1,2,3])
                sltt = hp.alm2cl(tlm1,tlm1); slee = hp.alm2cl(elm1,elm1); slbb = hp.alm2cl(blm1,blm1); slte = hp.alm2cl(tlm1,elm1)

                # Get Agora sim (signal + foregrounds)
                agora_095 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_95ghz_alm_lmax4096.fits'
                agora_150 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_150ghz_alm_lmax4096.fits'
                agora_220 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_220ghz_alm_lmax4096.fits'
                # Lensed CMB-only Agora sims
                lcmb_095 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lcmb_spt3g_95ghz_alm_lmax4096.fits'
                lcmb_150 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lcmb_spt3g_150ghz_alm_lmax4096.fits'
                lcmb_220 = '/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_lcmb_spt3g_220ghz_alm_lmax4096.fits'
                # ILC weights
                # Dimension (3, 6001) for 90, 150, 220 GHz respectively
                w_tsz_null = np.loadtxt('ilc_weights/weights1d_TT_spt3g_cmbynull.dat')
                w_Tmv = np.loadtxt('ilc_weights/weights1d_TT_spt3g_cmbmv.dat')
                w_Emv = np.loadtxt('ilc_weights/weights1d_EE_spt3g_cmbmv.dat')
                w_Bmv = np.loadtxt('ilc_weights/weights1d_BB_spt3g_cmbmv.dat')
                # If lmaxT != lmaxP, we add artificial noise in TT for ell > lmaxT
                artificial_noise = np.zeros(lmax+1)
                artificial_noise[lmaxT+2:] = 1.e10
                # Get Agora lensed CMB + foregrounds
                tlm_95, elm_95, blm_95 = hp.read_alm(agora_095,hdu=[1,2,3])
                tlm_150, elm_150, blm_150 = hp.read_alm(agora_150,hdu=[1,2,3])
                tlm_220, elm_220, blm_220 = hp.read_alm(agora_220,hdu=[1,2,3])
                if append == 'standard':
                    tlm = hp.almxfl(tlm_95,w_Tmv[0][:lmax+1]) + hp.almxfl(tlm_150,w_Tmv[1][:lmax+1]) + hp.almxfl(tlm_220,w_Tmv[2][:lmax+1])
                elif append == 'mh':
                    tlm = hp.almxfl(tlm_95,w_tsz_null[0][:lmax+1]) + hp.almxfl(tlm_150,w_tsz_null[1][:lmax+1]) + hp.almxfl(tlm_220,w_tsz_null[2][:lmax+1])
                elm = hp.almxfl(elm_95,w_Emv[0][:lmax+1]) + hp.almxfl(elm_150,w_Emv[1][:lmax+1]) + hp.almxfl(elm_220,w_Emv[2][:lmax+1])
                blm = hp.almxfl(blm_95,w_Bmv[0][:lmax+1]) + hp.almxfl(blm_150,w_Bmv[1][:lmax+1]) + hp.almxfl(blm_220,w_Bmv[2][:lmax+1])
                # Get Agora lensed CMB-only
                tlm_lcmb_95, elm_lcmb_95, blm_lcmb_95 = hp.read_alm(lcmb_095,hdu=[1,2,3])
                tlm_lcmb_150, elm_lcmb_150, blm_lcmb_150 = hp.read_alm(lcmb_150,hdu=[1,2,3])
                tlm_lcmb_220, elm_lcmb_220, blm_lcmb_220 = hp.read_alm(lcmb_220,hdu=[1,2,3])
                if append == 'standard':
                    tlm_lcmb = hp.almxfl(tlm_lcmb_95,w_Tmv[0][:lmax+1]) + hp.almxfl(tlm_lcmb_150,w_Tmv[1][:lmax+1]) + hp.almxfl(tlm_lcmb_220,w_Tmv[2][:lmax+1])
                elif append == 'mh':
                    tlm_lcmb = hp.almxfl(tlm_lcmb_95,w_tsz_null[0][:lmax+1]) + hp.almxfl(tlm_lcmb_150,w_tsz_null[1][:lmax+1]) + hp.almxfl(tlm_lcmb_220,w_tsz_null[2][:lmax+1])
                elm_lcmb = hp.almxfl(elm_lcmb_95,w_Emv[0][:lmax+1]) + hp.almxfl(elm_lcmb_150,w_Emv[1][:lmax+1]) + hp.almxfl(elm_lcmb_220,w_Emv[2][:lmax+1])
                blm_lcmb = hp.almxfl(blm_lcmb_95,w_Bmv[0][:lmax+1]) + hp.almxfl(blm_lcmb_150,w_Bmv[1][:lmax+1]) + hp.almxfl(blm_lcmb_220,w_Bmv[2][:lmax+1])
                # NOTE: AGORA CMB
                sltt_agora = hp.alm2cl(tlm_lcmb,tlm_lcmb) + artificial_noise
                slee_agora = hp.alm2cl(elm_lcmb,elm_lcmb)
                slbb_agora = hp.alm2cl(blm_lcmb,blm_lcmb)
                slte_agora = hp.alm2cl(tlm_lcmb,elm_lcmb)
                tlm_fg_95 = tlm_95 - tlm_lcmb_95; elm_fg_95 = elm_95 - elm_lcmb_95; blm_fg_95 = blm_95 - blm_lcmb_95
                tlm_fg_150 = tlm_150 - tlm_lcmb_150; elm_fg_150 = elm_150 - elm_lcmb_150; blm_fg_150 = blm_150 - blm_lcmb_150
                tlm_fg_220 = tlm_220 - tlm_lcmb_220; elm_fg_220 = elm_220 - elm_lcmb_220; blm_fg_220 = blm_220 - blm_lcmb_220
                if append == 'standard':
                    tlm_fg = hp.almxfl(tlm_fg_95,w_Tmv[0][:lmax+1]) + hp.almxfl(tlm_fg_150,w_Tmv[1][:lmax+1]) + hp.almxfl(tlm_fg_220,w_Tmv[2][:lmax+1])
                elif append == 'mh':
                    tlm_fg = hp.almxfl(tlm_fg_95,w_tsz_null[0][:lmax+1]) + hp.almxfl(tlm_fg_150,w_tsz_null[1][:lmax+1]) + hp.almxfl(tlm_fg_220,w_tsz_null[2][:lmax+1])
                elm_fg = hp.almxfl(elm_fg_95,w_Emv[0][:lmax+1]) + hp.almxfl(elm_fg_150,w_Emv[1][:lmax+1]) + hp.almxfl(elm_fg_220,w_Emv[2][:lmax+1])
                blm_fg = hp.almxfl(blm_fg_95,w_Bmv[0][:lmax+1]) + hp.almxfl(blm_fg_150,w_Bmv[1][:lmax+1]) + hp.almxfl(blm_fg_220,w_Bmv[2][:lmax+1])
                # NOTE: AGORA FG ONLY
                fltt_agora = hp.alm2cl(tlm_fg,tlm_fg) + artificial_noise
                flee_agora = hp.alm2cl(elm_fg,elm_fg)
                flbb_agora = hp.alm2cl(blm_fg,blm_fg)
                flte_agora = hp.alm2cl(tlm_fg,elm_fg)
                # Adding noise too to make it comparable with total Gaussian spectra
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
                # NOTE: AGORA NOISE ONLY
                nltt_agora = hp.alm2cl(tlm_n,tlm_n) + artificial_noise
                nlee_agora = hp.alm2cl(elm_n,elm_n)
                nlbb_agora = hp.alm2cl(blm_n,blm_n)
                nlte_agora = hp.alm2cl(tlm_n,elm_n)
                tlm_150 += nlmt_150; tlm_220 += nlmt_220; tlm_95 += nlmt_090
                elm_150 += nlme_150; elm_220 += nlme_220; elm_95 += nlme_090
                blm_150 += nlmb_150; blm_220 += nlmb_220; blm_95 += nlmb_090
                if append == 'standard':
                    tlm = hp.almxfl(tlm_95,w_Tmv[0][:lmax+1]) + hp.almxfl(tlm_150,w_Tmv[1][:lmax+1]) + hp.almxfl(tlm_220,w_Tmv[2][:lmax+1])
                elif append == 'mh':
                    tlm = hp.almxfl(tlm_95,w_tsz_null[0][:lmax+1]) + hp.almxfl(tlm_150,w_tsz_null[1][:lmax+1]) + hp.almxfl(tlm_220,w_tsz_null[2][:lmax+1])
                elm = hp.almxfl(elm_95,w_Emv[0][:lmax+1]) + hp.almxfl(elm_150,w_Emv[1][:lmax+1]) + hp.almxfl(elm_220,w_Emv[2][:lmax+1])
                blm = hp.almxfl(blm_95,w_Bmv[0][:lmax+1]) + hp.almxfl(blm_150,w_Bmv[1][:lmax+1]) + hp.almxfl(blm_220,w_Bmv[2][:lmax+1])
                # NOTE: AGORA TOTAL
                cltt_agora_tot = hp.alm2cl(tlm,tlm) + artificial_noise
                clee_agora_tot = hp.alm2cl(elm,elm)
                clbb_agora_tot = hp.alm2cl(blm,blm)
                clte_agora_tot = hp.alm2cl(tlm,elm)
                # Get Agora foreground + noise only
                tlm_fg_95 = tlm_95 - tlm_lcmb_95; elm_fg_95 = elm_95 - elm_lcmb_95; blm_fg_95 = blm_95 - blm_lcmb_95
                tlm_fg_150 = tlm_150 - tlm_lcmb_150; elm_fg_150 = elm_150 - elm_lcmb_150; blm_fg_150 = blm_150 - blm_lcmb_150
                tlm_fg_220 = tlm_220 - tlm_lcmb_220; elm_fg_220 = elm_220 - elm_lcmb_220; blm_fg_220 = blm_220 - blm_lcmb_220
                if append == 'standard':
                    tlm = hp.almxfl(tlm_fg_95,w_Tmv[0][:lmax+1]) + hp.almxfl(tlm_fg_150,w_Tmv[1][:lmax+1]) + hp.almxfl(tlm_fg_220,w_Tmv[2][:lmax+1])
                elif append == 'mh':
                    tlm = hp.almxfl(tlm_fg_95,w_tsz_null[0][:lmax+1]) + hp.almxfl(tlm_fg_150,w_tsz_null[1][:lmax+1]) + hp.almxfl(tlm_fg_220,w_tsz_null[2][:lmax+1])
                elm = hp.almxfl(elm_fg_95,w_Emv[0][:lmax+1]) + hp.almxfl(elm_fg_150,w_Emv[1][:lmax+1]) + hp.almxfl(elm_fg_220,w_Emv[2][:lmax+1])
                blm = hp.almxfl(blm_fg_95,w_Bmv[0][:lmax+1]) + hp.almxfl(blm_fg_150,w_Bmv[1][:lmax+1]) + hp.almxfl(blm_fg_220,w_Bmv[2][:lmax+1])
                # NOTE: AGORA FG + NOISE
                fnltt_agora = hp.alm2cl(tlm,tlm)
                fnlee_agora = hp.alm2cl(elm,elm)
                fnlbb_agora = hp.alm2cl(blm,blm)
                fnlte_agora = hp.alm2cl(tlm,elm)

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

                plt.figure(1)
                plt.clf()
                plt.plot(l, moving_average(clte_agora_tot/(totalcls_avg[:,3]),window_size=15), color='palegoldenrod', alpha=0.8, linestyle='--', label='clte ratio, Agora/Gaussian')
                plt.plot(l, moving_average(clbb_agora_tot/(totalcls_avg[:,2]),window_size=15), color='powderblue', alpha=0.8, linestyle='--', label='clbb ratio, Agora/Gaussian')
                plt.plot(l, moving_average(clee_agora_tot/(totalcls_avg[:,1]),window_size=15), color='lightgreen', alpha=0.8, linestyle='--', label='clee ratio, Agora/Gaussian')
                if append == 'standard':
                    plt.plot(l, moving_average(cltt_agora_tot/(totalcls_avg[:,0]),window_size=15), color='darksalmon', alpha=0.8, linestyle='--', label='cltt ratio, Agora/Gaussian')
                else:
                    plt.plot(l, moving_average(cltt_agora_tot/(totalcls_avg[:,5]),window_size=15), color='darksalmon', alpha=0.8, linestyle='--', label='cltt2 ratio, Agora/Gaussian')
                plt.axhline(y=1, color='k', linestyle='--')
                plt.xlabel('$\ell$')
                plt.title(f'Total Spectra Comparison, Agora/Gaussian')
                plt.legend(fontsize='x-small')
                plt.xscale('log')
                plt.xlim(300,lmax)
                plt.ylim(0.95,1.05)
                plt.savefig(dir_out+f'/figs/agora_spectra_ratio_comparison_vs_agora_{append}.png',bbox_inches='tight')

                plt.figure(1)
                plt.clf()
                plt.axhline(y=1, color='k', linestyle='--')
                plt.plot(l, moving_average(slte_agora/(slte),window_size=15), color='palegoldenrod', alpha=0.8, linestyle='--', label='slte ratio, Agora/Gaussian')
                plt.plot(l, moving_average(slbb_agora/(slbb),window_size=15), color='powderblue', alpha=0.8, linestyle='--', label='slbb ratio, Agora/Gaussian')
                plt.plot(l, moving_average(slee_agora/(slee),window_size=15), color='lightgreen', alpha=0.8, linestyle='--', label='slee ratio, Agora/Gaussian')
                if append == 'standard':
                    plt.plot(l, moving_average(sltt_agora/(sltt),window_size=15), color='darksalmon', alpha=0.8, linestyle='--', label='sltt ratio, Agora/Gaussian')
                else:
                    plt.plot(l, moving_average(sltt_agora/(sltt),window_size=15), color='darksalmon', alpha=0.8, linestyle='--', label='sltt ratio, Agora/Gaussian')
                plt.xlabel('$\ell$')
                plt.title(f'Lensed CMB Spectra Comparison, Agora/Gaussian')
                plt.legend(fontsize='x-small')
                plt.xscale('log')
                plt.xlim(300,lmax)
                plt.ylim(0.95,1.05)
                plt.savefig(dir_out+f'/figs/agora_lcmb_spectra_ratio_comparison_vs_agora_{append}.png',bbox_inches='tight')

                plt.figure(1)
                plt.clf()
                plt.axhline(y=1, color='k', linestyle='--')
                plt.plot(l, moving_average(nlbb_agora/(ret[:,2]),window_size=15), color='powderblue', alpha=0.8, linestyle='--', label='nlbb ratio, Agora/Gaussian')
                plt.plot(l, moving_average(nlee_agora/(ret[:,1]),window_size=15), color='lightgreen', alpha=0.8, linestyle='--', label='nlee ratio, Agora/Gaussian')
                if append == 'standard':
                    plt.plot(l, moving_average(nltt_agora/(ret[:,0]),window_size=15), color='darksalmon', alpha=0.8, linestyle='--', label='nltt ratio, Agora/Gaussian')
                else:
                    plt.plot(l, moving_average(nltt_agora/(ret[:,0]),window_size=15), color='darksalmon', alpha=0.8, linestyle='--', label='nltt2 ratio, Agora/Gaussian')
                plt.xlabel('$\ell$')
                plt.title(f'Noise Spectra Comparison, Agora/Gaussian')
                plt.legend(fontsize='x-small')
                plt.xscale('log')
                plt.xlim(300,lmax)
                plt.ylim(0.95,1.05)
                plt.savefig(dir_out+f'/figs/agora_n_spectra_ratio_comparison_vs_agora_{append}.png',bbox_inches='tight')

                plt.figure(1)
                plt.clf()
                plt.axhline(y=1, color='k', linestyle='--')
                plt.plot(l, moving_average(flte_agora/(totalcls_avg[:,3]-slte),window_size=15), color='palegoldenrod', alpha=0.8, linestyle='--', label='flte ratio, Agora/Gaussian')
                plt.plot(l, moving_average(flbb_agora/(totalcls_avg[:,2]-slbb-ret[:,2]),window_size=15), color='powderblue', alpha=0.8, linestyle='--', label='flbb ratio, Agora/Gaussian')
                plt.plot(l, moving_average(flee_agora/(totalcls_avg[:,1]-slee-ret[:,1]),window_size=15), color='lightgreen', alpha=0.8, linestyle='--', label='flee ratio, Agora/Gaussian')
                if append == 'standard':
                    plt.plot(l, moving_average(fltt_agora/(totalcls_avg[:,0]-sltt-ret[:,0]),window_size=15), color='darksalmon', alpha=0.8, linestyle='--', label='fltt ratio, Agora/Gaussian')
                else:
                    plt.plot(l, moving_average(fltt_agora/(totalcls_avg[:,5]-sltt-ret[:,0]),window_size=15), color='darksalmon', alpha=0.8, linestyle='--', label='fltt2 ratio, Agora/Gaussian')
                plt.xlabel('$\ell$')
                plt.title(f'Foreground Spectra Comparison, Agora/Gaussian')
                plt.legend(fontsize='x-small')
                plt.xscale('log')
                plt.xlim(300,lmax)
                plt.ylim(0.95,1.05)
                plt.savefig(dir_out+f'/figs/agora_fg_spectra_ratio_comparison_vs_agora_{append}.png',bbox_inches='tight')

                plt.figure(1)
                plt.clf()
                plt.axhline(y=1, color='k', linestyle='--')
                plt.plot(l, moving_average(fnlte_agora/(totalcls_avg[:,3]-slte),window_size=15), color='palegoldenrod', alpha=0.8, linestyle='--', label='flte + nlte ratio, Agora/Gaussian')
                plt.plot(l, moving_average(fnlbb_agora/(totalcls_avg[:,2]-slbb),window_size=15), color='powderblue', alpha=0.8, linestyle='--', label='flbb + nlbb ratio, Agora/Gaussian')
                plt.plot(l, moving_average(fnlee_agora/(totalcls_avg[:,1]-slee),window_size=15), color='lightgreen', alpha=0.8, linestyle='--', label='flee + nlee ratio, Agora/Gaussian')
                if append == 'standard':
                    plt.plot(l, moving_average(fnltt_agora/(totalcls_avg[:,0]-sltt),window_size=15), color='darksalmon', alpha=0.8, linestyle='--', label='fltt + nltt ratio, Agora/Gaussian')
                else:
                    plt.plot(l, moving_average(fnltt_agora/(totalcls_avg[:,5]-sltt),window_size=15), color='darksalmon', alpha=0.8, linestyle='--', label='fltt2 + nltt ratio, Agora/Gaussian')
                plt.xlabel('$\ell$')
                plt.title(f'Foreground + Noise Spectra Comparison, Agora/Gaussian')
                plt.legend(fontsize='x-small')
                plt.xscale('log')
                plt.xlim(300,lmax)
                plt.ylim(0.95,1.05)
                plt.savefig(dir_out+f'/figs/agora_nfg_spectra_ratio_comparison_vs_agora_{append}.png',bbox_inches='tight')

                plt.figure(1)
                plt.clf()
                plt.axhline(y=1, color='k', linestyle='--')
                plt.plot(l, moving_average((totalcls_avg[:,3]-slte)/(clte_old_tot-slte)), color='orange', alpha=0.8, linestyle=':', label='(clte-slte) ratio, New/Old')
                plt.plot(l, moving_average((totalcls_avg[:,2]-slbb)/(clbb_old_tot-slbb)), color='cornflowerblue', alpha=0.8, linestyle=':', label='(clbb-slbb) ratio, New/Old')
                plt.plot(l, moving_average((totalcls_avg[:,1]-slee)/(clee_old_tot-slee)), color='darkseagreen', alpha=0.8, linestyle=':', label='(clee-slee) ratio, New/Old')
                if append == 'standard':
                    plt.plot(l, moving_average((totalcls_avg[:,0]-sltt)/(cltt_old_tot-sltt)), color='rosybrown', alpha=0.8, linestyle=':', label='(cltt-sltt) ratio, New/Old')
                else:
                    plt.plot(l, moving_average((totalcls_avg[:,5]-sltt)/(cltt2_old_tot-sltt)), color='rosybrown', alpha=0.8, linestyle=':', label='(cltt2-sltt) ratio, New/Old')
                plt.xlabel('$\ell$')
                plt.title(f'Foreground + Noise Spectra Comparison')
                plt.legend(fontsize='x-small')
                plt.xscale('log')
                plt.xlim(300,lmax)
                plt.ylim(0.5,1.5)
                plt.savefig(dir_out+f'/figs/agora_spectra_ratio_comparison_vs_old_gaussian_{append}.png',bbox_inches='tight')

