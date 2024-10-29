#!/usr/bin/env python3
import numpy as np
import pickle
import healpy as hp
import camb
import os, sys
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import gmv_resp
import healqest_utils as utils
import matplotlib.pyplot as plt
import weights
import wignerd
import resp

def get_n0(sims,qetype,config,append,cmbonly=False,withT3=True):
    '''
    Get N0 bias. qetype should be 'gmv' or 'gmv_cinv'.
    Returns dictionary containing keys for each estimator.
    '''
    lmax = config['lensrec']['Lmax']
    lmin = config['lensrec']['lminT']
    lmaxT = config['lensrec']['lmaxT']
    lmaxP = config['lensrec']['lmaxP']
    nside = config['lensrec']['nside']
    cltype = config['lensrec']['cltype']
    sl = {ee:config['cls'][cltype][ee] for ee in config['cls'][cltype].keys()}
    dir_out = config['dir_out']
    l = np.arange(0,lmax+1)
    num = len(sims)
    append_original = append
    if cmbonly:
        append += '_cmbonly'
    if not withT3:
        filename = dir_out+f'/n0/n0_{num}simpairs_healqest_{qetype}_noT3_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_resp_from_sims_9ests_inflatedT2.pkl'
    elif qetype == 'gmv':
        filename = dir_out+f'/n0/n0_{num}simpairs_healqest_{qetype}_withT3_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_resp_from_sims_9ests_fixedweights_inflatedT2.pkl'
    else:
        filename = dir_out+f'/n0/n0_{num}simpairs_healqest_{qetype}_withT3_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_resp_from_sims_9ests_inflatedT2.pkl'

    if os.path.isfile(filename):
        n0 = pickle.load(open(filename,'rb'))

    elif qetype == 'gmv':
        # GMV response
        resp_gmv = get_sim_response('all',config,cinv=False,append=append_original,sims=np.append(sims,num+1),withT3=withT3)
        resp_gmv_TTEETE = get_sim_response('TTEETE',config,cinv=False,append=append_original,sims=np.append(sims,num+1),withT3=withT3)
        resp_gmv_TBEB = get_sim_response('TBEB',config,cinv=False,append=append_original,sims=np.append(sims,num+1),withT3=withT3)
        inv_resp_gmv = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv[1:] = 1./(resp_gmv)[1:]
        inv_resp_gmv_TTEETE = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TTEETE[1:] = 1./(resp_gmv_TTEETE)[1:]
        inv_resp_gmv_TBEB = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TBEB[1:] = 1./(resp_gmv_TBEB)[1:]

        n0 = {'total':0, 'TTEETE':0, 'TBEB':0}
        for i, sim1 in enumerate(sims):
            sim2 = sim1 + 1

            if not withT3:
                # Get the lensed ij sims
                plm_gmv_ij = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_noT3_inflatedT2.npy')
                plm_gmv_TTEETE_ij = np.load(dir_out+f'/plm_TTEETE_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_noT3_inflatedT2.npy')
                plm_gmv_TBEB_ij = np.load(dir_out+f'/plm_TBEB_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_noT3_inflatedT2.npy')

                # Now get the ji sims
                plm_gmv_ji = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_noT3_inflatedT2.npy')
                plm_gmv_TTEETE_ji = np.load(dir_out+f'/plm_TTEETE_healqest_gmv_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_noT3_inflatedT2.npy')
                plm_gmv_TBEB_ji = np.load(dir_out+f'/plm_TBEB_healqest_gmv_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_noT3_inflatedT2.npy')
            else:
                # Get the lensed ij sims
                plm_gmv_ij = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_fixedweights_inflatedT2.npy')
                plm_gmv_TTEETE_ij = np.load(dir_out+f'/plm_TTEETE_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_fixedweights_inflatedT2.npy')
                plm_gmv_TBEB_ij = np.load(dir_out+f'/plm_TBEB_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_fixedweights_inflatedT2.npy')

                # Now get the ji sims
                plm_gmv_ji = np.load(dir_out+f'/plm_all_healqest_gmv_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_fixedweights_inflatedT2.npy')
                plm_gmv_TTEETE_ji = np.load(dir_out+f'/plm_TTEETE_healqest_gmv_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_fixedweights_inflatedT2.npy')
                plm_gmv_TBEB_ji = np.load(dir_out+f'/plm_TBEB_healqest_gmv_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_fixedweights_inflatedT2.npy')

            # Response correct
            plm_gmv_resp_corr_ij = hp.almxfl(plm_gmv_ij,inv_resp_gmv)
            plm_gmv_resp_corr_TTEETE_ij = hp.almxfl(plm_gmv_TTEETE_ij,inv_resp_gmv_TTEETE)
            plm_gmv_resp_corr_TBEB_ij = hp.almxfl(plm_gmv_TBEB_ij,inv_resp_gmv_TBEB)

            plm_gmv_resp_corr_ji = hp.almxfl(plm_gmv_ji,inv_resp_gmv)
            plm_gmv_resp_corr_TTEETE_ji = hp.almxfl(plm_gmv_TTEETE_ji,inv_resp_gmv_TTEETE)
            plm_gmv_resp_corr_TBEB_ji = hp.almxfl(plm_gmv_TBEB_ji,inv_resp_gmv_TBEB)

            # Get ij auto spectra <ijij>
            auto = hp.alm2cl(plm_gmv_resp_corr_ij, plm_gmv_resp_corr_ij, lmax=lmax)
            auto_A = hp.alm2cl(plm_gmv_resp_corr_TTEETE_ij, plm_gmv_resp_corr_TTEETE_ij, lmax=lmax)
            auto_B = hp.alm2cl(plm_gmv_resp_corr_TBEB_ij, plm_gmv_resp_corr_TBEB_ij, lmax=lmax)

            # Get cross spectra <ijji>
            cross = hp.alm2cl(plm_gmv_resp_corr_ij, plm_gmv_resp_corr_ji, lmax=lmax)
            cross_A = hp.alm2cl(plm_gmv_resp_corr_TTEETE_ij, plm_gmv_resp_corr_TTEETE_ji, lmax=lmax)
            cross_B = hp.alm2cl(plm_gmv_resp_corr_TBEB_ij, plm_gmv_resp_corr_TBEB_ji, lmax=lmax)

            n0['total'] += auto + cross
            n0['TTEETE'] += auto_A + cross_A
            n0['TBEB'] += auto_B + cross_B

        n0['total'] *= 1/num
        n0['TTEETE'] *= 1/num
        n0['TBEB'] *= 1/num

        with open(filename, 'wb') as f:
            pickle.dump(n0, f)

    elif qetype == 'gmv_cinv':
        # SQE response
        ests = ['T1T2', 'T2T1', 'EE', 'TE', 'ET', 'TB', 'BT', 'EB', 'BE']
        resps = np.zeros((len(l),len(ests)), dtype=np.complex_)
        inv_resps = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
        for i, est in enumerate(ests):
            resps[:,i] = get_sim_response(est,config,cinv=True,append=append_original,sims=np.append(sims,num+1),withT3=withT3)
            inv_resps[1:,i] = 1/(resps)[1:,i]
        resp = 0.5*np.sum(resps[:,:2], axis=1)+np.sum(resps[:,2:], axis=1)
        resp_TTEETE = 0.5*np.sum(resps[:,:2], axis=1)+np.sum(resps[:,2:5], axis=1)
        resp_TBEB = np.sum(resps[:,5:], axis=1)
        inv_resp = np.zeros_like(l,dtype=np.complex_); inv_resp[1:] = 1/(resp)[1:]
        inv_resp_TTEETE = np.zeros_like(l,dtype=np.complex_); inv_resp_TTEETE[1:] = 1/(resp_TTEETE)[1:]
        inv_resp_TBEB = np.zeros_like(l,dtype=np.complex_); inv_resp_TBEB[1:] = 1/(resp_TBEB)[1:]

        n0 = {'total':0, 'T1T2':0, 'T2T1':0, 'EE':0, 'TE':0, 'ET':0, 'TB':0, 'BT':0, 'EB':0, 'BE':0, 'TTEETE':0, 'TBEB':0}
        for i, sim1 in enumerate(sims):
            sim2 = sim1 + 1

            if not withT3:
                # Get the lensed ij sims
                plms_ij = np.zeros((len(np.load(dir_out+f'/plm_T1T2_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_noT3_inflatedT2.npy')),len(ests)), dtype=np.complex_)
                for i, est in enumerate(ests):
                    plms_ij[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_noT3_inflatedT2.npy')
                plm_total_ij = 0.5*np.sum(plms_ij[:,:2], axis=1)+np.sum(plms_ij[:,2:], axis=1)

                # Now get the ji sims
                plms_ji = np.zeros((len(np.load(dir_out+f'/plm_T1T2_healqest_gmv_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_noT3_inflatedT2.npy')),len(ests)), dtype=np.complex_)
                for i, est in enumerate(ests):
                    plms_ji[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_noT3_inflatedT2.npy')
                plm_total_ji = 0.5*np.sum(plms_ji[:,:2], axis=1)+np.sum(plms_ji[:,2:], axis=1)
            else:
                # Get the lensed ij sims
                plms_ij = np.zeros((len(np.load(dir_out+f'/plm_T1T2_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_inflatedT2.npy')),len(ests)), dtype=np.complex_)
                for i, est in enumerate(ests):
                    plms_ij[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim1}_seed2_{sim2}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_inflatedT2.npy')
                plm_total_ij = 0.5*np.sum(plms_ij[:,:2], axis=1)+np.sum(plms_ij[:,2:], axis=1)

                # Now get the ji sims
                plms_ji = np.zeros((len(np.load(dir_out+f'/plm_T1T2_healqest_gmv_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_inflatedT2.npy')),len(ests)), dtype=np.complex_)
                for i, est in enumerate(ests):
                    plms_ji[:,i] = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim2}_seed2_{sim1}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_inflatedT2.npy')
                plm_total_ji = 0.5*np.sum(plms_ji[:,:2], axis=1)+np.sum(plms_ji[:,2:], axis=1)

            # NINE estimators!!!
            plm_TTEETE_ij = 0.5*np.sum(plms_ij[:,:2], axis=1)+np.sum(plms_ij[:,2:5], axis=1)
            plm_TTEETE_ji = 0.5*np.sum(plms_ji[:,:2], axis=1)+np.sum(plms_ji[:,2:5], axis=1)
            plm_TBEB_ij = np.sum(plms_ij[:,5:], axis=1)
            plm_TBEB_ji = np.sum(plms_ji[:,5:], axis=1)

            # Response correct
            plm_total_ij = hp.almxfl(plm_total_ij,inv_resp)
            plm_TTEETE_ij = hp.almxfl(plm_TTEETE_ij,inv_resp_TTEETE)
            plm_TBEB_ij = hp.almxfl(plm_TBEB_ij,inv_resp_TBEB)
            for i, est in enumerate(ests):
                plms_ij[:,i] = hp.almxfl(plms_ij[:,i],inv_resps[:,i])

            plm_total_ji = hp.almxfl(plm_total_ji,inv_resp)
            plm_TTEETE_ji = hp.almxfl(plm_TTEETE_ji,inv_resp_TTEETE)
            plm_TBEB_ji = hp.almxfl(plm_TBEB_ji,inv_resp_TBEB)
            for i, est in enumerate(ests):
                plms_ji[:,i] = hp.almxfl(plms_ji[:,i],inv_resps[:,i])

            # Get ij auto spectra <ijij>
            auto = hp.alm2cl(plm_total_ij, plm_total_ij, lmax=lmax)
            auto_TTEETE = hp.alm2cl(plm_TTEETE_ij, plm_TTEETE_ij, lmax=lmax)
            auto_TBEB = hp.alm2cl(plm_TBEB_ij, plm_TBEB_ij, lmax=lmax)
            auto_T1T2 = hp.alm2cl(plms_ij[:,0], plms_ij[:,0], lmax=lmax)
            auto_T2T1 = hp.alm2cl(plms_ij[:,1], plms_ij[:,1], lmax=lmax)
            auto_EE = hp.alm2cl(plms_ij[:,2], plms_ij[:,2], lmax=lmax)
            auto_TE = hp.alm2cl(plms_ij[:,3], plms_ij[:,3], lmax=lmax)
            auto_ET = hp.alm2cl(plms_ij[:,4], plms_ij[:,4], lmax=lmax)
            auto_TB = hp.alm2cl(plms_ij[:,5], plms_ij[:,5], lmax=lmax)
            auto_BT = hp.alm2cl(plms_ij[:,6], plms_ij[:,6], lmax=lmax)
            auto_EB = hp.alm2cl(plms_ij[:,7], plms_ij[:,7], lmax=lmax)
            auto_BE = hp.alm2cl(plms_ij[:,8], plms_ij[:,8], lmax=lmax)

            # Get cross spectra <ijji>
            cross = hp.alm2cl(plm_total_ij, plm_total_ji, lmax=lmax)
            cross_TTEETE = hp.alm2cl(plm_TTEETE_ij, plm_TTEETE_ji, lmax=lmax)
            cross_TBEB = hp.alm2cl(plm_TBEB_ij, plm_TBEB_ji, lmax=lmax)
            cross_T1T2 = hp.alm2cl(plms_ij[:,0], plms_ji[:,0], lmax=lmax)
            cross_T2T1 = hp.alm2cl(plms_ij[:,1], plms_ji[:,1], lmax=lmax)
            cross_EE = hp.alm2cl(plms_ij[:,2], plms_ji[:,2], lmax=lmax)
            cross_TE = hp.alm2cl(plms_ij[:,3], plms_ji[:,3], lmax=lmax)
            cross_ET = hp.alm2cl(plms_ij[:,4], plms_ji[:,4], lmax=lmax)
            cross_TB = hp.alm2cl(plms_ij[:,5], plms_ji[:,5], lmax=lmax)
            cross_BT = hp.alm2cl(plms_ij[:,6], plms_ji[:,6], lmax=lmax)
            cross_EB = hp.alm2cl(plms_ij[:,7], plms_ji[:,7], lmax=lmax)
            cross_BE = hp.alm2cl(plms_ij[:,8], plms_ji[:,8], lmax=lmax)

            n0['total'] += auto + cross
            n0['TTEETE'] += auto_TTEETE + cross_TTEETE
            n0['TBEB'] += auto_TBEB + cross_TBEB
            n0['T1T2'] += auto_T1T2 + cross_T1T2
            n0['T2T1'] += auto_T2T1 + cross_T2T1
            n0['EE'] += auto_EE + cross_EE
            n0['TE'] += auto_TE + cross_TE
            n0['ET'] += auto_ET + cross_ET
            n0['TB'] += auto_TB + cross_TB
            n0['BT'] += auto_BT + cross_BT
            n0['EB'] += auto_EB + cross_EB
            n0['BE'] += auto_BE + cross_BE

        n0['total'] *= 1/num
        n0['TTEETE'] *= 1/num
        n0['TBEB'] *= 1/num
        n0['T1T2'] *= 1/num
        n0['T2T1'] *= 1/num
        n0['EE'] *= 1/num
        n0['TE'] *= 1/num
        n0['ET'] *= 1/num
        n0['TB'] *= 1/num
        n0['BT'] *= 1/num
        n0['EB'] *= 1/num
        n0['BE'] *= 1/num

        with open(filename, 'wb') as f:
            pickle.dump(n0, f)

    elif qetype == 'sqe':
        pass

    else:
        print('Invalid argument qetype.')

    return n0

def get_sim_response(est,config,cinv,append,sims,filename=None,withT3=True,gmv=True):
    '''
    Make sure the sims are lensed, not unlensed!
    '''
    lmax = config['lensrec']['Lmax']
    lmaxT = config['lensrec']['lmaxT']
    lmaxP = config['lensrec']['lmaxP']
    lmin = config['lensrec']['lminT']
    nside = config['lensrec']['nside']
    cltype = config['lensrec']['cltype']
    dir_out = config['dir_out']
    l = np.arange(0,lmax+1)
    num = len(sims)
    if filename is None:
        fn = ''
        if not gmv:
            fn += f'_sqe_est{est}'
        elif cinv:
            fn += f'_gmv_cinv_est{est}'
        else:
            fn += f'_gmv_est{est}'
        fn += f'_lmaxT{lmaxT}_lmaxP{lmaxP}_lmin{lmin}_cltype{cltype}_{append}'
        if not withT3:
            fn += '_noT3'
        else:
            fn += '_withT3'
            if gmv and not cinv:
                fn += '_fixedweights'
        filename = dir_out+f'/resp/sim_resp{fn}_inflatedT2.npy'

    if os.path.isfile(filename):
        print('Loading from existing file!')
        sim_resp = np.load(filename)
    else:
        # File doesn't exist!
        cross_uncorrected_all = 0
        auto_input_all = 0
        for ii, sim in enumerate(sims):
            # Load plm
            if not gmv:
                if not withT3:
                    pass
                else:
                    plm = np.load(dir_out+f'/plm_{est}_healqest_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_inflatedT2.npy')
            else:
                if not withT3:
                    if not cinv:
                        plm = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_noT3_inflatedT2.npy')
                    else:
                        plm = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_noT3_inflatedT2.npy')
                else:
                    if not cinv:
                        plm = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_fixedweights_inflatedT2.npy')
                    else:
                        plm = np.load(dir_out+f'/plm_{est}_healqest_gmv_seed1_{sim}_seed2_{sim}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}_cinv_inflatedT2.npy')
            input_plm = hp.read_alm(f'/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/inputcmb/phi/phi_lmax_{lmax}/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_cambphiG_phi1_seed{sim}_lmax{lmax}.alm')
            # Cross correlate with input plm
            # For response from sims, want to use plms that are not response corrected
            cross_uncorrected_all += hp.alm2cl(input_plm, plm, lmax=lmax)
            auto_input_all += hp.alm2cl(input_plm, input_plm, lmax=lmax)

        # Get "response from sims" calculated the same way as the MC response
        auto_input_avg = auto_input_all / num
        cross_uncorrected_avg = cross_uncorrected_all / num
        sim_resp = cross_uncorrected_avg / auto_input_avg
        np.save(filename, sim_resp)
    return sim_resp

####################

n0_n1_sims=np.arange(98)+1
config_file='test_yuka.yaml'
append='mh'
config = utils.parse_yaml(config_file)
withT3 = False
n0_cinv = get_n0(sims=n0_n1_sims,qetype='gmv_cinv',config=config,
                 append=append,withT3=withT3)
