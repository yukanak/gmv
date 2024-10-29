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
import gmv_resp_eq43

def debug_resp():
    config_file='test_yuka.yaml'
    config = utils.parse_yaml(config_file)
    lmax = config['lensrec']['Lmax']
    lmaxT = config['lensrec']['lmaxT']
    lmaxP = config['lensrec']['lmaxP']
    lmin = config['lensrec']['lminT']
    dir_out = config['dir_out']
    nside = config['lensrec']['nside']
    cltype = config['lensrec']['cltype']
    cls = config['cls']
    sl = {ee:config['cls'][cltype][ee] for ee in config['cls'][cltype].keys()}
    ell = np.arange(lmax+1,dtype=np.float_)
    l = np.arange(0,lmax+1)
    append = 'standard'
    totalcls = np.load(dir_out+f'totalcls/totalcls_average_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
    cltt = totalcls[:,0]; clee = totalcls[:,1]; clbb = totalcls[:,2]; clte = totalcls[:,3]
    filename_eq43 = dir_out+f'/resp/an_resp_eq43_debug.npy'
    filename = dir_out+f'/resp/an_resp_debug.npy'

    resp_eq43 = gmv_resp_eq43.gmv_resp(config,cltype,totalcls,u=None,crossilc=False,save_path=filename_eq43)
    resp = gmv_resp.gmv_resp(config,cltype,totalcls,u=None,crossilc=False,save_path=filename)

    ll = 1000
    resp_eq43.A(ll) # 3.6094545146750925e+18; 2.4822905100941862e+17 with f_TE only
    resp.var_d(resp.A_1(ll),resp.A_2(ll)) # 4.3767623658509066e+18; 2.561013720354738e+17 with f_TE only

    ll = 100
    resp_eq43.A(ll) # 1429880012526626.0
    resp.var_d(resp.A_1(ll),resp.A_2(ll)) # 1443347104110679.5

def compare_resp():
    config_file='test_yuka.yaml'
    config = utils.parse_yaml(config_file)
    lmax = config['lensrec']['Lmax']
    lmin = config['lensrec']['lminT']
    dir_out = config['dir_out']
    l = np.arange(0,lmax+1)
    #filename_eq43 = dir_out+f'/resp/an_resp_eq43_debug_numericalinv.npy'
    #filename_eq43 = dir_out+f'/resp/an_resp_eq43_debug_nofTE.npy'
    #filename_gmv = dir_out+f'/resp/an_resp_debug_nofTE.npy'

    # GMV response
    resp_gmv_alt = get_analytic_response('all',config,eq43=True,filename=None)
    inv_resp_gmv_alt = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_alt[1:] = 1./(resp_gmv_alt)[1:]

    resp_gmv = get_analytic_response('all',config,eq43=False,filename=None)
    resp_gmv_TTEETE = get_analytic_response('TTEETE',config,eq43=False)
    resp_gmv_TBEB = get_analytic_response('TBEB',config,eq43=False)
    inv_resp_gmv = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv[1:] = 1./(resp_gmv)[1:]
    inv_resp_gmv_TTEETE = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TTEETE[1:] = 1./(resp_gmv_TTEETE)[1:]
    inv_resp_gmv_TBEB = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_TBEB[1:] = 1./(resp_gmv_TBEB)[1:]

    # GMV response with numerical inversion of Cl matrix
    #resp_gmv_alt_debug = get_analytic_response('all',config,eq43=True,filename=filename_eq43)
    #inv_resp_gmv_alt_debug = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_alt_debug[1:] = 1./(resp_gmv_alt_debug)[1:]

    # GMV response with no f_TE terms
    #resp_gmv_alt_debug = get_analytic_response('all',config,eq43=True,filename=filename_eq43)
    #inv_resp_gmv_alt_debug = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_alt_debug[1:] = 1./(resp_gmv_alt_debug)[1:]

    #resp_gmv_debug = get_analytic_response('all',config,eq43=False,filename=filename_gmv)
    #inv_resp_gmv_debug = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_debug[1:] = 1./(resp_gmv_debug)[1:]

    # Response from sims
    filename_sim_resp = dir_out+f'/resp/sim_resp_gmv_estall_lmaxT3000_lmaxP4096_lmin300_cltypelcmb_standard.npy'
    sim_resp = np.load(filename_sim_resp)
    inv_resp_gmv_sim = np.zeros(len(l),dtype=np.complex_); inv_resp_gmv_sim[1:] = 1./(sim_resp)[1:]

    # Theory spectrum
    clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
    ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
    clkk = slpp * (l*(l+1))**2/4

    plt.figure(0)
    plt.clf()
    plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')

    plt.plot(l, inv_resp_gmv * (l*(l+1))**2/4, color='darkblue', linestyle='-', label='$1/R$ (GMV)')
    plt.plot(l, inv_resp_gmv_alt * (l*(l+1))**2/4, color='cornflowerblue', linestyle='-', label='$1/R$ (GMV, using Eq. 43)')

    #plt.plot(l, inv_resp_gmv_debug * (l*(l+1))**2/4, color='darkblue', linestyle='--', label='$1/R$ (GMV, no f_TE terms)')
    #plt.plot(l, inv_resp_gmv_alt_debug * (l*(l+1))**2/4, color='forestgreen', linestyle='--', label='$1/R$ (GMV, using Eq. 43, numerical inversion of Cl)')

    plt.plot(l, inv_resp_gmv_sim * (l*(l+1))**2/4, color='lightsteelblue', linestyle=':', label='$1/R$ (GMV, sim response)')

    plt.ylabel("$C_\ell^{\kappa\kappa}$")
    plt.xlabel('$\ell$')
    plt.title('$1/R$')
    plt.legend(loc='center left', fontsize='small', bbox_to_anchor=(1,0.5))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,lmax)
    #plt.ylim(8e-9,1e-5)
    #plt.ylim(8e-9,1e-6)
    plt.savefig(dir_out+f'/figs/gmv_cinv_style_response_comparison.png',bbox_inches='tight')

def get_analytic_response(est, config, append='standard', eq43=True,
                          filename=None):
    '''
    For gmv, est should be 'TTEETE'/'TBEB'/'all'.
    Also, we are taking lmax values from the config file, so make sure those are right.
    Note we are also assuming noise files for 2019/2020 analysis, and just loading the saved totalcl file.
    '''
    print(f'Computing analytic response for est {est}')
    lmax = config['lensrec']['Lmax']
    lmaxT = config['lensrec']['lmaxT']
    lmaxP = config['lensrec']['lmaxP']
    lmin = config['lensrec']['lminT']
    nside = config['lensrec']['nside']
    cltype = config['lensrec']['cltype']
    cls = config['cls']
    sl = {ee:config['cls'][cltype][ee] for ee in config['cls'][cltype].keys()}
    ell = np.arange(lmax+1,dtype=np.float_)
    dir_out = config['dir_out']

    if filename is None:
        fn = ''
        if eq43:
            fn += '_eq43'
        if est=='all' or est=='TTEETE' or est=='TBEB':
            fn += '_gmv_estall'
        else:
            fn += f'_gmv_est{est}'
        fn += f'_lmaxT{lmaxT}_lmaxP{lmaxP}_lmin{lmin}_cltype{cltype}_{append}'
        filename = dir_out+f'/resp/an_resp{fn}.npy'

    if os.path.isfile(filename):
        print('Loading from existing file!')
        R = np.load(filename)
    else:
        # File doesn't exist!
        # Load total Cls
        totalcls = np.load(dir_out+f'totalcls/totalcls_average_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')

        # GMV response
        if eq43:
            gmv_r = gmv_resp_eq43.gmv_resp(config,cltype,totalcls,u=None,crossilc=False,save_path=filename)
        else:
            gmv_r = gmv_resp.gmv_resp(config,cltype,totalcls,u=None,crossilc=False,save_path=filename)
        if est == 'TTEETE' or est == 'TBEB' or est == 'all':
            gmv_r.calc_tvar()
        elif est == 'TTEETEprf':
            gmv_r.calc_tvar_PRF(cross=False)
        elif est == 'TTEETETTEETEprf':
            gmv_r.calc_tvar_PRF(cross=True)
        R = np.load(filename)

    if not eq43:
        # If GMV, save file has columns L, TTEETE, TBEB, all
        if est == 'TTEETE' or est == 'TTEETEprf' or est == 'TTEETETTEETEprf':
            R = R[:,1]
        elif est == 'TBEB':
            R = R[:,2]
        elif est == 'all':
            R = R[:,3]
    else:
        assert est == 'all'
        R = R[:,1]

    return R

####################

compare_resp()
