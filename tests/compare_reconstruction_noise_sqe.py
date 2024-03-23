import os,sys
import pickle
import camb
import healpy as hp
import numpy as np
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import gmv_resp
import healqest_utils as utils
import matplotlib.pyplot as plt
import weights
import wignerd
import resp

def plot(config_file='test_yuka.yaml'):
    '''
    Compare SQE reconstruction noise for T-only vs pol-only for different noise levels.
    '''
    config = utils.parse_yaml(config_file)
    lmax = config['lensrec']['Lmax']
    lmin = config['lensrec']['lminT']
    lmaxT = config['lensrec']['lmaxT']
    lmaxP = config['lensrec']['lmaxP']
    nside = config['lensrec']['nside']
    dir_out = config['dir_out']
    l = np.arange(0,lmax+1)
    ests = ['TT', 'EE', 'TE', 'ET', 'TB', 'BT', 'EB', 'BE']

    # Get 1/R for SPT-3G 2019-2020 noise levels
    resps_spt = np.zeros((len(l),len(ests)), dtype=np.complex_)
    inv_resps_spt = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
    for i, est in enumerate(ests):
        resps_spt[:,i] = get_sqe_analytic_response(est,config,append='spt20192020')
        inv_resps_spt[1:,i] = 1/(resps_spt)[1:,i]
    resp_spt = np.sum(resps_spt, axis=1)
    inv_resp_spt = np.zeros_like(l,dtype=np.complex_); inv_resp_spt[1:] = 1/(resp_spt)[1:]
    # Get pol-only reconstruction noise
    resp_spt_pol = resps_spt[:,1]+resps_spt[:,6]#+resps_spt[:,7]
    inv_resp_spt_pol = np.zeros_like(l,dtype=np.complex_); inv_resp_spt_pol[1:] = 1/(resp_spt_pol)[1:]

    # Get 1/R for ACT DR6 noise levels
    resps_act = np.zeros((len(l),len(ests)), dtype=np.complex_)
    inv_resps_act = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
    for i, est in enumerate(ests):
        resps_act[:,i] = get_sqe_analytic_response(est,config,append='actdr6')
        inv_resps_act[1:,i] = 1/(resps_act)[1:,i]
    resp_act = np.sum(resps_act, axis=1)
    inv_resp_act = np.zeros_like(l,dtype=np.complex_); inv_resp_act[1:] = 1/(resp_act)[1:]
    # Get pol-only reconstruction noise
    resp_act_pol = resps_act[:,1]+resps_act[:,6]#+resps_act[:,7]
    inv_resp_act_pol = np.zeros_like(l,dtype=np.complex_); inv_resp_act_pol[1:] = 1/(resp_act_pol)[1:]

    # Get 1/R for 5 uK-arcmin noise levels (SPT-3G approximate)
    resps_spt2 = np.zeros((len(l),len(ests)), dtype=np.complex_)
    inv_resps_spt2 = np.zeros((len(l),len(ests)) ,dtype=np.complex_)
    for i, est in enumerate(ests):
        resps_spt2[:,i] = get_sqe_analytic_response(est,config,append='spt5ukarcmin')
        inv_resps_spt2[1:,i] = 1/(resps_spt2)[1:,i]
    resp_spt2 = np.sum(resps_spt2, axis=1)
    inv_resp_spt2 = np.zeros_like(l,dtype=np.complex_); inv_resp_spt2[1:] = 1/(resp_spt2)[1:]
    # Get pol-only reconstruction noise
    resp_spt2_pol = resps_spt2[:,1]+resps_spt2[:,6]#+resps_spt2[:,7]
    inv_resp_spt2_pol = np.zeros_like(l,dtype=np.complex_); inv_resp_spt2_pol[1:] = 1/(resp_spt2_pol)[1:]

    # Theory spectrum
    clfile_path = '/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat'
    ell,sltt,slee,slbb,slte,slpp,sltp,slep = utils.get_unlensedcls(clfile_path,lmax)
    clkk = slpp * (l*(l+1))**2/4

    # Plot
    plt.figure(0)
    plt.clf()
    plt.plot(l, clkk, 'k', label='Theory $C_\ell^{\kappa\kappa}$')

    plt.plot(l, inv_resp_spt_pol * (l*(l+1))**2/4, color='navy', linestyle='--', label='SPT-3G Noise Levels (polarization-only)')
    plt.plot(l, inv_resps_spt[:,0] * (l*(l+1))**2/4, color='cornflowerblue', linestyle='--', label='SPT-3G Noise Levels (temperature-only)')
    #plt.plot(l, inv_resp_spt2_pol * (l*(l+1))**2/4, color='rebeccapurple', linestyle='--', label='Approximate SPT-3G Noise Levels (5 $\mu K$-arcmin in T/P) (polarization-only)')
    #plt.plot(l, inv_resps_spt2[:,0] * (l*(l+1))**2/4, color='plum', linestyle='--', label='Approximate SPT-3G Noise Levels (5 $\mu K$-arcmin in T/P) (temperature-only)')
    plt.plot(l, inv_resp_act_pol * (l*(l+1))**2/4, color='firebrick', linestyle='--', label='Approximate ACT DR6 Noise Levels (15 $\mu K$-arcmin in T) (polarization-only)')
    plt.plot(l, inv_resps_act[:,0] * (l*(l+1))**2/4, color='lightcoral', linestyle='--', label='Approximate ACT DR6 Noise Levels (15 $\mu K$-arcmin in T) (temperature-only)')

    plt.ylabel("$[\ell(\ell+1)]^2$$N_0$ / 4 $[\mu K^2]$")
    plt.xlabel('$\ell$')
    plt.title(f'SQE Reconstruction Noise Comparison')
    plt.legend(loc='upper left', fontsize='x-small')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(10,lmax)
    plt.ylim(1e-8,1e-5)
    plt.savefig(dir_out+f'/figs/sqe_reconstruction_noise_comparison.png',bbox_inches='tight')

def get_sqe_analytic_response(est, config, append):
    '''
    Argument est should be 'TT'/'EE'/'TE'/'TB'/'EB'.
    Also, we are taking lmax values from the config file, so make sure those are right.
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

    filename = dir_out+f'/resp/an_resp_sqe_est{est}_lmaxT{lmaxT}_lmaxP{lmaxP}_lmin{lmin}_cltype{cltype}_{append}.npy'

    if os.path.isfile(filename):
        print('Loading from existing file!')
        R = np.load(filename)
    else:
        # File doesn't exist!
        artificial_noise = np.zeros(lmax+1)
        artificial_noise[lmaxT+2:] = 1.e10
        if append == 'spt20192020':
            noise_file = 'noise_curves/nl_cmbmv_20192020.dat'
            fsky_corr = 25.308939726920805
            noise_curves = np.loadtxt(noise_file)
            nltt = fsky_corr * noise_curves[:,1]; nlee = fsky_corr * noise_curves[:,2]; nlbb = fsky_corr * noise_curves[:,2]
        elif append == 'actdr6':
            # Noise levels from https://arxiv.org/pdf/2303.04180.pdf
            # Value for nlev_p is approximately true (it's exactly true in the case of using the same data for polarization as for temperature)
            fwhm=0; nlev_t=15; nlev_p=np.sqrt(2)*nlev_t
            bl = hp.gauss_beam(fwhm=fwhm*0.00029088,lmax=lmax)
            nltt = (np.pi/180./60.*nlev_t)**2 / bl**2
            nlee=nlbb = (np.pi/180./60.*nlev_p)**2 / bl**2
        elif append == 'spt5ukarcmin':
            # Kimmy says: iirc, when the lmax = 4000 for T, then the TT and EB N0 are about the same for the SPT-3G 2019+2020 noise levels
            fwhm=0; nlev_t=5; nlev_p=5
            bl = hp.gauss_beam(fwhm=fwhm*0.00029088,lmax=lmax)
            nltt = (np.pi/180./60.*nlev_t)**2 / bl**2
            nlee=nlbb = (np.pi/180./60.*nlev_p)**2 / bl**2
        cltt = sl['tt'][:lmax+1] + nltt[:lmax+1] + artificial_noise
        clee = sl['ee'][:lmax+1] + nlee[:lmax+1]
        clbb = sl['bb'][:lmax+1] + nlbb[:lmax+1]
        clte = sl['te'][:lmax+1]

        # Create 1/Nl filters
        flt = np.zeros(lmax+1); flt[lmin:] = 1./cltt[lmin:]
        fle = np.zeros(lmax+1); fle[lmin:] = 1./clee[lmin:]
        flb = np.zeros(lmax+1); flb[lmin:] = 1./clbb[lmin:]

        if est[0] == 'T': flX = flt
        if est[0] == 'E': flX = fle
        if est[0] == 'B': flX = flb

        if est[1] == 'T': flY = flt
        if est[1] == 'E': flY = fle
        if est[1] == 'B': flY = flb

        qeXY = weights.weights(est,cls[cltype],lmax,u=None)
        qeZA = None
        R = resp.fill_resp(qeXY,np.zeros(lmax+1, dtype=np.complex_),flX,flY,qeZA=qeZA)
        np.save(filename, R)

    return R

plot()
