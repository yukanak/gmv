import sys, os
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import numpy as np
import matplotlib.pyplot as plt
import healqest_utils as utils
import healpy as hp

config_file = 'mh_yuka.yaml'
config = utils.parse_yaml(config_file)
lmax = config['lensrec']['Lmax']
lmaxT = config['lensrec']['lmaxT']
lmaxP = config['lensrec']['lmaxP']
lmin = config['lensrec']['lminT']
nside = config['lensrec']['nside']
dir_out = config['dir_out']
append = 'mh'
l = np.arange(0,lmax+1)

ell,sltt,slee,slbb,slte = utils.get_lensedcls('/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat' ,lmax=lmax)
n = 99

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
# totalcls: T3T3, EE, BB, T3E, T1T1, T2T2, T1T2, T1T3, T2T3, T1E, T2E
totalcls_avg = np.vstack((cltt3,clee,clbb,clte,cltt1,cltt2,clttx,clt1t3,clt2t3,clt1e,clt2e)).T
np.save(dir_out+f'totalcls/totalcls_average_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy',totalcls_avg)

#totalcls_avg = np.load(dir_out+f'totalcls/totalcls_average_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')

plt.figure(0)
plt.clf()
plt.plot(ell, sltt, color='firebrick', linestyle='-', label='sltt')
plt.plot(ell, slee, color='forestgreen', linestyle='-', label='slee')
plt.plot(ell, slbb, color='darkblue', linestyle='-', label='slbb')
plt.plot(ell, slte, color='gold', linestyle='-', label='slte')
plt.plot(ell, totalcls_avg[:,4], color='pink', linestyle='--', label='total TT1')
plt.plot(ell, totalcls_avg[:,5], color='darkorchid', linestyle='--', label='total TT2')
plt.plot(ell, totalcls_avg[:,6], color='darksalmon', linestyle='--', label='total T1T2')
plt.plot(ell, totalcls_avg[:,1], color='lightgreen', linestyle='--', label='total EE')
plt.plot(ell, totalcls_avg[:,2], color='powderblue', linestyle='--', label='total BB')
plt.plot(ell, totalcls_avg[:,3], color='palegoldenrod', linestyle='--', label='total TE')
plt.xscale('log')
plt.yscale('log')
plt.xlim(10,lmax)
plt.ylim(1e-9,1e2)
plt.legend(loc='center left', bbox_to_anchor=(1,0.5))
plt.title(f'average of sims 1 through {n}')
plt.ylabel("$C_\ell$")
plt.xlabel('$\ell$')
plt.savefig(dir_out+f'/figs/totalcls_vs_signal_mh_average.png',bbox_inches='tight')
