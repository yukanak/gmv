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
append = f'mh'
l = np.arange(0,lmax+1)

ell,sltt,slee,slbb,slte = utils.get_lensedcls('/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lensedCls.dat' ,lmax=lmax)
n = 40

cltt1 = np.zeros(lmax+1)
cltt2 = np.zeros(lmax+1)
clee = np.zeros(lmax+1)
clbb = np.zeros(lmax+1)
clte = np.zeros(lmax+1)
for i in np.arange(n)+1:
    totalcls = np.load(dir_out+f'totalcls/totalcls_seed1_{i}_seed2_{i}_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy')
    cltt1 += totalcls[:,0]
    cltt2 += totalcls[:,1]
    clee += totalcls[:,2]
    clbb += totalcls[:,3]
    clte += totalcls[:,4]
cltt1 /= 40
cltt2 /= 40
clee /= 40
clbb /= 40
clte /= 40
totalcls_avg = np.vstack((cltt1,cltt2,clee,clbb,clte)).T
np.save(dir_out+f'totalcls/totalcls_average_lmaxT{lmaxT}_lmaxP{lmaxP}_nside{nside}_{append}.npy',totalcls_avg)

plt.figure(0)
plt.clf()
plt.plot(ell, sltt, color='firebrick', linestyle='-', label='sltt')
plt.plot(ell, slee, color='forestgreen', linestyle='-', label='slee')
plt.plot(ell, slbb, color='darkblue', linestyle='-', label='slbb')
plt.plot(ell, slte, color='gold', linestyle='-', label='slte')
plt.plot(ell, totalcls_avg[:,0], color='pink', linestyle='--', label='total TT1')
plt.plot(ell, totalcls_avg[:,1], color='darkorchid', linestyle='--', label='total TT2')
plt.plot(ell, totalcls_avg[:,2], color='lightgreen', linestyle='--', label='total EE')
plt.plot(ell, totalcls_avg[:,3], color='powderblue', linestyle='--', label='total BB')
plt.plot(ell, totalcls_avg[:,4], color='palegoldenrod', linestyle='--', label='total TE')
plt.xscale('log')
plt.yscale('log')
plt.xlim(10,lmax)
plt.ylim(1e-9,1e2)
plt.legend(loc='center left', bbox_to_anchor=(1,0.5))
plt.title('average of sims 1 through 40')
plt.ylabel("$C_\ell$")
plt.xlabel('$\ell$')
plt.savefig(dir_out+f'/figs/totalcls_vs_signal_mh_average.png',bbox_inches='tight')
