from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pickle
import healpy as hp

l = np.arange(0,4096+1)
#ell, sltt, slee, slbb, slte, slpp, sltp, slep = np.loadtxt('planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat', unpack=True)
ell, sltt, slee, slbb, slte, slpp, sltp, slep = np.loadtxt('/home/users/yukanaka/healqest/healqest/camb/planck2018_base_plikHM_TTTEEE_lowl_lowE_lensing_lenspotentialCls.dat', unpack=True)
slpp = slpp / ell / ell / (ell + 1) / (ell + 1) * 2 * np.pi
slpp = np.insert(slpp, 0, 0)
slpp = np.insert(slpp, 0, 0)
clkk = slpp[:4097] * (l*(l+1))**2/4
lmax = 4096

# Results from healqest
healqest = np.load('/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/outputs/lensrec/initial_test/gmvjtp/clkk_polspice_nopsresp_nops/cls_kgmv_nsims1_98_nsims2_48_mcresp_nopsresp.npz',allow_pickle=True)
healqest_ratio = np.load('/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/outputs/lensrec/initial_test/gmvjtp/clkk_polspice_nopsresp_nops/ratio_cls_kgmv_nsims1_98_nsims2_48_mcresp_nopsresp.npz',allow_pickle=True)
healqest_TTEETE = np.load('/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/outputs/lensrec/initial_test/gmvjtp/clkk_polspice_nopsresp_nops/cls_kgmvtteete_nsims1_98_nsims2_48_mcresp_nopsresp.npz',allow_pickle=True)
healqest_TTEETE_ratio = np.load('/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/outputs/lensrec/initial_test/gmvjtp/clkk_polspice_nopsresp_nops/ratio_cls_kgmvtteete_nsims1_98_nsims2_48_mcresp_nopsresp.npz',allow_pickle=True)
healqest_TBEB = np.load('/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/outputs/lensrec/initial_test/gmvjtp/clkk_polspice_nopsresp_nops/cls_kgmvtbeb_nsims1_98_nsims2_48_mcresp_nopsresp.npz',allow_pickle=True)
healqest_TBEB_ratio = np.load('/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/outputs/lensrec/initial_test/gmvjtp/clkk_polspice_nopsresp_nops/ratio_cls_kgmvtbeb_nsims1_98_nsims2_48_mcresp_nopsresp.npz',allow_pickle=True)

# Results from my pipeline
n0_yuka = pickle.load(open('/oak/stanford/orgs/kipac/users/yukanaka/outputs/n0/n0_249simpairs_healqest_gmv_cinv_lmaxT3500_lmaxP4096_nside2048_standard_resp_from_sims.pkl','rb'))
n0_yuka_total = n0_yuka['total'] * (l*(l+1))**2/4
n0_yuka_TTEETE = n0_yuka['TTEETE'] * (l*(l+1))**2/4
n0_yuka_TBEB = n0_yuka['TBEB'] * (l*(l+1))**2/4
n0_yuka_TT = n0_yuka['TT'] * (l*(l+1))**2/4
n0_yuka_TE = n0_yuka['TE'] * (l*(l+1))**2/4
n0_yuka_EE = n0_yuka['EE'] * (l*(l+1))**2/4
n0_yuka_EB = n0_yuka['EB'] * (l*(l+1))**2/4
n0_yuka_TB = n0_yuka['TB'] * (l*(l+1))**2/4

# Plot
plt.figure(0)
plt.clf()
plt.plot(l, clkk, 'k', label='Fiducial $C_L^{\kappa\kappa}$')
plt.plot(l[:4001], healqest['N0'], color='firebrick', linestyle='-', alpha=0.8, label='N0 from healqest')
plt.plot(l[:4001], healqest_TTEETE['N0'], color='forestgreen', linestyle='-', alpha=0.8, label='N0 from healqest, TTEETE')
plt.plot(l[:4001], healqest_TBEB['N0'], color='darkblue', linestyle='-', alpha=0.8, label='N0 from healqest, TBEB')
plt.plot(l, n0_yuka_total, color='lightcoral', linestyle='--', alpha=0.5, label="N0 from Yuka's Pipeline")
plt.plot(l, n0_yuka_TTEETE, color='lightgreen', linestyle='--', alpha=0.5, label="N0 from Yuka's Pipeline, TTEETE")
plt.plot(l, n0_yuka_TBEB, color='cornflowerblue', linestyle='--', alpha=0.5, label="N0 from Yuka's Pipeline, TBEB")
plt.grid(True, linestyle="--", alpha=0.5)
plt.ylabel("$C_L^{\kappa\kappa}$")
plt.xlabel('$L$')
plt.title(f"Reconstruction Comparison",pad=10)
plt.legend(loc='lower right', fontsize='x-small')
plt.xscale('log')
plt.yscale('log')
plt.xlim(10,lmax)
plt.ylim(1e-9,1e-6)
plt.tight_layout()
plt.savefig(f'/oak/stanford/orgs/kipac/users/yukanaka/outputs/figs/healqest_pipeline_reconstruction_result_lmaxT3500_compare.png',bbox_inches='tight')

bb = np.round(np.geomspace(24, 3000, 18 + 1))
bin_centers_healqest = (bb[:-1] + bb[1:]) / 2
digitized = np.digitize(l, bb)
r_N0 = healqest['N0']/n0_yuka_total[:4001]
r_N0_TTEETE = healqest_TTEETE['N0']/n0_yuka_TTEETE[:4001]
r_N0_TBEB = healqest_TBEB['N0']/n0_yuka_TBEB[:4001]
plt.clf()
plt.axhline(y=1, color='k', linestyle='--')
plt.plot(l[:4001], r_N0, color='firebrick', linestyle='-', alpha=0.8, label='N0')
plt.plot(l[:4001], r_N0_TTEETE, color='forestgreen', linestyle='-', alpha=0.8, label='N0, TTEETE')
plt.plot(l[:4001], r_N0_TBEB, color='darkblue', linestyle='-', alpha=0.8, label='N0, TBEB')
plt.grid(True, linestyle="--", alpha=0.5)
plt.xlabel('$L$')
plt.title(f"Ratio healqest Pipeline / Yuka's Pipeline",pad=10)
plt.legend(loc='upper left', fontsize='small')
plt.xscale('log')
plt.ylim(0.9,1.7)
plt.xlim(10,lmax)
plt.tight_layout()
plt.savefig(f'/oak/stanford/orgs/kipac/users/yukanaka/outputs/figs/healqest_pipeline_reconstruction_result_lmaxT3500_compare_ratio.png',bbox_inches='tight')

#=============================================================================#

# Compute the SNR
# \Sum_L Clkk_L^theory / sqrt( Var( Clkk_L^measured ))
# You can estimate Var(Clkk) analytically by: Var(Clkk^measured) = ( 2/((2L+1)*fsky) ) * (Clkk^theory + N0)^2
# Or just take the variance from the sims (when taking variance from the sims, the full-sky one will have much smaller variance, hence correcting for the ~ 3.5%)
fsky = 0.035
var_healqest = ( 2/((2*l[:4001]+1)*fsky) ) * (clkk[:4001] + healqest['N0'])**2
snr_healqest = np.sum(clkk[24:3001]/np.sqrt(var_healqest[24:3001]))
var_yuka = ( 2/((2*l+1)*fsky) ) * (clkk + n0_yuka_total)**2
snr_yuka = np.sum(clkk[24:3001]/np.sqrt(var_yuka[24:3001]))
print(snr_healqest) # 2228
print(snr_yuka) # 2596
# Now, take the variance from sims
var_yuka = np.load('/oak/stanford/orgs/kipac/users/yukanaka/outputs/agora_reconstruction/measurement_uncertainty_lmaxT3500_standard_cinv.npy')
var_yuka = var_yuka**2
snr_yuka = np.sum(clkk[24:3001]/np.sqrt(var_yuka[24:3001])) * fsky
# For healqest... Need to get standard deviation
x = np.zeros((98,4001),dtype=np.complex_)
spec = 'kk'
qe = 'gmv'
for i in tqdm(range(1, 98 + 1)):
    f = '/oak/stanford/orgs/kipac/users/yukanaka/lensing19-20/outputs/lensrec/initial_test/gmvjtp/clkk_polspice_nopsresp_nops/' + f"cl{spec}_k{qe}_{i}a_{i}a_{i}a_{i}a.npz"
    x[i-1,:] = np.load(f)["cls"][: 4000 + 1, 1] - healqest['N0'][: 4000 + 1] - healqest['N1'][: 4000 + 1]
var_healqest = np.std(x,axis=0)**2
snr_healqest = np.sum(clkk[24:3001]/np.sqrt(var_healqest[24:3001]))
print(snr_healqest) # 7554
print(snr_yuka) # 474
