import numpy as np
import healpy as hp
import sys, os
import matplotlib.pyplot as plt

dir_out = '/oak/stanford/orgs/kipac/users/yukanaka/outputs/'
lmax = 4096
l = np.arange(0,lmax+1)

# My noise curves
# In muK^2-radians^2
noise_curves_090_090 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_090_090.txt'))
noise_curves_150_150 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_150_150.txt'))
noise_curves_220_220 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_220_220.txt'))
noise_curves_090_150 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_090_150.txt'))
noise_curves_090_220 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_090_220.txt'))
noise_curves_150_220 = np.nan_to_num(np.loadtxt('noise_curves/nl_fromstack_150_220.txt'))
nltt_090_090 = noise_curves_090_090[:,1]; nlee_090_090 = noise_curves_090_090[:,2]; nlbb_090_090 = noise_curves_090_090[:,3]
nltt_150_150 = noise_curves_150_150[:,1]; nlee_150_150 = noise_curves_150_150[:,2]; nlbb_150_150 = noise_curves_150_150[:,3]
nltt_220_220 = noise_curves_220_220[:,1]; nlee_220_220 = noise_curves_220_220[:,2]; nlbb_220_220 = noise_curves_220_220[:,3]
nltt_090_150 = noise_curves_090_150[:,1]; nlee_090_150 = noise_curves_090_150[:,2]; nlbb_090_150 = noise_curves_090_150[:,3]
nltt_090_220 = noise_curves_090_220[:,1]; nlee_090_220 = noise_curves_090_220[:,2]; nlbb_090_220 = noise_curves_090_220[:,3]
nltt_150_220 = noise_curves_150_220[:,1]; nlee_150_220 = noise_curves_150_220[:,2]; nlbb_150_220 = noise_curves_150_220[:,3]

# Yuuki's ILC weights
# Dimension (3, 6001) for 90, 150, 220 GHz respectively
w_Tmv = np.loadtxt('ilc_weights/weights1d_TT_spt3g_cmbmv.dat')
w_Emv = np.loadtxt('ilc_weights/weights1d_EE_spt3g_cmbmv.dat')
w_Bmv = np.loadtxt('ilc_weights/weights1d_BB_spt3g_cmbmv.dat')

# My ILC weights
# Dimension (3, 6001) for 90, 150, 220 GHz respectively
w_Emv_yuka = np.loadtxt('ilc_weights/weights1d_EE_spt3g_cmbmv_yuka.dat')
w_Bmv_yuka = np.loadtxt('ilc_weights/weights1d_BB_spt3g_cmbmv_yuka.dat')

#-----------------------------------------------------------------------------#

# Ana manually adds SPT noise to the Websky maps to make ILC weights for me
# Need to check if our noise files match

# Ana's noise
white_noise_arcmin = np.array([5.36, 4.21, 15.72]) # in muK-arcmin
white_noise = white_noise_arcmin * np.pi/(60*180) # arcmin to rad
l_knee = np.array([1052, 2023, 1873])
alpha = np.array([-4.68, -4.11, -4.22])
beam = 1 #TODO
L = np.arange(0,lmax+1)
# T noise curves
#noise_eff = (np.square(white_noise)*(1+(L/l_knee)**alpha))/np.square(beam) # in muK^2-radians^2
noise_eff_90_90 = (np.square(white_noise[0])*(1+(L/l_knee[0])**alpha[0]))/np.square(beam)# * ((60*180)/np.pi)**2
noise_eff_150_150 = (np.square(white_noise[1])*(1+(L/l_knee[1])**alpha[1]))/np.square(beam)# * ((60*180)/np.pi)**2
noise_eff_220_220 = (np.square(white_noise[2])*(1+(L/l_knee[2])**alpha[2]))/np.square(beam)# * ((60*180)/np.pi)**2
# T noise is smaller by factor of 2 in power spec compared to P
nltt_090_090_ana = noise_eff_90_90; nlee_090_090_ana = nltt_090_090_ana * 2; nlbb_090_090_ana = nltt_090_090_ana * 2
nltt_150_150_ana = noise_eff_150_150; nlee_150_150_ana = nltt_150_150_ana * 2; nlbb_150_150_ana = nltt_150_150_ana * 2
nltt_220_220_ana = noise_eff_220_220; nlee_220_220_ana = nltt_220_220_ana * 2; nlbb_220_220_ana = nltt_220_220_ana * 2

# Plot to compare
plt.figure(0)
plt.clf()
plt.plot(l, nltt_090_090[:lmax+1], color='firebrick', linestyle='-', label='nltt 90 x 90 from file')
plt.plot(l, nltt_150_150[:lmax+1], color='sienna', linestyle='-', label='nltt 150 x 150 from file')
plt.plot(l, nltt_220_220[:lmax+1], color='orange', linestyle='-', label='nltt 220 x 220 from file')
plt.plot(l, nltt_090_090_ana[:lmax+1], color='pink', linestyle='--', label="Ana's nltt 95 x 95")
plt.plot(l, nltt_150_150_ana[:lmax+1], color='sandybrown', linestyle='--', label="Ana's nltt 150 x 150")
plt.plot(l, nltt_220_220_ana[:lmax+1], color='bisque', linestyle='--', label="Ana's nltt 220 x 220")
plt.xscale('log')
plt.yscale('log')
plt.xlim(50,lmax)
#plt.ylim(1e4,1e10)
#plt.ylim(1e-6,1e2)
plt.legend(loc='center left', bbox_to_anchor=(1,0.5))
plt.ylabel("$N_\ell$ [$\mu K^2$-$\mathrm{rad}^2$]")
plt.xlabel('$\ell$')
plt.savefig(dir_out+f'/figs/test.png',bbox_inches='tight')

#-----------------------------------------------------------------------------#

# Need MV-ILC weights for E and B

# https://github.com/SouthPoleTelescope/spt3g_software/blob/master/scratch/wlwu/weddingcake/prelim.py#L253
fg90x90    = np.zeros(lmax+1)
fg90x150   = np.zeros(lmax+1)
fg90x220   = np.zeros(lmax+1)
fg150x150  = np.zeros(lmax+1)
fg150x220  = np.zeros(lmax+1)
fg220x220  = np.zeros(lmax+1)

C_ell = np.zeros([lmax+1, 3, 3])
C_ell[:,0,0] = fg90x90   + nlee_090_090[:lmax+1]
C_ell[:,0,1] = fg90x150  + nlee_090_150[:lmax+1]
C_ell[:,0,2] = fg90x220  + nlee_090_220[:lmax+1]
C_ell[:,1,1] = fg150x150 + nlee_150_150[:lmax+1]
C_ell[:,1,2] = fg150x220 + nlee_150_220[:lmax+1]
C_ell[:,2,2] = fg220x220 + nlee_220_220[:lmax+1]
C_ell[:,1,0] = C_ell[:,0,1]
C_ell[:,2,0] = C_ell[:,0,2]
C_ell[:,1,2] = C_ell[:,2,1]

# https://github.com/SouthPoleTelescope/spt3g_software/blob/master/scratch/wlwu/weddingcake/prelim.py#L283
a = np.array([[1, 1, 1]]).T
# TODO: Using pinv instead of inv because numpy thinks C_ell is singular otherwise
# But it's not; np.linalg.det(C_ell) shows it's just really small but not exactly zero
#Cinv = np.linalg.inv(C_ell)
Cinv = np.linalg.pinv(C_ell)
atR  = a.T@Cinv
#atRa_inv = np.linalg.inv(atR @ a)
atRa_inv = np.linalg.pinv(atR @ a)
w_Emv_new = atRa_inv @ atR # dims = [lmax+1, 1, N_freq]

C_ell = np.zeros([lmax+1, 3, 3])
C_ell[:,0,0] = fg90x90   + nlbb_090_090[:lmax+1]
C_ell[:,0,1] = fg90x150  + nlbb_090_150[:lmax+1]
C_ell[:,0,2] = fg90x220  + nlbb_090_220[:lmax+1]
C_ell[:,1,1] = fg150x150 + nlbb_150_150[:lmax+1]
C_ell[:,1,2] = fg150x220 + nlbb_150_220[:lmax+1]
C_ell[:,2,2] = fg220x220 + nlbb_220_220[:lmax+1]
C_ell[:,1,0] = C_ell[:,0,1]
C_ell[:,2,0] = C_ell[:,0,2]
C_ell[:,1,2] = C_ell[:,2,1]

# https://github.com/SouthPoleTelescope/spt3g_software/blob/master/scratch/wlwu/weddingcake/prelim.py#L283
a = np.array([[1, 1, 1]]).T
#Cinv = np.linalg.inv(C_ell)
Cinv = np.linalg.pinv(C_ell)
atR  = a.T@Cinv
#atRa_inv = np.linalg.inv(atR @ a)
atRa_inv = np.linalg.pinv(atR @ a)
w_Bmv_new = atRa_inv @ atR # dims = [lmax+1, 1, N_freq]

# Plot to compare
plt.figure(1)
plt.clf()
plt.plot(l, w_Emv[0][:lmax+1], color='forestgreen', linestyle='-', label='E 90 GHz MV-ILC weights from Yuuki')
plt.plot(l, w_Emv[1][:lmax+1], color='darkblue', linestyle='-', label='E 150 GHz MV-ILC weights from Yuuki')
plt.plot(l, w_Emv[2][:lmax+1], color='violet', linestyle='-', label='E 220 GHz MV-ILC weights from Yuuki')
plt.plot(l, w_Emv_yuka[0][:lmax+1], color='lightgreen', linestyle='--', label="E 90 GHz MV-ILC weights from inverse noise")
plt.plot(l, w_Emv_yuka[1][:lmax+1], color='cornflowerblue', linestyle='--', label="E 150 GHz MV-ILC weights from inverse noise")
plt.plot(l, w_Emv_yuka[2][:lmax+1], color='pink', linestyle='--', label="E 220 GHz MV-ILC weights from inverse noise")
#plt.plot(l, w_Emv_new[:lmax+1,0,0], color='lightgreen', linestyle='--', label="E 90 GHz MV-ILC weights from inverse noise")
#plt.plot(l, w_Emv_new[:lmax+1,0,1], color='cornflowerblue', linestyle='--', label="E 150 GHz MV-ILC weights from inverse noise")
#plt.plot(l, w_Emv_new[:lmax+1,0,2], color='pink', linestyle='--', label="E 220 GHz MV-ILC weights from inverse noise")
plt.xscale('log')
plt.yscale('log')
plt.xlim(200,lmax)
#plt.ylim(1e4,1e10)
#plt.ylim(1e-6,1e2)
plt.legend(loc='upper left')
plt.xlabel('$\ell$')
plt.savefig(dir_out+f'/figs/test.png',bbox_inches='tight')

# Save
#w_Emv_save = np.zeros((3,lmax+1))
#w_Emv_save[0,:] = w_Emv_new[:,0,0]
#w_Emv_save[1,:] = w_Emv_new[:,0,1]
#w_Emv_save[2,:] = w_Emv_new[:,0,2]
#np.savetxt('ilc_weights/weights1d_EE_spt3g_cmbmv_yuka.dat',w_Emv_save)
#w_Bmv_save = np.zeros((3,lmax+1))
#w_Bmv_save[0,:] = w_Bmv_new[:,0,0]
#w_Bmv_save[1,:] = w_Bmv_new[:,0,1]
#w_Bmv_save[2,:] = w_Bmv_new[:,0,2]
#np.savetxt('ilc_weights/weights1d_BB_spt3g_cmbmv_yuka.dat',w_Bmv_save)

#-----------------------------------------------------------------------------#

# Check my weights...
# Apply my weights to the noise maps, then compute the power spectrum,
# and see if they match my noise prediction
nlm_090_filename = dir_out + f'nlm/nlm_090_lmax{lmax}_seed999.alm'
nlm_150_filename = dir_out + f'nlm/nlm_150_lmax{lmax}_seed999.alm'
nlm_220_filename = dir_out + f'nlm/nlm_220_lmax{lmax}_seed999.alm'
nlmt_090,nlme_090,nlmb_090 = hp.read_alm(nlm_090_filename,hdu=[1,2,3])
nlmt_150,nlme_150,nlmb_150 = hp.read_alm(nlm_150_filename,hdu=[1,2,3])
nlmt_220,nlme_220,nlmb_220 = hp.read_alm(nlm_220_filename,hdu=[1,2,3])
elm_yuka = hp.almxfl(nlme_090,w_Emv_yuka[0][:lmax+1]) + hp.almxfl(nlme_150,w_Emv_yuka[1][:lmax+1]) + hp.almxfl(nlme_220,w_Emv_yuka[2][:lmax+1])
nlee_yuka = hp.alm2cl(elm_yuka,elm_yuka)
# Do the same thing using Yuuki’s weights and then compare the
# MV-ILC map noise level
elm_yuuki = hp.almxfl(nlme_090,w_Emv[0][:lmax+1]) + hp.almxfl(nlme_150,w_Emv[1][:lmax+1]) + hp.almxfl(nlme_220,w_Emv[2][:lmax+1])
nlee_yuuki = hp.alm2cl(elm_yuuki,elm_yuuki)

# Combine cross frequency spectra with ILC weights
ret = np.zeros((lmax+1,2))
for a in range(2):
    if a == 0: b='ee'; c=2; w1=w_Emv; w2=w_Emv
    if a == 1: b='ee'; c=2; w1=w_Emv_yuka; w2=w_Emv_yuka
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

# My weights should give lower noise
plt.figure(2)
plt.clf()
plt.plot(l, nlee_yuka, color='cornflowerblue', linestyle='--', alpha=0.8, label='nlee from map with Yuka MV-ILC weights')
plt.plot(l, nlee_yuuki, color='lightgreen', linestyle='--', alpha=0.8, label='nlee from map with Yuuki MV-ILC weights')
plt.plot(l, ret[:,1], color='darkblue', linestyle='-', alpha=0.8, label='nlee input spectrum combined with Yuka MV-ILC weights')
plt.plot(l, ret[:,0], color='forestgreen', linestyle='-', alpha=0.8, label='nlee input spectrum combined with Yuuki MV-ILC weights')
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-6,1e-5)
plt.xlim(200,lmax)
plt.legend(loc='upper left')
plt.title('ILC-weighted noise EE autospectra')
plt.ylabel("$C_\ell$")
plt.xlabel('$\ell$')
plt.savefig(dir_out+f'/figs/test.png',bbox_inches='tight')

