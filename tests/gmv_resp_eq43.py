import numpy as np
import os
import os.path
from scipy import special, integrate
from scipy.interpolate import interp1d
from time import time
from pathos.multiprocessing import ProcessingPool as Pool
import healpy as hp

class gmv_resp(object):
    '''
    Not implemented: doing semi = True (rlzcls defined) with crossilc = True
    '''

    def __init__(self,config,cltype,totalcls,u=None,crossilc=False,rlzcls=None,save_path=None):

        if crossilc:
            assert totalcls.shape[1]==11, "If temperature map T1 != T2, must provide cltt for both autospectra and cross spectrum"

        if not crossilc:
            cltt = totalcls[:,0]
            clee = totalcls[:,1]
            clbb = totalcls[:,2]
            clte = totalcls[:,3]

            self.totalTT = interp1d(np.arange(len(cltt)),cltt,kind='linear',bounds_error=False,fill_value=0.)
            self.totalEE = interp1d(np.arange(len(clee)),clee,kind='linear',bounds_error=False,fill_value=0.)
            self.totalBB = interp1d(np.arange(len(clbb)),clbb,kind='linear',bounds_error=False,fill_value=0.)
            self.totalTE = interp1d(np.arange(len(clte)),clte,kind='linear',bounds_error=False,fill_value=0.)
        else:
            # totalcls: T3T3, EE, BB, T3E, T1T1, T2T2, T1T2, T1T3, T2T3, T1E, T2E
            cltt3 = totalcls[:,0]
            cltt1 = totalcls[:,4]
            cltt2 = totalcls[:,5]
            clttx = totalcls[:,6]
            clee = totalcls[:,1]
            clbb = totalcls[:,2]
            clte = totalcls[:,3]
            clt1e = totalcls[:,9]
            clt2e = totalcls[:,10]

            self.totalTT1 = interp1d(np.arange(len(cltt1)),cltt1,kind='linear',bounds_error=False,fill_value=0.)
            self.totalTT2 = interp1d(np.arange(len(cltt2)),cltt2,kind='linear',bounds_error=False,fill_value=0.)
            self.totalTTx = interp1d(np.arange(len(clttx)),clttx,kind='linear',bounds_error=False,fill_value=0.)
            self.totalTT3 = interp1d(np.arange(len(cltt3)),cltt3,kind='linear',bounds_error=False,fill_value=0.)
            self.totalEE = interp1d(np.arange(len(clee)),clee,kind='linear',bounds_error=False,fill_value=0.)
            self.totalBB = interp1d(np.arange(len(clbb)),clbb,kind='linear',bounds_error=False,fill_value=0.)
            self.totalTE = interp1d(np.arange(len(clte)),clte,kind='linear',bounds_error=False,fill_value=0.)
            self.totalT1E = interp1d(np.arange(len(clt1e)),clte,kind='linear',bounds_error=False,fill_value=0.)
            self.totalT2E = interp1d(np.arange(len(clt2e)),clte,kind='linear',bounds_error=False,fill_value=0.)

        self.crossilc = crossilc
        self.Lmax = config['lensrec']['Lmax']
        self.l1Min = config['lensrec']['lmin']
        # Max value for l1 and l2 is taken to be same
        self.l1Max = max(config['lensrec']['lmaxT'],config['lensrec']['lmaxP'])
        self.u = interp1d(np.arange(len(u)), u, kind='linear', bounds_error=False, fill_value=0.) if u is not None else None
        self.save_path = save_path

        # L = l1 + l2; L for reconstructed phi field
        self.L = np.arange(self.Lmax+1)
        self.Nl = len(self.L)
        #TODO: Try increasing!
        self.N_phi = 50  # Number of steps for angular integration steps
        # Reduce to 50 if you need around 0.6% max accuracy til L = 3000
        # From 200 to 400, there is just 0.03% change in the noise curves til L=3000

        sl = {ii:config['cls'][cltype][ii] for ii in config['cls'][cltype].keys()}
        self.sltt = interp1d(np.arange(len(sl['tt'])), sl['tt'], kind='linear', bounds_error=False, fill_value=0.)
        self.slee = interp1d(np.arange(len(sl['ee'])), sl['ee'], kind='linear', bounds_error=False, fill_value=0.)
        self.slbb = interp1d(np.arange(len(sl['bb'])), sl['bb'], kind='linear', bounds_error=False, fill_value=0.)
        self.slte = interp1d(np.arange(len(sl['te'])), sl['te'], kind='linear', bounds_error=False, fill_value=0.)

        self.rlzcls = rlzcls
        self.semi = False if rlzcls is None else True

    """
    L = l1 + l2
    phi1 = angle betweeen vectors (L, l1)
    phi2 = angle betweeen vectors (L, l2)
    phi12 = phi1 - phi2
    """

    def l2(self, L, l_1, phi1):
        """
        This is mod of l2 = (L-11) given phi1
        """
        return np.sqrt(L**2 + l_1**2 - 2*L*l_1*np.cos(phi1))

    def phi12(self, L, l_1, phi1):
        """
        phi12 = phi1 - phi2
        """
        x = L*np.cos(-phi1) - l_1
        y = L*np.sin(-phi1)
        result = -np.arctan2(y, x)
        # Need negative sign because we want phi1 - phi2
        return result

    def phi2(self, L, l_1, phi1):
        """
        phi2 = phi1 - phi12
        """
        result = phi1 - self.phi12(L, l_1, phi1)
        return result

    def f_XY(self, L, l_1, phi1, XY):
        """
        Lensing response such that
        <X_l1 Y_{L-l1}> = f_XY(l1, L-l1)*\phi_L
        """
        l_2 = self.l2(L, l_1, phi1)
        phi12 = self.phi12(L, l_1, phi1)
        phi2 = self.phi2(L, l_1, phi1)

        Ldotl_1 = L*l_1*np.cos(phi1)
        Ldotl_2 = L*l_2*np.cos(phi2)

        if XY == 'TT':
            result = self.sltt(l_1)*Ldotl_1
            result += self.sltt(l_2)*Ldotl_2
        elif XY == 'EE':
            result = self.slee(l_1)*Ldotl_1
            result += self.slee(l_2)*Ldotl_2
            result *= np.cos(2.*phi12)
        elif XY == 'TE':
            # There is a typo in HO02!!!!!!!!!
            # Instead of cos(phi12) it should be cos(2*phi12)!!!!!
            result = self.slte(l_1)*np.cos(2.*phi12)*Ldotl_1
            result += self.slte(l_2)*Ldotl_2
        elif XY == 'TB':
            result = self.slte(l_1)*np.sin(2.*phi12)*Ldotl_1
        elif XY == 'EB':
            # There is a typo in HO02!!!!!!!!!
            # Instead of - it should be + between first and second term!!!!!
            result = self.slee(l_1)*Ldotl_1
            result += self.slbb(l_2)*Ldotl_2
            result *= np.sin(2.*phi12)
        elif XY == 'BB':
            result = self.slbb(l_1)*Ldotl_1
            result += self.slbb(l_2)*Ldotl_2
            result *= np.cos(2.*phi12)
        return result

    def f_XY_PRF(self, L, l_1, phi1, XY):
        """
        Lensing response such that
        <X_l1 Y_{L-l1}> = f_XY(l1, L-l1)*\phi_L
        """
        u = self.u

        if XY == 'TT':
            result = u(l_1)
        elif XY == 'EE':
            result = np.zeros(len(l_1))
        elif XY == 'TE':
            result = np.zeros(len(l_1))
        elif XY == 'TB':
            result = np.zeros(len(l_1))
        elif XY == 'EB':
            result = np.zeros(len(l_1))
        elif XY == 'BB':
            result = np.zeros(len(l_1))
        return result

    def Cl1(self, l_1):

        cl = np.zeros((len(l_1), 3, 3))

        if self.semi:
            ll  = np.arange(len(self.rlzcls[:,0]))
            tTT = interp1d(ll,self.rlzcls[:,0],kind='linear',bounds_error=False,fill_value=0.)
            tEE = interp1d(ll,self.rlzcls[:,1],kind='linear',bounds_error=False,fill_value=0.)
            tBB = interp1d(ll,self.rlzcls[:,2],kind='linear',bounds_error=False,fill_value=0.)
            tTE = interp1d(ll,self.rlzcls[:,3],kind='linear',bounds_error=False,fill_value=0.)
        else:
            if not self.crossilc:
                tTT = self.totalTT
                tEE = self.totalEE
                tBB = self.totalBB
                tTE = self.totalTE
            else:
                tTT = self.totalTT1
                tEE = self.totalEE
                tBB = self.totalBB
                tTE = self.totalT1E

        cl[:,0,0] = tTT(l_1)
        cl[:,1,1] = tEE(l_1)
        cl[:,2,2] = tBB(l_1)
        cl[:,0,1] = cl[:,1,0] = tTE(l_1)
        cl[:,0,2] = cl[:,2,0] = 0
        cl[:,1,2] = cl[:,2,1] = 0

        return cl

    def Cl2(self, l_2):

        cl = np.zeros((len(l_2), 3, 3))

        if self.semi:
            ll  = np.arange(len(self.rlzcls[:,0]))
            tTT = interp1d(ll,self.rlzcls[:,0],kind='linear',bounds_error=False,fill_value=0.)
            tEE = interp1d(ll,self.rlzcls[:,1],kind='linear',bounds_error=False,fill_value=0.)
            tBB = interp1d(ll,self.rlzcls[:,2],kind='linear',bounds_error=False,fill_value=0.)
            tTE = interp1d(ll,self.rlzcls[:,3],kind='linear',bounds_error=False,fill_value=0.)
        else:
            if not self.crossilc:
                tTT = self.totalTT
                tEE = self.totalEE
                tBB = self.totalBB
                tTE = self.totalTE
            else:
                tTT = self.totalTT2
                tEE = self.totalEE
                tBB = self.totalBB
                tTE = self.totalT2E

        cl[:,0,0] = tTT(l_2)
        cl[:,1,1] = tEE(l_2)
        cl[:,2,2] = tBB(l_2)
        cl[:,0,1] = cl[:,1,0] = tTE(l_2)
        cl[:,0,2] = cl[:,2,0] = 0
        cl[:,1,2] = cl[:,2,1] = 0

        return cl

    def f(self, L, l_1, phi1):

        l_2 = self.l2(L, l_1, phi1)
        phi2 = self.phi2(L, l_1, phi1)
        f = np.zeros((len(l_1), 3, 3))

        #f_TE_sym = (self.f_XY(L,l_1,phi1,'TE') + self.f_XY(L,l_2,phi2,'TE'))/2.
        #f_TE_asym = (self.f_XY(L,l_1,phi1,'TE') - self.f_XY(L,l_2,phi2,'TE'))/2.

        f[:,0,0] = self.f_XY(L, l_1, phi1, 'TT')
        f[:,1,1] = self.f_XY(L, l_1, phi1, 'EE')
        f[:,2,2] = self.f_XY(L, l_1, phi1, 'BB')
        f[:,0,1] = f[:,1,0] = self.f_XY(L, l_1, phi1, 'TE')
        f[:,0,2] = f[:,2,0] = self.f_XY(L, l_1, phi1, 'TB')
        f[:,0,3] = f[:,2,1] = self.f_XY(L, l_1, phi1, 'EB')

        return f

    def f_PRF(self, L, l_1, phi1):

        l_2 = self.l2(L, l_1, phi1)
        phi2 = self.phi2(L, l_1, phi1)
        f = np.zeros((len(l_1), 3, 3))

        f[:,0,0] = self.f_XY_PRF(L, l_1, phi1, 'TT')
        f[:,1,1] = self.f_XY_PRF(L, l_1, phi1, 'EE')
        f[:,2,2] = self.f_XY_PRF(L, l_1, phi1, 'BB')
        f[:,0,1] = f[:,1,0] = self.f_XY_PRF(L, l_1, phi1, 'TE')
        f[:,0,2] = f[:,2,0] = self.f_XY_PRF(L, l_1, phi1, 'TB')
        f[:,0,3] = f[:,2,1] = self.f_XY_PRF(L, l_1, phi1, 'EB')

        return f

    def Cl1_inv(self, l_1):

        inv_cl = np.zeros((len(l_1), 3, 3))

        if self.semi:
            ll  = np.arange(len(self.rlzcls[:,0]))
            tTT = interp1d(ll,self.rlzcls[:,0],kind='linear',bounds_error=False,fill_value=0.)
            tEE = interp1d(ll,self.rlzcls[:,1],kind='linear',bounds_error=False,fill_value=0.)
            tBB = interp1d(ll,self.rlzcls[:,2],kind='linear',bounds_error=False,fill_value=0.)
            tTE = interp1d(ll,self.rlzcls[:,3],kind='linear',bounds_error=False,fill_value=0.)
        else:
            if not self.crossilc:
                tTT = self.totalTT
                tEE = self.totalEE
                tBB = self.totalBB
                tTE = self.totalTE
            else:
                tTT = self.totalTT1
                tEE = self.totalEE
                tBB = self.totalBB
                tTE = self.totalT1E

        dl = tTT(l_1)*tEE(l_1) - tTE(l_1)*tTE(l_1)

        inv_cl[:,0,0] = tEE(l_1) / dl
        inv_cl[:,1,1] = tTT(l_1) / dl
        inv_cl[:,2,2] = 1 ./ tBB(l_1)
        inv_cl[:,0,1] = inv_cl[:,1,0] = -1*tTE(l_1) / dl
        inv_cl[:,0,2] = inv_cl[:,2,0] = 0
        inv_cl[:,1,2] = inv_cl[:,2,1] = 0

        return inv_cl

    def Cl2_inv(self, l_2):

        inv_cl = np.zeros((len(l_2), 3, 3))

        if self.semi:
            ll  = np.arange(len(self.rlzcls[:,0]))
            tTT = interp1d(ll,self.rlzcls[:,0],kind='linear',bounds_error=False,fill_value=0.)
            tEE = interp1d(ll,self.rlzcls[:,1],kind='linear',bounds_error=False,fill_value=0.)
            tBB = interp1d(ll,self.rlzcls[:,2],kind='linear',bounds_error=False,fill_value=0.)
            tTE = interp1d(ll,self.rlzcls[:,3],kind='linear',bounds_error=False,fill_value=0.)
        else:
            if not self.crossilc:
                tTT = self.totalTT
                tEE = self.totalEE
                tBB = self.totalBB
                tTE = self.totalTE
            else:
                tTT = self.totalTT2
                tEE = self.totalEE
                tBB = self.totalBB
                tTE = self.totalT2E

        dl = tTT(l_2)*tEE(l_2) - tTE(l_2)*tTE(l_2)

        inv_cl[:,0,0] = tEE(l_2) / dl
        inv_cl[:,1,1] = tTT(l_2) / dl
        inv_cl[:,2,2] = 1 ./ tBB(l_2)
        inv_cl[:,0,1] = inv_cl[:,1,0] = -1*tTE(l_2) / dl
        inv_cl[:,0,2] = inv_cl[:,2,0] = 0
        inv_cl[:,1,2] = inv_cl[:,2,1] = 0

        return inv_cl

    def A(self, L):

        l1min = self.l1Min
        l1max = self.l1Max

        if L > 2.*l1max:
            # L = l1 + l2 thus max L = 2*l1
            return 0.

        def integrand(l_1, phil, semi=self.semi):

            l_2 = self.l2(L, l_1, phil)
            phi2 = self.phi2(L, l_1, phil)

            if semi:
                #TODO
            else:
                inv_Cl2 = self.Cl2_inv(l_2)
                fl2l1 = self.f(L, l_2, phi2)
                inv_Cl1 = self.Cl1_inv(l_1)
                fl1l2 = self.f(L, l_1, phi1)
                res = np.einsum('ijk, ikl -> ijl', inv_Cl2, fl2l1)
                res = np.einsum('ijk, ikl -> ijl', fl1l2, res)
                res = np.einsum('ijk, ikl -> ijl', inv_Cl1, res)
            res *= 2*l_1 #TODO: not understanding this
            """
            Factor of 2 above because phi integral is symmetric.
            Thus we've put instead of 0 to 2pi, 2 times 0 to pi.
            Also, l_1^2 instead of l_1 if we are taking log spacing for l_1.
            """
            res /= (2.*np.pi)**2
            idx = np.where((l_1 < l1min) | (l_1 > l1max) | (l_2 < l1min) | (l_2 > l1max))[0]
            res[idx] = 0.
            return res

        l1 = np.linspace(l1min, l1max, int(l1max-l1min+1))
        # l1 = np.logspace(np.log10(l1min), np.log10(l1max), int(l1max-l1min+1))
        phi1 = np.linspace(0., np.pi, self.N_phi)
        int_1 = np.zeros(len(phi1))
        for i in range(len(phi1)):
            intgnd = integrand(l1, phi1[i])
            int_1[i] = integrate.simps(intgnd, x=l1, even='avg')

        int_ll = integrate.simps(int_1, x=phi1, even='avg')

        result = int_ll
        #result = 1./int_ll
        #result *= L**2
        # Factor of L**2 if we are calculating the reconstruction noise for d field instead of the phi field.

        if not np.isfinite(result):
            result = 0.

        return result

    def A_PRF(self, L, cross=False):
        '''
        If cross is True, does the calculation for the cross estimator response
        est*src.
        Note that we assume there is no cross-ILC stuff going on for profile hardening.
        '''
        l1min = self.l1Min
        l1max = self.l1Max

        if L > 2.*l1max:
            # L = l1 + l2 thus max L = 2*l1
            return 0.

        def integrand(l_1, phil, semi=self.semi):

            l_2 = self.l2(L, l_1, phil)
            phi2 = self.phi2(L, l_1, phil)

            if semi:
                #TODO
            else:
                if cross:
                    # For R^SKappa
                    fl1l2 = self.f(L, l_1, phi1)
                else:
                    # For R^SS
                    fl1l2 = self.f_PRF(L, l_1, phi1)
                inv_Cl1 = self.Cl1_inv(l_1)
                inv_Cl2 = self.Cl2_inv(l_2)
                fl2l1 = self.f_PRF(L, l_2, phi2)
                res = np.einsum('ijk, ikl -> ijl', inv_Cl2, fl2l1)
                res = np.einsum('ijk, ikl -> ijl', fl1l2, res)
                res = np.einsum('ijk, ikl -> ijl', inv_Cl1, res)
            res *= 2*l_1 #TODO: not understanding this
            """
            Factor of 2 above because phi integral is symmetric.
            Thus we've put instead of 0 to 2pi, 2 times 0 to pi.
            Also, l_1^2 instead of l_1 if we are taking log spacing for l_1.
            """
            res /= (2.*np.pi)**2
            idx = np.where((l_1 < l1min) | (l_1 > l1max) | (l_2 < l1min) | (l_2 > l1max))[0]
            res[idx] = 0.
            return res

        l1 = np.linspace(l1min, l1max, int(l1max-l1min+1))
        # l1 = np.logspace(np.log10(l1min), np.log10(l1max), int(l1max-l1min+1))
        phi1 = np.linspace(0., np.pi, self.N_phi)
        int_1 = np.zeros(len(phi1))
        for i in range(len(phi1)):
            intgnd = integrand(l1, phi1[i])
            int_1[i] = integrate.simps(intgnd, x=l1, even='avg')

        int_ll = integrate.simps(int_1, x=phi1, even='avg')

        result = int_ll
        #result = 1./int_ll
        #result *= L**2
        # Factor of L**2 if we are calculating the reconstruction noise for d field instead of the phi field.

        if not np.isfinite(result):
            result = 0.

        if result < 0.:
            print(L)

        return result

    def var_d(self, var_d1, var_d2):
        vard = var_d1+var_d2
        return vard

    def calc_tvar(self):
        data = np.zeros((self.Nl, 2))
        data[:, 0] = np.copy(self.L)
        pool = Pool(ncpus=4)

        def ff(l):
            return self.A(l)

        print("Computing variance")
        data[:, 1] = np.array(pool.map(ff, self.L))

        # N calculated here is 1/R
        if self.save_path:
            np.save(self.save_path, data)

    def calc_tvar_PRF(self, cross=False):
        data = np.zeros((self.Nl, 2))
        data[:, 0] = np.copy(self.L)
        pool = Pool(ncpus=4)

        def ff(l):
            return self.A_PRF(l,cross)

        print("Computing variance")
        data[:, 1] = np.array(pool.map(ff, self.L))

        #TODO: Need -1 fudge factor for cross response
        if cross:
            data[:,1] *= -1

        if self.save_path:
            np.save(self.save_path, data)

    def interp_tvar(self):
        print("Interpolating variances")

        self.N_d = {}
        data = np.load(self.save_path)

        L = data[:, 0]

        norm1 = data[:, 1].copy()
        self.N_d['d'] = interp1d(L, norm1, kind='linear', bounds_error=False, fill_value=0.)

