import numpy as np
import os
import os.path
from scipy import special, integrate
from scipy.interpolate import interp1d
from time import time
from pathos.multiprocessing import ProcessingPool as Pool
import healpy as hp

class gmv_resp_alt(object):
    '''
    Not implemented: doing semi = True (rlzcls defined) with crossilc = True
    ONLY changing M1_inv!
    '''

    def __init__(self,config,cltype,totalcls,u=None,crossilc=False,rlzcls=None,save_path=None):

        if crossilc:
            assert totalcls.shape[1]==5, "If temperature map T1 != T2, must provide cltt for both"

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
            cltt1 = totalcls[:,0]
            cltt2 = totalcls[:,1]
            clee = totalcls[:,2]
            clbb = totalcls[:,3]
            clte = totalcls[:,4]

            self.totalTT1 = interp1d(np.arange(len(cltt1)),cltt1,kind='linear',bounds_error=False,fill_value=0.)
            self.totalTT2 = interp1d(np.arange(len(cltt2)),cltt2,kind='linear',bounds_error=False,fill_value=0.)
            self.totalEE = interp1d(np.arange(len(clee)),clee,kind='linear',bounds_error=False,fill_value=0.)
            self.totalBB = interp1d(np.arange(len(clbb)),clbb,kind='linear',bounds_error=False,fill_value=0.)
            self.totalTE = interp1d(np.arange(len(clte)),clte,kind='linear',bounds_error=False,fill_value=0.)

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

    def M_1(self, L, l_1, phi1):

        l_2 = self.l2(L, l_1, phi1)
        m1 = np.zeros((len(l_1), 4, 4))

        if self.semi:
            ll  = np.arange(len(self.rlzcls[:,0]))
            tTT = interp1d(ll,self.rlzcls[:,0],kind='linear',bounds_error=False,fill_value=0.)
            tEE = interp1d(ll,self.rlzcls[:,1],kind='linear',bounds_error=False,fill_value=0.)
            tTE = interp1d(ll,self.rlzcls[:,3],kind='linear',bounds_error=False,fill_value=0.)
        else:
            if not self.crossilc:
                tTT = self.totalTT
                tEE = self.totalEE
                tTE = self.totalTE
            else:
                tTT1 = self.totalTT1
                tTT2 = self.totalTT2
                tEE = self.totalEE
                tTE = self.totalTE

        if not self.crossilc:
            m1[:, 0, 0] = 2.*tTT(l_1)*tTT(l_2)
            m1[:, 1, 1] = 2.*tEE(l_1)*tEE(l_2)
            m1[:, 2, 2] = 0.5*(tTT(l_1)*tEE(l_2) +
                               tEE(l_1)*tTT(l_2)) + \
                               tTE(l_1)*tTE(l_2)
            m1[:, 3, 3] = 0.5*(tTT(l_1)*tEE(l_2) +
                               tEE(l_1)*tTT(l_2)) - \
                               tTE(l_1)*tTE(l_2)
            m1[:, 0, 1] = m1[:, 1, 0] = 2.*tTE(l_1)*tTE(l_2)

            ################################

            m1[:, 0, 2] = m1[:, 2, 0] = (tTT(l_1)*tTE(l_2) +
                                         tTE(l_1)*tTT(l_2))
            m1[:, 0, 3] = m1[:, 3, 0] = (tTT(l_1)*tTE(l_2) -
                                         tTE(l_1)*tTT(l_2))
            m1[:, 1, 2] = m1[:, 2, 1] = (tEE(l_1)*tTE(l_2) +
                                         tTE(l_1)*tEE(l_2))
            m1[:, 1, 3] = m1[:, 3, 1] = -(tEE(l_1)*tTE(l_2) -
                                         tTE(l_1)*tEE(l_2))
            m1[:, 2, 3] = m1[:, 3, 2] = 0.5*(tTT(l_1)*tEE(l_2) -
                                             tEE(l_1)*tTT(l_2))
        else:
            m1[:, 0, 0] = 2.*tTT1(l_1)*tTT2(l_2)
            m1[:, 1, 1] = 2.*tEE(l_1)*tEE(l_2)
            m1[:, 2, 2] = 0.5*(tTT1(l_1)*tEE(l_2) +
                               tEE(l_1)*tTT2(l_2)) + \
                               tTE(l_1)*tTE(l_2)
            m1[:, 3, 3] = 0.5*(tTT1(l_1)*tEE(l_2) +
                               tEE(l_1)*tTT2(l_2)) - \
                               tTE(l_1)*tTE(l_2)
            m1[:, 0, 1] = m1[:, 1, 0] = 2.*tTE(l_1)*tTE(l_2)

            ################################

            m1[:, 0, 2] = m1[:, 2, 0] = (tTT1(l_1)*tTE(l_2) +
                                         tTE(l_1)*tTT2(l_2))
            m1[:, 0, 3] = m1[:, 3, 0] = (tTT1(l_1)*tTE(l_2) -
                                         tTE(l_1)*tTT2(l_2))
            m1[:, 1, 2] = m1[:, 2, 1] = (tEE(l_1)*tTE(l_2) +
                                         tTE(l_1)*tEE(l_2))
            m1[:, 1, 3] = m1[:, 3, 1] = -(tEE(l_1)*tTE(l_2) -
                                         tTE(l_1)*tEE(l_2))
            m1[:, 2, 3] = m1[:, 3, 2] = 0.5*(tTT1(l_1)*tEE(l_2) -
                                             tEE(l_1)*tTT2(l_2))

        return m1

    def f_1(self, L, l_1, phi1):

        l_2 = self.l2(L, l_1, phi1)
        phi2 = self.phi2(L, l_1, phi1)

        f_TE_sym = (self.f_XY(L,l_1,phi1,'TE') + self.f_XY(L,l_2,phi2,'TE'))/2.
        f_TE_asym = (self.f_XY(L,l_1,phi1,'TE') - self.f_XY(L,l_2,phi2,'TE'))/2.

        f1 = np.zeros((len(l_1), 4))
        f1[:, 0] = self.f_XY(L, l_1, phi1, 'TT')
        f1[:, 1] = self.f_XY(L, l_1, phi1, 'EE')
        f1[:, 2] = f_TE_sym
        f1[:, 3] = f_TE_asym
        return f1

    def f_1_PRF(self, L, l_1, phi1):

        l_2 = self.l2(L, l_1, phi1)
        phi2 = self.phi2(L, l_1, phi1)

        f_TE_sym = (self.f_XY_PRF(L,l_1,phi1,'TE') + self.f_XY_PRF(L,l_2,phi2,'TE'))/2.
        f_TE_asym = (self.f_XY_PRF(L,l_1,phi1,'TE') - self.f_XY_PRF(L,l_2,phi2,'TE'))/2.

        f1 = np.zeros((len(l_1), 4))
        f1[:, 0] = self.f_XY_PRF(L, l_1, phi1, 'TT')
        f1[:, 1] = self.f_XY_PRF(L, l_1, phi1, 'EE')
        f1[:, 2] = f_TE_sym
        f1[:, 3] = f_TE_asym
        return f1

    def M1_inv(self, L, l_1, phi1):

        l_2 = self.l2(L, l_1, phi1)
        nl2 = len(l_2)
        inv_m1 = np.zeros((nl2, 4, 4))

        if not self.crossilc:
            det = 4*self.totalTE(l_1)**2*self.totalTE(l_2)**2
            det *= (self.totalTE(l_1)**2*self.totalTE(l_2)**2 - \
                    2*self.totalEE(l_1)*self.totalTE(l_2)**2*self.totalTT(l_1) - \
                    2*self.totalEE(l_2)*self.totalTE(l_1)**2*self.totalTT(l_2) + \
                    3*self.totalEE(l_1)*self.totalEE(l_2)*self.totalTT(l_1)*self.totalTT(l_2))

            inv_m1[:,0,0] = 2*self.totalEE(l_1)*self.totalEE(l_2)* \
                            self.totalTE(l_1)**2*self.totalTE(l_2)**2
            inv_m1[:,1,1] = 2*self.totalTT(l_1)*self.totalTT(l_2)* \
                            self.totalTE(l_1)**2*self.totalTE(l_2)**2
            inv_m1[:,2,2] = 4*(self.totalTE(l_1)**2-self.totalEE(l_1)*self.totalTT(l_1))* \
                            (self.totalTE(l_1)*self.totalTE(l_2)+self.totalEE(l_2)*self.totalTT(l_1))* \
                            (self.totalTE(l_2)**2-self.totalEE(l_2)*self.totalTT(l_2))
            inv_m1[:,3,3] = -4*(self.totalTE(l_1)**2-self.totalEE(l_1)*self.totalTT(l_1))* \
                            (self.totalTE(l_1)*self.totalTE(l_2)-self.totalEE(l_2)*self.totalTT(l_1))* \
                            (self.totalTE(l_2)**2-self.totalEE(l_2)*self.totalTT(l_2))

            inv_m1[:,0,1] = 2*self.totalTE(l_1)**3*self.totalTE(l_2)* \
                            (self.totalTE(l_2)**2-2*self.totalEE(l_2)*self.totalTT(l_2))
            inv_m1[:,0,2] = -2*self.totalTE(l_1)* \
                            (self.totalEE(l_2)*self.totalTE(l_1)**2*self.totalTE(l_2)**2+ \
                             self.totalEE(l_1)*self.totalTE(l_1)*self.totalTE(l_2)**3- \
                             self.totalEE(l_1)*self.totalEE(l_2)*self.totalTE(l_2)**2*self.totalTT(l_1)- \
                             2*self.totalEE(l_2)**2*self.totalTE(l_1)**2*self.totalTT(l_2)- \
                             self.totalEE(l_1)*self.totalEE(l_2)*self.totalTE(l_1)*self.totalTE(l_2)*self.totalTT(l_2)+ \
                             2*self.totalEE(l_1)*self.totalEE(l_2)**2*self.totalTT(l_1)*self.totalTT(l_2))
            inv_m1[:,0,3] = 2*self.totalTE(l_1)* \
                            (self.totalEE(l_2)*self.totalTE(l_1)**2*self.totalTE(l_2)**2- \
                             self.totalEE(l_1)*self.totalTE(l_1)*self.totalTE(l_2)**3- \
                             self.totalEE(l_1)*self.totalEE(l_2)*self.totalTE(l_2)**2*self.totalTT(l_1)- \
                             2*self.totalEE(l_2)**2*self.totalTE(l_1)**2*self.totalTT(l_2)+ \
                             self.totalEE(l_1)*self.totalEE(l_2)*self.totalTE(l_1)*self.totalTE(l_2)*self.totalTT(l_2)+ \
                             2*self.totalEE(l_1)*self.totalEE(l_2)**2*self.totalTT(l_1)*self.totalTT(l_2))

            inv_m1[:,1,0] = 2*self.totalTE(l_1)*self.totalTE(l_2)**3* \
                            (self.totalTE(l_1)**2-2*self.totalEE(l_1)*self.totalTT(l_1))
            inv_m1[:,1,2] = -2*self.totalTE(l_2)* \
                            (self.totalTE(l_1)**2*self.totalTE(l_2)**2*self.totalTT(l_1)- \
                             2*self.totalEE(l_1)*self.totalTE(l_2)**2*self.totalTT(l_1)**2+ \
                             self.totalTE(l_1)**3*self.totalTE(l_2)*self.totalTT(l_2)- \
                             self.totalEE(l_2)*self.totalTE(l_1)**2*self.totalTT(l_1)*self.totalTT(l_2)- \
                             self.totalEE(l_1)*self.totalTE(l_1)*self.totalTE(l_2)*self.totalTT(l_1)*self.totalTT(l_2)+ \
                             2*self.totalEE(l_1)*self.totalEE(l_2)*self.totalTT(l_1)**2*self.totalTT(l_2))
            inv_m1[:,1,3] = 2*self.totalTE(l_2)* \
                            (self.totalTE(l_1)**2*self.totalTE(l_2)**2*self.totalTT(l_1)- \
                             2*self.totalEE(l_1)*self.totalTE(l_2)**2*self.totalTT(l_1)**2- \
                             self.totalTE(l_1)**3*self.totalTE(l_2)*self.totalTT(l_2)- \
                             self.totalEE(l_2)*self.totalTE(l_1)**2*self.totalTT(l_1)*self.totalTT(l_2)+ \
                             self.totalEE(l_1)*self.totalTE(l_1)*self.totalTE(l_2)*self.totalTT(l_1)*self.totalTT(l_2)+ \
                             2*self.totalEE(l_1)*self.totalEE(l_2)*self.totalTT(l_1)**2*self.totalTT(l_2))

            inv_m1[:,2,0] = -4*self.totalEE(l_2)*self.totalTE(l_1)*self.totalTE(l_2)**2* \
                            (self.totalTE(l_1)**2-self.totalEE(l_1)*self.totalTT(l_1))
            inv_m1[:,2,1] = -4*self.totalTE(l_1)**2*self.totalTE(l_2)*self.totalTT(l_1)* \
                            (self.totalTE(l_2)**2-self.totalEE(l_2)*self.totalTT(l_2))
            inv_m1[:,2,3] = 4*(-1*self.totalEE(l_2)*self.totalTE(l_1)**2*self.totalTE(l_2)**2*self.totalTT(l_1)+ \
                               self.totalEE(l_1)*self.totalTE(l_1)*self.totalTE(l_2)**3*self.totalTT(l_1)+ \
                               self.totalEE(l_1)*self.totalEE(l_2)*self.totalTE(l_2)**2*self.totalTT(l_1)**2+ \
                               self.totalEE(l_2)*self.totalTE(l_1)**3*self.totalTE(l_2)*self.totalTT(l_2)+ \
                               self.totalEE(l_2)**2*self.totalTE(l_1)**2*self.totalTT(l_1)*self.totalTT(l_2)- \
                               2*self.totalEE(l_1)*self.totalEE(l_2)*self.totalTE(l_1)*self.totalTE(l_2)*self.totalTT(l_1)*self.totalTT(l_2)- \
                               self.totalEE(l_1)*self.totalEE(l_2)**2*self.totalTT(l_1)**2*self.totalTT(l_2))

            inv_m1[:,3,0] = 4*self.totalEE(l_2)*self.totalTE(l_1)*self.totalTE(l_2)**2* \
                            (self.totalTE(l_1)**2-self.totalEE(l_1)*self.totalTT(l_1))
            inv_m1[:,3,1] = 4*self.totalTE(l_1)**2*self.totalTE(l_2)*self.totalTT(l_1)* \
                            (self.totalTE(l_2)**2-self.totalEE(l_2)*self.totalTT(l_2))
            inv_m1[:,3,2] = -4*(self.totalEE(l_2)*self.totalTE(l_1)**2*self.totalTE(l_2)**2*self.totalTT(l_1)+ \
                                self.totalEE(l_1)*self.totalTE(l_1)*self.totalTE(l_2)**3*self.totalTT(l_1)- \
                                self.totalEE(l_1)*self.totalEE(l_2)*self.totalTE(l_2)**2*self.totalTT(l_1)**2+ \
                                self.totalEE(l_2)*self.totalTE(l_1)**3*self.totalTE(l_2)*self.totalTT(l_2)- \
                                self.totalEE(l_2)**2*self.totalTE(l_1)**2*self.totalTT(l_1)*self.totalTT(l_2)- \
                                2*self.totalEE(l_1)*self.totalEE(l_2)*self.totalTE(l_1)*self.totalTE(l_2)*self.totalTT(l_1)*self.totalTT(l_2)+ \
                                self.totalEE(l_1)*self.totalEE(l_2)**2*self.totalTT(l_1)**2*self.totalTT(l_2))

        else:
            det = self.totalTT1(l_1)*self.totalEE(l_1)-self.totalTE(l_1)**2
            det *= self.totalTT2(l_2)*self.totalEE(l_2)-self.totalTE(l_2)**2
            # Determinant = 1./det
            inv_m1[:, 0, 0] = 0.5*self.totalEE(l_1)*self.totalEE(l_2)
            inv_m1[:, 1, 1] = 0.5*self.totalTT1(l_1)*self.totalTT2(l_2)
            inv_m1[:, 2, 2] = 0.5*(self.totalTT1(l_1)*self.totalEE(l_2) +
                                   self.totalEE(l_1)*self.totalTT2(l_2)) + \
                                   self.totalTE(l_1)*self.totalTE(l_2)
            inv_m1[:, 3, 3] = 0.5*(self.totalTT1(l_1)*self.totalEE(l_2) +
                                   self.totalEE(l_1)*self.totalTT2(l_2)) - \
                                   self.totalTE(l_1)*self.totalTE(l_2)

            inv_m1[:, 0, 1] = inv_m1[:, 1, 0] = 0.5*self.totalTE(l_1)*self.totalTE(l_2)
            inv_m1[:, 0, 2] = inv_m1[:, 2, 0] = -0.5*(self.totalEE(l_1)*self.totalTE(l_2) +
                                                      self.totalTE(l_1)*self.totalEE(l_2))
            inv_m1[:, 0, 3] = inv_m1[:, 3, 0] = 0.5*(self.totalTE(l_1)*self.totalEE(l_2) -
                                                     self.totalEE(l_1)*self.totalTE(l_2))

            inv_m1[:, 1, 2] = inv_m1[:, 2, 1] = -0.5*(self.totalTT1(l_1)*self.totalTE(l_2) +
                                                      self.totalTE(l_1)*self.totalTT2(l_2))
            inv_m1[:, 1, 3] = inv_m1[:, 3, 1] = -0.5*(self.totalTE(l_1)*self.totalTT2(l_2) -
                                                      self.totalTT1(l_1)*self.totalTE(l_2))
            inv_m1[:, 2, 3] = inv_m1[:, 3, 2] = 0.5*(self.totalEE(l_1)*self.totalTT2(l_2) -
                                                     self.totalTT1(l_1)*self.totalEE(l_2))

        return inv_m1/det[:, None, None]

    def F1prime(self, L, l_1, phi1):
        """
        F1 = A1(L)*M1^{-1}*f1
        F1prime = M1^{-1}*f1
        """
        f_1 = self.f_1(L, l_1, phi1)
        M1_inv = self.M1_inv(L, l_1, phi1)
        M1invf1 = np.einsum('ijk, ij -> ik', M1_inv, f_1)
        return M1invf1

    def F1prime_PRF(self, L, l_1, phi1):
        """
        F1 = A1(L)*M1^{-1}*f1
        F1prime = M1^{-1}*f1
        """
        f_1 = self.f_1_PRF(L, l_1, phi1)
        M1_inv = self.M1_inv(L, l_1, phi1)
        M1invf1 = np.einsum('ijk, ij -> ik', M1_inv, f_1)
        return M1invf1

    def A_1(self, L):

        l1min = self.l1Min
        l1max = self.l1Max

        if L > 2.*l1max:
            # L = l1 + l2 thus max L = 2*l1
            return 0.

        def integrand(l_1, phil, semi=self.semi):

            l_2 = self.l2(L, l_1, phil)
            M1invf1 = self.F1prime(L, l_1, phil)
            if semi:
                M1 = self.M_1(L, l_1, phil)
                M1_M1invf1 = np.einsum('ijk, ij -> ik', M1, M1invf1)
                result = np.sum(M1invf1*M1_M1invf1, -1)
            else:
                f_1 = self.f_1(L, l_1, phil)
                Fdotf = np.sum(M1invf1*f_1, -1)
                result = Fdotf
            result *= 2*l_1
            """
            Factor of 2 above because phi integral is symmetric.
            Thus we've put instead of 0 to 2pi, 2 times 0 to pi.
            Also, l_1^2 instead of l_1 if we are taking log spacing for l_1.
            """
            result /= (2.*np.pi)**2
            idx = np.where((l_1 < l1min) | (l_1 > l1max) | (l_2 < l1min) | (l_2 > l1max))[0]
            result[idx] = 0.
            return result

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

    def A_1_PRF(self, L, cross=False):
        '''
        If cross is True, does the calculation for the cross estimator response
        est*src.
        '''
        l1min = self.l1Min
        l1max = self.l1Max

        if L > 2.*l1max:
            # L = l1 + l2 thus max L = 2*l1
            return 0.

        def integrand(l_1, phil, semi=self.semi):

            l_2 = self.l2(L, l_1, phil)

            if cross:
                # For R^SKappa
                M1invf1 = self.F1prime(L, l_1, phil)
            else:
                # For R^SS
                M1invf1 = self.F1prime_PRF(L, l_1, phil)
            if semi:
                M1 = self.M_1(L, l_1, phil)
                M1_M1invf1 = np.einsum('ijk, ij -> ik', M1, M1invf1)
                if cross: M1invf1 = self.F1prime_PRF(L, l_1, phil)
                result = np.sum(M1invf1*M1_M1invf1, -1)
            else:
                f_1 = self.f_1_PRF(L, l_1, phil)
                Fdotf = np.sum(M1invf1*f_1, -1)
                result = Fdotf
            result *= 2*l_1
            """
            Factor of 2 above because phi integral is symmetric.
            Thus we've put instead of 0 to 2pi, 2 times 0 to pi.
            Also, l_1^2 instead of l_1 if we are taking log spacing for l_1.
            """
            result /= (2.*np.pi)**2
            idx = np.where((l_1 < l1min) | (l_1 > l1max) | (l_2 < l1min) | (l_2 > l1max))[0]
            result[idx] = 0.
            return result

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

    def M_2(self, L, l_1, phi1):
        m2 = np.zeros((len(l_1), 2, 2))
        l_2 = self.l2(L, l_1, phi1)

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
                tTT1 = self.totalTT1
                tTT2 = self.totalTT2
                tEE = self.totalEE
                tBB = self.totalBB
                tTE = self.totalTE

        if not self.crossilc:
            m2[:, 0, 0] = (tTT(l_1)*tBB(l_2))
            m2[:, 1, 1] = (tEE(l_1)*tBB(l_2))
            m2[:, 0, 1] = m2[:, 1, 0] = (tTE(l_1)*tBB(l_2))
        else:
            m2[:, 0, 0] = (tTT1(l_1)*tBB(l_2))
            m2[:, 1, 1] = (tEE(l_1)*tBB(l_2))
            m2[:, 0, 1] = m2[:, 1, 0] = (tTE(l_1)*tBB(l_2))

        return m2

    def f_2(self, L, l_1, phi1):

        f2 = np.zeros((len(l_1), 2))
        f2[:, 0] = self.f_XY(L, l_1, phi1, 'TB')
        f2[:, 1] = self.f_XY(L, l_1, phi1, 'EB')

        return f2

    def f_2_PRF(self, L, l_1, phi1):

        f2 = np.zeros((len(l_1), 2))
        f2[:, 0] = self.f_XY_PRF(L, l_1, phi1, 'TB')
        f2[:, 1] = self.f_XY_PRF(L, l_1, phi1, 'EB')

        return f2

    def M2_inv(self, L, l_1, phi1):

        if not self.crossilc:
            l_2 = self.l2(L, l_1, phi1)
            nl2 = len(l_2)
            inv_m2 = np.zeros((nl2, 2, 2))
            det = self.totalTT(l_1)*self.totalEE(l_1)*self.totalBB(l_2)**2
            det -= self.totalTE(l_1)**2*self.totalBB(l_2)**2

            inv_m2[:, 0, 0] = self.totalEE(l_1)*self.totalBB(l_2)
            inv_m2[:, 1, 1] = self.totalTT(l_1)*self.totalBB(l_2)
            inv_m2[:, 0, 1] = inv_m2[:, 1, 0] = -self.totalTE(l_1)*self.totalBB(l_2)
        else:
            l_2 = self.l2(L, l_1, phi1)
            nl2 = len(l_2)
            inv_m2 = np.zeros((nl2, 2, 2))
            det = self.totalTT1(l_1)*self.totalEE(l_1)*self.totalBB(l_2)**2
            det -= self.totalTE(l_1)**2*self.totalBB(l_2)**2

            inv_m2[:, 0, 0] = self.totalEE(l_1)*self.totalBB(l_2)
            inv_m2[:, 1, 1] = self.totalTT1(l_1)*self.totalBB(l_2)
            inv_m2[:, 0, 1] = inv_m2[:, 1, 0] = -self.totalTE(l_1)*self.totalBB(l_2)

        return inv_m2/det[:, None, None]

    def F2prime(self, L, l_1, phi1):
        """
        F2 = A2(L)*M2^{-1}*f2
        F2prime = M2^{-1}*f2
        """
        f_2 = self.f_2(L, l_1, phi1)
        M2_inv = self.M2_inv(L, l_1, phi1)
        M2invf2 = np.einsum('ijk, ij -> ik', M2_inv, f_2)
        return M2invf2

    def F2prime_PRF(self, L, l_1, phi1):
        """
        F2 = A2(L)*M2^{-1}*f2
        F2prime = M2^{-1}*f2
        """
        f_2 = self.f_2_PRF(L, l_1, phi1)
        M2_inv = self.M2_inv(L, l_1, phi1)
        M2invf2 = np.einsum('ijk, ij -> ik', M2_inv, f_2)
        return M2invf2

    def A_2(self, L):

        l1min = self.l1Min
        l1max = self.l1Max
        if L > 2.*l1max:
            # L = l1 + l2 thus max L = 2*l1
            return 0.

        def integrand(l_1, phil, semi=self.semi):
            l_2 = self.l2(L, l_1, phil)
            F2p = self.F2prime(L, l_1, phil)
            if semi:
                M2 = self.M_2(L, l_1, phil)
                M2_M2invf2 = np.einsum('ijk, ij -> ik', M2, F2p)
                result = np.sum(F2p*M2_M2invf2, -1)
            else:
                f_2 = self.f_2(L, l_1, phil)
                Fdotf = np.sum(F2p*f_2, -1)
                result = Fdotf
            result *= 2*l_1  # **2
            """
            Factor of 2 above because phi integral is symmetric.
            Thus we've put instead of 0 to 2pi, 2 times 0 to pi.
            Also, l_1^2 instead of l_1 if we are taking log spacing for l_1.
            """
            result /= (2.*np.pi)**2
            idx = np.where((l_1 < l1min) | (l_1 > l1max) | (l_2 < l1min) | (l_2 > l1max))
            result[idx] = 0.
            return result

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

    def A_2_PRF(self, L, cross=False):

        l1min = self.l1Min
        l1max = self.l1Max
        # """
        if L > 2.*l1max:
            # L = l1 + l2 thus max L = 2*l1
            return 0.

        def integrand(l_1, phil):
            l_2 = self.l2(L, l_1, phil)

            if cross:
                # For R^SKappa
                F2p = self.F2prime(L, l_1, phil)
            else:
                # For R^SS
                F2p = self.F2prime_PRF(L, l_1, phil)
            f_2 = self.f_2_PRF(L, l_1, phil)
            Fdotf = np.sum(F2p*f_2, -1)
            result = Fdotf
            result *= 2*l_1
            """
            Factor of 2 above because phi integral is symmetric.
            Thus we've put instead of 0 to 2pi, 2 times 0 to pi.
            Also, l_1^2 instead of l_1 if we are taking log spacing for l_1.
            """
            result /= (2.*np.pi)**2
            idx = np.where((l_1 < l1min) | (l_1 > l1max) | (l_2 < l1min) | (l_2 > l1max))
            result[idx] = 0.
            return result

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

        if not np.isfinite(result):
            result = 0.

        if result < 0.:
            print(L)

        return result

    def var_d(self, var_d1, var_d2):
        vard = var_d1+var_d2
        return vard

    def calc_tvar(self):
        data = np.zeros((self.Nl, 4))
        data[:, 0] = np.copy(self.L)
        pool = Pool(ncpus=4)

        def f1(l):
            return self.A_1(l)

        def f2(l):
            return self.A_2(l)

        print("Computing variance for d1")
        data[:, 1] = np.array(pool.map(f1, self.L))

        print("Computing variance for d2")
        data[:, 2] = np.array(pool.map(f2, self.L))

        print("Computing variance for d")
        data[:, 3] = self.var_d(data[:, 1], data[:, 2])

        # N calculated here is 1/R
        #data[:,1] = 1/data[:,1]
        #data[:,2] = 1/data[:,2]
        #data[:,3] = 1/data[:,3]
        if self.save_path:
            np.save(self.save_path, data)

    def calc_tvar_PRF(self, cross=False):
        data = np.zeros((self.Nl, 4))
        data[:, 0] = np.copy(self.L)
        pool = Pool(ncpus=4)

        def f1(l):
            return self.A_1_PRF(l,cross)

        def f2(l):
            return self.A_2_PRF(l,cross)

        print("Computing variance for d1")
        data[:, 1] = np.array(pool.map(f1, self.L))

        print("Computing variance for d2")
        data[:, 2] = np.array(pool.map(f2, self.L))

        print("Computing variance for d")
        data[:, 3] = self.var_d(data[:, 1], data[:, 2])

        # N calculated here is 1/R
        #data[:,1] = 1/data[:,1]
        #data[:,2] = 1/data[:,2]
        #data[:,3] = 1/data[:,3]

        #TODO: Need -1 fudge factor for cross response
        if cross:
            data[:,1] *= -1
            data[:,2] *= -1
            data[:,3] *= -1

        if self.save_path:
            np.save(self.save_path, data)

    def interp_tvar(self):
        print("Interpolating variances")

        self.N_d = {}
        data = np.load(self.save_path)

        L = data[:, 0]

        norm1 = data[:, 1].copy()
        self.N_d['d1'] = interp1d(L, norm1, kind='linear', bounds_error=False, fill_value=0.)

        norm2 = data[:, 2].copy()
        self.N_d['d2'] = interp1d(L, norm2, kind='linear', bounds_error=False, fill_value=0.)

        norm = data[:, 3].copy()
        self.N_d['d'] = interp1d(L, norm, kind='quadratic', bounds_error=False, fill_value=0.)
