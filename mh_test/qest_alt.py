import os,sys,pickle
import weights,resp
import numpy as np
import healpy as hp
import healqest_utils as utils

np.seterr(all='ignore')

class qest(object):

    def __init__(self,config,cls):
        '''
        Set up the quadratic estimator calculation

        Parameters
        ----------
        config : dict
          Dictionary of lmin/lmax settings
        els: dict
          Dictionary of cls
        almbar1: complex
          First filtered alms
        almbar2: complex
          Second filtered alms
        '''

        #assert est=='lens' or est=='src' or est=='prof', "est expected to be lens/src/prof, got: %s"%est
        #assert cltype=='grad' or cltype=='len' or cltype=='unl', "cltype expected to be grad/len/unl, got: %s"%cltype

        print('Setting up lensing reconstruction')
        self.config  = config
        #self.almbar1 = almbar1
        #self.almbar2 = almbar2

        self.lmax    = self.config['lensrec']['lmax'] = max(config['lensrec']['lmaxT'],config['lensrec']['lmaxP'])
        self.Lmax    = self.config['lensrec']['Lmax']
        self.cltype  = self.config['lensrec']['cltype']
        self.glm = {}
        self.clm = {}

        self.cls     = cls

        if self.cltype!='ucmb' and self.cltype!='lcmb' and self.cltype!='gcmb':
            sys.exit('cltype must be ucmb, lcmb or gcmb')

        if 'nside' in self.config['lensrec']:
            print("-- Overwrite default nside")
            self.nside = self.config['lensrec']['nside'] # Overwrite automatic setting of nside<2*lmax
        else:
            self.nside   = utils.get_nside(self.Lmax)

        print("-- Nside to project: %d"%self.nside)
        print("-- lmax:%d"%self.lmax)
        print("-- Lmax:%d"%self.Lmax)
        print("-- Using %s cls"%self.cltype)

    def eval(self,qe,almbar1,almbar2,u=None):
        '''
        Compute quadratic estimator

        Parameters
        ----------
        qe : str
          Quadratic estimator type: 'TT'/'EE'/'TE'/'EB'/'TB'/'TTprf'
        almbar1: complex array healpy alm
          First filtered alm
        almbar2: complex array healpy alm
          Second filtered alm
        u  : profile
          Profile instance

        Returns
        ----------
        glm: complex
          Gradient component of the plm
        clm:
          Curl component of the plm
        '''

        if qe in self.glm:
            print("We've already computed this!")
        else:
            #if qe is None:
            #sys.exit('Need to specify estimator')
            if qe == 'TTprf':
                assert u is not None, "Need profile function to compute this estimator"

            #ef __init__(self,config,cls,est,u=None,totalcls=None):
            q = weights.weights(qe, self.cls[self.cltype], self.lmax, u=u)

            #sys.exit()
            print('Running lensing reconstruction')

            if qe=='TB' or qe=='EB':
                # Hack to get TB/EB working. currently not understanding some factors of j
                print('WARNING: Currently using a hacky implementation for TB/EB -- should probably revisit!')

                wX1,wY1,wP1,sX1,sY1,sP1 = q.w[0][0],q.w[0][1],q.w[0][2],q.s[0][0],q.s[0][1],q.s[0][2]
                wX3,wY3,wP3,sX3,sY3,sP3 = q.w[2][0],q.w[2][1],q.w[2][2],q.s[2][0],q.s[2][1],q.s[2][2]

                walmbar1 = hp.almxfl(almbar1,wX1) # T1/E1
                walmbar3 = hp.almxfl(almbar1,wX3) # T3/E3
                walmbar2 = hp.almxfl(almbar2,wY1) # B2

                SpX1, SmX1 = hp.alm2map_spin([walmbar1,np.zeros_like(walmbar1)], self.nside, 1, self.lmax)
                SpX3, SmX3 = hp.alm2map_spin([walmbar3,np.zeros_like(walmbar3)], self.nside, 3, self.lmax)
                SpY2, SmY2 = hp.alm2map_spin([np.zeros_like(walmbar2),-1j*walmbar2], self.nside, 2, self.lmax)
                #SpY2, SmY2 = hp.alm2map_spin([np.zeros_like(walmbar2),-1j*walmbar2], self.nside, 2, self.lmax)

                SpZ =  SpY2*(SpX1-SpX3) + SmY2*(SmX1-SmX3)
                SmZ = -SpY2*(SmX1+SmX3) + SmY2*(SpX1+SpX3)

                glm,clm = hp.map2alm_spin([SpZ,SmZ],1,self.Lmax)

                if qe=='TT' or qe=='EE' or qe=='TE' or qe=='ET':
                    nrm=0.5
                elif qe=='EB':
                    nrm=-1
                else:
                    nrm=1

                self.glm[qe] = hp.almxfl(glm,nrm*wP1)
                self.clm[qe] = hp.almxfl(clm,nrm*wP1)

            elif qe=='BT' or qe=='BE':
                # Hack to get TB/EB working. currently not understanding some factors of j
                print('WARNING: Currently using a hacky implementation for TB/EB -- should probably revisit!')
                print('using est: %s'%qe )
                wX1,wY1,wP1,sX1,sY1,sP1 = q.w[0][0],q.w[0][1],q.w[0][2],q.s[0][0],q.s[0][1],q.s[0][2]
                wX3,wY3,wP3,sX3,sY3,sP3 = q.w[2][0],q.w[2][1],q.w[2][2],q.s[2][0],q.s[2][1],q.s[2][2]

                walmbar1 = hp.almxfl(almbar2,wY1)
                walmbar3 = hp.almxfl(almbar2,wY3)
                walmbar2 = hp.almxfl(almbar1,wX1)

                SpX1, SmX1 = hp.alm2map_spin([walmbar1,np.zeros_like(walmbar1)], self.nside, 1, self.lmax)
                SpX3, SmX3 = hp.alm2map_spin([walmbar3,np.zeros_like(walmbar3)], self.nside, 3, self.lmax)
                SpY2, SmY2 = hp.alm2map_spin([np.zeros_like(walmbar2),-1j*walmbar2], self.nside, 2, self.lmax)
                #SpY2, SmY2 = hp.alm2map_spin([np.zeros_like(walmbar2),-1j*walmbar2], self.nside, 2, self.lmax)

                SpZ =  SpY2*(SpX1-SpX3) + SmY2*(SmX1-SmX3)
                SmZ = -SpY2*(SmX1+SmX3) + SmY2*(SpX1+SpX3)

                glm,clm = hp.map2alm_spin([SpZ,SmZ],1,self.Lmax)

                if qe=='TT' or qe=='EE' or qe=='TE' or qe=='ET':
                    nrm=0.5
                elif qe=='BE':
                    nrm=-1
                else:
                    nrm=1

                self.glm[qe] = hp.almxfl(glm,nrm*wP1)
                self.clm[qe] = hp.almxfl(clm,nrm*wP1)

            else:
                # More traditional quicklens style calculation
                retglm  = 0
                retclm  = 0

                for i in range(0,q.ntrm):

                    wX,wY,wP,sX,sY,sP = q.w[i][0],q.w[i][1],q.w[i][2],q.s[i][0],q.s[i][1],q.s[i][2]
                    print("-- Computing term %d/%d, sj = [%d,%d,%d]"%(i+1,q.ntrm,sX,sY,sP))
                    walmbar1 = hp.almxfl(almbar1,wX)
                    walmbar2 = hp.almxfl(almbar2,wY)

                    # Input takes in a^+ and a^-, but in this case we are inserting spin-0 maps i.e. tlm,elm,blm
                    #-----------------------------------------------------------------------------------------------

                    if qe[0]=='B':
                        SpX, SmX = hp.alm2map_spin([np.zeros_like(walmbar1),1j*walmbar1],self.nside,np.abs(sX),self.lmax)
                        sys.exit('broken')
                    else:
                        SpX, SmX = hp.alm2map_spin([walmbar1,np.zeros_like(walmbar1)],self.nside,np.abs(sX),self.lmax)

                    X  = SpX+1j*SmX # Complex map _{+s}S or _{-s}S

                    if sX<0:
                        X = np.conj(X)*(-1)**(sX)
                    #-----------------------------------------------------------------------------------------------
                    if qe[1]=='B':
                        SpY, SmY = hp.alm2map_spin([np.zeros_like(walmbar2),1j*walmbar2],self.nside,np.abs(sY),self.lmax)
                        sys.exit('broken')
                    else:
                        SpY, SmY = hp.alm2map_spin([walmbar2,np.zeros_like(walmbar2)],self.nside,np.abs(sY),self.lmax)

                    Y  = SpY+1j*SmY

                    if sY<0:
                        Y = np.conj(Y)*(-1)**(sY)

                    #-----------------------------------------------------------------------------------------------

                    XY = X*Y

                    if sP<0:
                        XY = np.conj(XY)*(-1)**(sP)

                    glm,clm  = hp.map2alm_spin([XY.real,XY.imag], np.abs(sP), self.Lmax)

                    glm = hp.almxfl(glm,0.5*wP)
                    clm = hp.almxfl(clm,0.5*wP)

                    retglm  += glm
                    retclm  += clm

                self.glm[qe] = retglm
                self.clm[qe] = retclm
                #if qe == self.qe:
                #    self.retglm = retglm
                #    self.retclm = retclm
                #elif qe == 'TTprf':
                #    self.retglm_prf = retglm
                #    self.retclm_prf = retclm

        return self.glm[qe], self.clm[qe]

    def get_aresp(self,flX,flY,qe1=None,qe2=None,u=None):
        '''
        Compute analytical response function for 1D filtering

        Parameters
        ----------
        flX, flY
          1D real arrays representing the filter functions for the X and Y fields
        qe1: string
          First estimator
        qe2: string
          Second estimator; if None, assumes it is the same as qe1

        Returns
        ----------
        aresp:
          Analytical response function
        '''
        if qe1 is None:
            assert 0, "qe1 must be defined"

        qeXY = weights.weights(qe1, self.cls[self.cltype], self.lmax, u=u)

        if qe2 is None or qe2==qe1:
            qeZA = None
        else:
            qeZA = weights.weights(qe2, self.cls[self.cltype], self.lmax, u=u)

        aresp = resp.fill_resp(qeXY,np.zeros(self.Lmax+1, dtype=np.complex_),flX,flY,qeZA=qeZA)

        return aresp

    def harden(self, qe, almbar1, almbar2, flX, flY, u, qe_hrd='TTprf'):
        '''
        Get the source hardened glm and the response function.
        Need arguments flX, flY in order to compute the analytical response
        needed for hardening.

        Parameters
        ----------
        flX, flY
          1D real arrays representing the filter functions for the X and Y fields

        Returns
        ----------
        plm :
          Source hardened glm
        resp :
          Response function
        '''
        assert qe=='TT', "We only harden for qe 'TT', got: %s"%qe

        ss = self.get_aresp(flX, flY, qe1=qe_hrd, u=u)
        es = self.get_aresp(flX, flY, qe1=qe_hrd, qe2=qe, u=u)
        ee = self.get_aresp(flX, flY, qe1=qe)

        plm1,_ = self.eval(qe,almbar1,almbar2)
        plm2,_ = self.eval(qe_hrd,almbar1,almbar2,u)

        weight = -1*es/ss
        plm    = plm1 + hp.almxfl(plm2, weight)
        resp   = ee + weight*es

        return plm, resp

class qest_gmv(object):

    def __init__(self,config,cls):
        '''
        Set up the quadratic estimator calculation for GMV

        Parameters
        ----------
        config : dict
          Dictionary of settings
        qe  : str
          Quadratic estimator type: 'all'/'TTEETE' (TT/EE/TE only)/'TBEB' (TB/EB only)
        alm1all: complex
          First unfiltered alms; N x 5 arrays for each of the 5 estimators in the order TT/EE/TE/TB/EB
        alm2all: complex
          Second unfiltered alms; N x 5 arrays for each of the 5 estimators in the order TT/EE/TE/TB/EB
        totalcls:
          The signal + noise spectra for TT, EE, BB, TE needed for the weights
        cltype : str
          Should be one of 'grad'/'len'/'unl'
        '''
        import gmv_resp_alt as gmv_resp

        print('Setting up lensing reconstruction')
        self.config     = config
        self.lmax    = self.config['lensrec']['lmax'] = max(config['lensrec']['lmaxT'],config['lensrec']['lmaxP'])
        self.Lmax    = self.config['lensrec']['Lmax']
        self.cltype  = self.config['lensrec']['cltype']
        self.cls     = cls
        self.glm = {}
        self.clm = {}

        if self.cltype!='ucmb' and self.cltype!='lcmb' and self.cltype!='grad':
            sys.exit('cltype must be ucmb, lcmb or grad')

        if 'nside' in self.config['lensrec']:
            print("-- Overwrite default nside")
            self.nside = self.config['lensrec']['nside'] # Overwrite automatic setting of nside<2*lmax
        else:
            self.nside   = utils.get_nside(self.Lmax)

        print("-- Nside to project: %d"%self.nside)
        print("-- lmax:%d"%self.lmax)
        print("-- Lmax:%d"%self.Lmax)
        print("-- Using %s cls"%self.cltype)

    def eval(self,qe,alm1all,alm2all,totalcls,u=None,crossilc=False):
        '''
        Compute quadratic estimator

        Parameters
        ----------
        qe : str
          Quadratic estimator type: 'all'/'TTEETE' (TT/EE/TE only)/'TBEB' (TB/EB only)/'TTEETEprf'

        Returns
        ----------
        glm: complex
          Gradient component of the plm
        clm:
          Curl component of the plm
        '''
        if qe in self.glm:
            print("We've already computed this!")
        else:
            if qe == 'TTEETEprf':
                assert u is not None, "Need profile function to compute this estimator"

            if qe == 'all':
                if crossilc:
                    ests = ['TT_GMV', 'TT_GMV', 'EE_GMV', 'TE_GMV', 'ET_GMV', 'TB_GMV', 'BT_GMV', 'EB_GMV', 'BE_GMV']
                    idxs = [0, 1, 2, 3, 4, 5, 6, 7, 8]
                else:
                    ests = ['TT_GMV', 'EE_GMV', 'TE_GMV', 'ET_GMV', 'TB_GMV', 'BT_GMV', 'EB_GMV', 'BE_GMV']
                    idxs = [0, 1, 2, 3, 4, 5, 6, 7]
            elif qe == 'TTEETE':
                if crossilc:
                    ests = ['TT_GMV', 'TT_GMV', 'EE_GMV', 'TE_GMV', 'ET_GMV']
                    idxs = [0, 1, 2, 3, 4]
                else:
                    ests = ['TT_GMV', 'EE_GMV', 'TE_GMV', 'ET_GMV']
                    idxs = [0, 1, 2, 3]
            elif qe == 'TBEB':
                ests = ['TB_GMV', 'BT_GMV', 'EB_GMV', 'BE_GMV']
                if crossilc:
                    idxs = [5, 6, 7, 8]
                else:
                    idxs = [4, 5, 6, 7]
            elif qe == 'TTEETEprf':
                ests = ['TT_GMV_PRF', 'EE_GMV_PRF', 'TE_GMV_PRF','ET_GMV_PRF']
                idxs = [0, 1, 2, 3]
            else:
                print("For GMV, we can only calculate estimators for argument qe 'all', 'TTEETE', 'TBEB', or 'TTEETEprf'")

            print('Running lensing reconstruction')
            retglm = 0
            retclm = 0

            for i, est in enumerate(ests):
                print('Doing estimator: %s'%est)
                idx = idxs[i]
                alm1 = alm1all[:,idx]
                alm2 = alm2all[:,idx]
                q = weights.weights(est,self.cls[self.cltype],self.lmax,u=u,totalcls=totalcls,crossilc=crossilc)
                glmsum = 0
                clmsum = 0

                if est=='TB_GMV' or est=='EB_GMV':
                    print('WARNING: Currently using a hacky implementation for TB/EB -- should probably revisit!')

                    # TB first!
                    wX1,wY1,wP1,sX1,sY1,sP1 = q.w[0][0],q.w[0][1],q.w[0][2],q.s[0][0],q.s[0][1],q.s[0][2]
                    wX3,wY3,wP3,sX3,sY3,sP3 = q.w[2][0],q.w[2][1],q.w[2][2],q.s[2][0],q.s[2][1],q.s[2][2]

                    walm1 = hp.almxfl(alm1,wX1) # T1
                    walm3 = hp.almxfl(alm1,wX3) # T3
                    walm2 = hp.almxfl(alm2,wY1) # B2

                    SpX1, SmX1 = hp.alm2map_spin([walm1,np.zeros_like(walm1)], self.nside, 1, self.lmax)
                    SpX3, SmX3 = hp.alm2map_spin([walm3,np.zeros_like(walm3)], self.nside, 3, self.lmax)
                    SpY2, SmY2 = hp.alm2map_spin([np.zeros_like(walm2),-1j*walm2], self.nside, 2, self.lmax)

                    SpZ =  SpY2*(SpX1-SpX3) + SmY2*(SmX1-SmX3)
                    SmZ = -SpY2*(SmX1+SmX3) + SmY2*(SpX1+SpX3)

                    glm_TB,clm_TB = hp.map2alm_spin([SpZ,SmZ],1,self.Lmax)

                    nrm = 1

                    glm_TB = hp.almxfl(glm_TB,nrm*wP1)
                    clm_TB = hp.almxfl(clm_TB,nrm*wP1)

                    # EB next!
                    wX1,wY1,wP1,sX1,sY1,sP1 = q.w[4][0],q.w[4][1],q.w[4][2],q.s[4][0],q.s[4][1],q.s[4][2]
                    wX3,wY3,wP3,sX3,sY3,sP3 = q.w[6][0],q.w[6][1],q.w[6][2],q.s[6][0],q.s[6][1],q.s[6][2]

                    walm1 = hp.almxfl(alm1,wX1) # E1
                    walm3 = hp.almxfl(alm1,wX3) # E3
                    walm2 = hp.almxfl(alm2,wY1) # B2

                    SpX1, SmX1 = hp.alm2map_spin([walm1,np.zeros_like(walm1)], self.nside, 1, self.lmax)
                    SpX3, SmX3 = hp.alm2map_spin([walm3,np.zeros_like(walm3)], self.nside, 3, self.lmax)
                    SpY2, SmY2 = hp.alm2map_spin([np.zeros_like(walm2),-1j*walm2], self.nside, 2, self.lmax)

                    SpZ =  SpY2*(SpX1-SpX3) + SmY2*(SmX1-SmX3)
                    SmZ = -SpY2*(SmX1+SmX3) + SmY2*(SpX1+SpX3)

                    glm_EB,clm_EB = hp.map2alm_spin([SpZ,SmZ],1,self.Lmax)

                    nrm = -1

                    glm_EB = hp.almxfl(glm_EB,nrm*wP1)
                    clm_EB = hp.almxfl(clm_EB,nrm*wP1)

                    # Sum
                    glmsum = glm_TB + glm_EB
                    clmsum = clm_TB + clm_EB

                elif est=='BT_GMV' or est=='BE_GMV':
                    print('WARNING: Currently using a hacky implementation for TB/EB -- should probably revisit!')

                    # BT first!
                    wX1,wY1,wP1,sX1,sY1,sP1 = q.w[0][0],q.w[0][1],q.w[0][2],q.s[0][0],q.s[0][1],q.s[0][2]
                    wX3,wY3,wP3,sX3,sY3,sP3 = q.w[2][0],q.w[2][1],q.w[2][2],q.s[2][0],q.s[2][1],q.s[2][2]

                    walm1 = hp.almxfl(alm2,wY1) # T1
                    walm3 = hp.almxfl(alm2,wY3) # T3
                    walm2 = hp.almxfl(alm1,wX1) # B2

                    SpX1, SmX1 = hp.alm2map_spin([walm1,np.zeros_like(walm1)], self.nside, 1, self.lmax)
                    SpX3, SmX3 = hp.alm2map_spin([walm3,np.zeros_like(walm3)], self.nside, 3, self.lmax)
                    SpY2, SmY2 = hp.alm2map_spin([np.zeros_like(walm2),-1j*walm2], self.nside, 2, self.lmax)

                    SpZ =  SpY2*(SpX1-SpX3) + SmY2*(SmX1-SmX3)
                    SmZ = -SpY2*(SmX1+SmX3) + SmY2*(SpX1+SpX3)

                    glm_BT,clm_BT = hp.map2alm_spin([SpZ,SmZ],1,self.Lmax)

                    nrm = 1

                    glm_BT = hp.almxfl(glm_BT,nrm*wP1)
                    clm_BT = hp.almxfl(clm_BT,nrm*wP1)

                    # BE now...
                    wX1,wY1,wP1,sX1,sY1,sP1 = q.w[4][0],q.w[4][1],q.w[4][2],q.s[4][0],q.s[4][1],q.s[4][2]
                    wX3,wY3,wP3,sX3,sY3,sP3 = q.w[6][0],q.w[6][1],q.w[6][2],q.s[6][0],q.s[6][1],q.s[6][2]

                    walm1 = hp.almxfl(alm2,wY1) # E1
                    walm3 = hp.almxfl(alm2,wY3) # E3
                    walm2 = hp.almxfl(alm1,wX1) # B2

                    SpX1, SmX1 = hp.alm2map_spin([walm1,np.zeros_like(walm1)], self.nside, 1, self.lmax)
                    SpX3, SmX3 = hp.alm2map_spin([walm3,np.zeros_like(walm3)], self.nside, 3, self.lmax)
                    SpY2, SmY2 = hp.alm2map_spin([np.zeros_like(walm2),-1j*walm2], self.nside, 2, self.lmax)

                    SpZ =  SpY2*(SpX1-SpX3) + SmY2*(SmX1-SmX3)
                    SmZ = -SpY2*(SmX1+SmX3) + SmY2*(SpX1+SpX3)

                    glm_BE,clm_BE = hp.map2alm_spin([SpZ,SmZ],1,self.Lmax)

                    nrm = -1

                    glm_BE = hp.almxfl(glm_BE,nrm*wP1)
                    clm_BE = hp.almxfl(clm_BE,nrm*wP1)

                    # Sum
                    glmsum = glm_BT + glm_BE
                    clmsum = clm_BT + glm_BE

                else:
                    # More traditional quicklens style calculation
                    for i in range(0,q.ntrm):
                        wX,wY,wP,sX,sY,sP = q.w[i][0],q.w[i][1],q.w[i][2],q.s[i][0],q.s[i][1],q.s[i][2]
                        print("Computing term %d/%d sj = [%d,%d,%d] of est %s"%(i+1,q.ntrm,sX,sY,sP,est))
                        walm1 = hp.almxfl(alm1,wX)
                        walm2 = hp.almxfl(alm2,wY)

                        # Input takes in a^+ and a^-, but in this case we are inserting spin-0 maps i.e. tlm,elm,blm
                        # -----------------------------------------------------------------------------------------------
                        if est[0]=='B':
                            SpX, SmX = hp.alm2map_spin([np.zeros_like(walm1),1j*walm1], self.nside, np.abs(sX), self.lmax)
                        else:
                            SpX, SmX = hp.alm2map_spin([walm1,np.zeros_like(walm1)], self.nside, np.abs(sX), self.lmax)

                        X = SpX+1j*SmX # Complex map _{+s}S or _{-s}S

                        if sX<0:
                            X = np.conj(X)*(-1)**(sX)
                        # -----------------------------------------------------------------------------------------------
                        if est[1]=='B':
                            SpY, SmY = hp.alm2map_spin([np.zeros_like(walm2),1j*walm2], self.nside, np.abs(sY), self.lmax)
                        else:
                            SpY, SmY = hp.alm2map_spin([walm2,np.zeros_like(walm2)], self.nside, np.abs(sY), self.lmax)

                        Y = SpY+1j*SmY

                        if sY<0:
                            Y = np.conj(Y)*(-1)**(sY)
                        # -----------------------------------------------------------------------------------------------

                        XY = X*Y

                        if sP<0:
                            XY = np.conj(XY)*(-1)**(sP)

                        glm,clm  = hp.map2alm_spin([XY.real,XY.imag], np.abs(sP), self.Lmax)

                        glmsum += hp.almxfl(glm,0.5*wP)
                        clmsum += hp.almxfl(clm,0.5*wP)

                if est=='TT_GMV' and crossilc is True:
                    nrm = 0.5
                else:
                    nrm = 1
                retglm += nrm*glmsum
                retclm += nrm*clmsum

            self.glm[qe] = retglm
            self.clm[qe] = retclm

        return self.glm[qe], self.clm[qe]

    def get_aresp(self,qe1=None,qe2=None,u=None,filename=None,crossilc=False):
        '''
        Compute analytical response function

        Parameters
        ----------
        filename: string
          Where to save the aresp output to
        qe1: string
          First estimator; if None, assumes it is self.qe
        qe2: string
          Second estimator; if None, assumes it is the same as qe1

        Returns
        ----------
        aresp:
          Analytical response function
        '''
        r = gmv_resp.gmv_resp(self.config,self.cltype,self.totalcls,u=u,save_path=filename,crossilc=crossilc)
        if qe1 is None:
            qe1 = self.qe

        if (qe1 == 'TTEETE' or qe1 == 'TBEB' or qe1 == 'all') and (qe2 is None or qe2 == qe1):
            # Lensing response
            r.calc_tvar()
        elif qe1 == 'TTEETEprf' and (qe2 is None or qe2 == qe1):
            # Source response
            r.calc_tvar_PRF(cross=False)
        elif (qe1=='TTEETE' and qe2=='TTEETEprf') or (qe2=='TTEETE' and qe1=='TTEETEprf'):
            # Cross estimator response of lensing and source
            r.calc_tvar_PRF(cross=True)
        aresp = np.load(filename)

        # Save file has columns L, TTEETE, TBEB, all
        if qe1 == 'TTEETE' or qe1 == 'TTEETEprf':
            aresp = aresp[:,1]
        elif qe1 == 'TBEB':
            aresp = aresp[:,2]
        elif qe1 == 'all':
            aresp = aresp[:,3]
        return aresp

    def harden(self,qe,alm1all,alm2all,totalcls,u,qe_hrd='TTEETEprf',fn_ss=None,fn_es=None,fn_ee=None):
        '''
        Note: We only harden for qe 'all' and 'TTEETE'.
        Getting the hardened plm for TTEETE and then getting the total hardened plm by
        adding it to the unhardened TBEB is equivalent to
        doing the hardening for all in one step (weight is the same in both cases).

        Parameters
        ----------

        Returns
        ----------
        plm :
          Source hardened glm
        resp :
          Response function
        '''
        assert self.qe=='all' or self.qe=='TTEETE', "We only harden for qe 'all' and 'TTEETE', got: %s"%self.qe

        # ee : Response of est*est
        # es : Cross-estimator response of est*src
        # ss : Response of src*src
        ss = self.get_aresp(qe1=qe_hrd,u=u,filename=fn_ss)
        es = self.get_aresp(qe1=qe_hrd,qe2=qe,u=u,filename=fn_es)
        ee = self.get_aresp(qe1=qe,filename=fn_ee)

        plm1,_ = self.eval(qe,alm1all,alm2all,totalcls)
        plm2,_ = self.eval(qe_hrd,alm1all,alm2all,totalcls,u)

        weight = -1*es/ss
        plm    = plm1 + hp.almxfl(plm2, weight)
        resp   = ee + weight*es

        return plm, resp
