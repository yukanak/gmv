import os,sys
import utils
import pickle
import weights_combined_qestobj
import numpy as np
import healpy as hp
np.seterr(all='ignore')

class qest(object):

    def __init__(self,config,qe,almbar1,almbar2,cltype='grad',u=None):
        '''
        Sets up the quatratic estimator calculation 
    
        Parameters
        ----------
        est : str
          Define what the estimator is reconstructing. Should be one of 'lens'/'src'/'prof'.  
        qe  : str
          quadratic estimator type: 'TT'/'EE'/'TE'/'EB'/'TB'
        almbar1: complex
          first filtered alms
        almbar2: complex
          second filtered alms
        config : dictionary of settings
        '''

        #assert est=='lens' or est=='src' or est=='prof', "est expected to be lens/src/prof, got: %s"%est
        assert cltype=='grad' or cltype=='len' or cltype=='unl', "cltype expected to be grad/len/unl, got: %s"%cltype

        #clfile = config['clfile']
        print('Setting up lensing reconstruction')
        print('-- Estimator: %s'%qe)
        self.qe      = qe
        self.almbar1 = almbar1
        self.almbar2 = almbar2
        self.config  = config
        self.retglm  = 0
        self.retclm  = 0
        self.lmax1   = hp.Alm.getlmax(self.almbar1.shape[0])
        self.lmax2   = hp.Alm.getlmax(self.almbar2.shape[0])
        self.Lmax    = self.config['Lmax']
        self.nside   = utils.get_nside(self.Lmax)
        self.cltype  = cltype
        print("-- Nside to project: %d"%self.nside)
        print("-- lmax:%d"%max(self.lmax1,self.lmax2))
        print("-- Lmax:%d"%self.config['Lmax'])
        print("-- cltype: %s"%self.cltype)
        if u is not None:
            print("-- Using profile to harden")
        self.q       = weights_combined_qestobj.weights(qe,max(self.lmax1,self.lmax2),self.config,cltype=self.cltype,u=u)

    def eval(self):
        '''
        Compute equatratic estimator 
    
        Returns
        ----------
        glm: complex
          Gradient component of the plm
        clm: 
          Curl component of the plm

        '''      

        print('Running lensing reconstruction')

        if self.qe=='TB' or self.qe=='EB':
            # hack to get TB/EB working. currently not understanding some factors of j
            print('WARNING: Currently using a hacky implementation for TB/EB -- should probably revisit!')

            wX1,wY1,wP1,sX1,sY1,sP1 = self.q.w[0][0],self.q.w[0][1],self.q.w[0][2],self.q.s[0][0],self.q.s[0][1],self.q.s[0][2]
            wX3,wY3,wP3,sX3,sY3,sP3 = self.q.w[2][0],self.q.w[2][1],self.q.w[2][2],self.q.s[2][0],self.q.s[2][1],self.q.s[2][2]

            walmbar1          = hp.almxfl(self.almbar1,wX1) #T1/E1
            walmbar3          = hp.almxfl(self.almbar1,wX3) #T3/E3
            walmbar2          = hp.almxfl(self.almbar2,wY1) #B2

            SpX1, SmX1   = hp.alm2map_spin( [walmbar1,np.zeros_like(walmbar1)], self.nside , 1,self.lmax1)
            SpX3, SmX3   = hp.alm2map_spin( [walmbar3,np.zeros_like(walmbar3)], self.nside , 3,self.lmax1)
            SpY2, SmY2   = hp.alm2map_spin( [np.zeros_like(walmbar2),-1j*walmbar2], self.nside, 2,self.lmax2)

            SpZ =  SpY2*(SpX1-SpX3) + SmY2*(SmX1-SmX3)
            SmZ = -SpY2*(SmX1+SmX3) + SmY2*(SpX1+SpX3)

            glm,clm  = hp.map2alm_spin([SpZ,SmZ],1, self.Lmax)

            if self.qe=='TT' or self.qe=='EE' or self.qe=='TE' or self.qe=='ET':
                nrm=0.5
            elif self.qe=='EB':
                nrm=-1
            else:
                nrm=1

            glm = hp.almxfl(glm,nrm*wP1)
            clm = hp.almxfl(clm,nrm*wP1)
            return glm,clm

        else:
            # More traditional quicklens style calculation
            
            for i in range(0,self.q.ntrm):
                
                wX,wY,wP,sX,sY,sP = self.q.w[i][0],self.q.w[i][1],self.q.w[i][2],self.q.s[i][0],self.q.s[i][1],self.q.s[i][2]
                print("-- Computing term %d/%d sj=[%d,%d,%d]"%(i+1,self.q.ntrm,sX,sY,sP))
                walmbar1          = hp.almxfl(self.almbar1,wX)
                walmbar2          = hp.almxfl(self.almbar2,wY)

                #print(sP,u[i])

                ### input takes in a^+ and a^-, but in this case we are inserting spin-0 maps i.e. tlm,elm,blm
                #-----------------------------------------------------------------------------------------------
                if self.qe[0]=='B':
                    SpX, SmX = hp.alm2map_spin( [np.zeros_like(walmbar1),1j*walmbar1], self.nside, np.abs(sX),self.lmax1)
                else:
                    SpX, SmX = hp.alm2map_spin( [walmbar1,np.zeros_like(walmbar1)],self.nside, np.abs(sX),self.lmax1)
                    
                X  = SpX+1j*SmX # Complex map _{+s}S or _{-s}S
                
                if sX<0:
                    X = np.conj(X)*(-1)**(sX)
                #-----------------------------------------------------------------------------------------------
                if self.qe[1]=='B':
                    SpY, SmY = hp.alm2map_spin( [np.zeros_like(walmbar2),1j*walmbar2], self.nside, np.abs(sY),self.lmax2)
                else:
                    SpY, SmY = hp.alm2map_spin( [walmbar2,np.zeros_like(walmbar2)], self.nside, np.abs(sY),self.lmax2)
                    
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

                self.retglm  += glm
                self.retclm  += clm
             
            print(" ")
            return self.retglm, self.retclm


    def get_aresp(self,flm1,flm2):
        '''
        Compute analytical response function

        Parameters
        ----------
        flm1 : float, array
          file containing plm1 dictionary. Should have entries 'glm' and 'analytical_resp'
        flm2 : float, array
          file containing plm2 dictionary. Should have entries 'glm' and 'analytical_resp'
        
        Returns
        -------
        resp:
          Analytical response function
        '''

        aresp   = resp.fill_resp(weights_combined_qestobj.weights(qe,dict_lrange['Lmax'],cambcls,u=u_ell),
                                 np.zeros(dict_lrange['Lmax']+1, dtype=np.complex_), flm1, flm2)



    def harden(self,u=None):
        '''cross-estimator response needed for source hardening
        TO DO: make it pass arrays directly

        Parameters
        ----------
        file_plm1 : dict
          file containing plm1 dictionary. Should have entries 'glm' and 'analytical_resp'
        file_plm2 : dict
          file containing plm2 dictionary. Should have entries 'glm' and 'analytical_resp'
        qe1 : str   
          quadratic estimator 
        qe2 : str   
          quadratic estimator 
        cambcls: 
          Cl file produced by camb (fix soon)
        dict_cls:
          Dictionary containing various Cls
        dict_lrange:
          Dictionary containing lcuts
        u : float 
          Array containing profile shape

        Returns
        -------
          source hardened glm and response function
        '''
        tmp        = np.load(file_plm1)
        plm1,resp1 = tmp['glm'], tmp['analytical_resp']

        tmp   = np.load(file_plm2)
        plm2,resp2 = tmp['glm'], tmp['analytical_resp']

        resp12     = resp_xest(qe1,qe2,cambcls,dict_cls,dict_lrange,u=u_ell)

        weight     = -1*resp12 / resp2
        plm        = plm1 + hp.almxfl(plm2, weight)
        resp       = srchard_weighting(resp1,resp12,resp2,weight)
        return plm, resp

class qest_gmv(object):

    def __init__(self,config,qe,alm1all,alm2all,totalcls,cltype='len',u=None):
        '''
        Sets up the quatratic estimator calculation for GMV. 
    
        Parameters
        ----------
        qe  : str
          Quadratic estimator type: 'all'/'TTEETE' (TT/EE/TE only)/'TBEB' (TB/EB only)
        almbar1: complex
          First unfiltered alms; N x 5 arrays for each of the 5 estimators in the order TT/EE/TE/TB/EB
        almbar2: complex
          Second unfiltered alms; N x 5 arrays for each of the 5 estimators in the order TT/EE/TE/TB/EB
        config : dict
          Dictionary of settings
        totalcls:
          The signal + noise spectra for TT, EE, BB, TE needed for the weights
        '''

        assert cltype=='grad' or cltype=='len' or cltype=='unl', "cltype expected to be grad/len/unl, got: %s"%cltype
        if qe == 'all':
            self.ests = ['TT_GMV', 'EE_GMV', 'TE_GMV', 'TB_GMV', 'EB_GMV']
            self.idxs = [0, 1, 2, 3, 4]
        elif qe == 'TTEETE':
            self.ests = ['TT_GMV', 'EE_GMV', 'TE_GMV']
            self.idxs = [0, 1, 2]
        elif qe == 'TBEB':
            self.ests = ['TB_GMV', 'EB_GMV']
            self.idxs = [3, 4]
        else:
            raise Exception("For GMV, argument est must be either 'all', 'A', or 'B'")

        print('Setting up lensing reconstruction')
        print('-- Estimator: %s'%qe)
        self.qe         = qe
        self.alm1all    = alm1all
        self.alm2all    = alm2all
        self.config     = config
        self.retglm     = 0
        self.retclm     = 0
        self.retglm_prf = 0
        self.retclm_prf = 0
        self.lmax1      = hp.Alm.getlmax(self.alm1all[:,0].shape[0])
        self.lmax2      = hp.Alm.getlmax(self.alm2all[:,0].shape[0])
        self.Lmax       = self.config['Lmax']
        self.nside      = utils.get_nside(self.Lmax)
        self.cltype     = cltype
        self.u          = u
        self.totalcls   = totalcls
        print("-- nside to project: %d"%self.nside)
        print("-- lmax:%d"%max(self.lmax1,self.lmax2))
        print("-- Lmax:%d"%self.config['Lmax'])
        print("-- cltype: %s"%self.cltype)
        if u is not None:
            print("-- Using profile to harden")

    def eval(self):
        '''
        Compute equatratic estimator
    
        Returns
        ----------
        glm: complex
          Gradient component of the plm
        clm:
          Curl component of the plm
        ''' 

        print('Running lensing reconstruction')

        for i, est in enumerate(self.ests):
            print('Doing estimator: %s'%est)
            idx = self.idxs[i]
            alm1 = self.alm1all[:,idx]
            alm2 = self.alm2all[:,idx]
            q = weights_combined_qestobj.weights(est,max(self.lmax1,self.lmax2),self.config,cltype=self.cltype,u=self.u,totalcls=self.totalcls)
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
    
                SpX1, SmX1 = hp.alm2map_spin([walm1,np.zeros_like(walm1)], self.nside, 1, self.lmax1)
                SpX3, SmX3 = hp.alm2map_spin([walm3,np.zeros_like(walm3)], self.nside, 3, self.lmax1)
                SpY2, SmY2 = hp.alm2map_spin([np.zeros_like(walm2),-1j*walm2], self.nside, 2, self.lmax2)
    
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
    
                SpX1, SmX1 = hp.alm2map_spin([walm1,np.zeros_like(walm1)], self.nside, 1, self.lmax1)
                SpX3, SmX3 = hp.alm2map_spin([walm3,np.zeros_like(walm3)], self.nside, 3, self.lmax1)
                SpY2, SmY2 = hp.alm2map_spin([np.zeros_like(walm2),-1j*walm2], self.nside, 2, self.lmax2)
    
                SpZ =  SpY2*(SpX1-SpX3) + SmY2*(SmX1-SmX3)
                SmZ = -SpY2*(SmX1+SmX3) + SmY2*(SpX1+SpX3)
    
                glm_EB,clm_EB = hp.map2alm_spin([SpZ,SmZ],1,self.Lmax)

                nrm = -1
    
                glm_EB = hp.almxfl(glm_EB,nrm*wP1)
                clm_EB = hp.almxfl(clm_EB,nrm*wP1)

                # BT now...
                wX1,wY1,wP1,sX1,sY1,sP1 = q.w[12][0],q.w[12][1],q.w[12][2],q.s[12][0],q.s[12][1],q.s[12][2]
                wX3,wY3,wP3,sX3,sY3,sP3 = q.w[14][0],q.w[14][1],q.w[14][2],q.s[14][0],q.s[14][1],q.s[14][2]
    
                walm1 = hp.almxfl(alm1,wY1) # T1
                walm3 = hp.almxfl(alm1,wY3) # T3
                walm2 = hp.almxfl(alm2,wX1) # B2
    
                SpX1, SmX1 = hp.alm2map_spin([walm1,np.zeros_like(walm1)], self.nside, 1, self.lmax1)
                SpX3, SmX3 = hp.alm2map_spin([walm3,np.zeros_like(walm3)], self.nside, 3, self.lmax1)
                SpY2, SmY2 = hp.alm2map_spin([np.zeros_like(walm2),-1j*walm2], self.nside, 2, self.lmax2)
    
                SpZ =  SpY2*(SpX1-SpX3) + SmY2*(SmX1-SmX3)
                SmZ = -SpY2*(SmX1+SmX3) + SmY2*(SpX1+SpX3)
    
                glm_BT,clm_BT = hp.map2alm_spin([SpZ,SmZ],1,self.Lmax)

                nrm = 1
    
                glm_BT = hp.almxfl(glm_BT,nrm*wP1)
                clm_BT = hp.almxfl(clm_BT,nrm*wP1)

                # BE now...
                wX1,wY1,wP1,sX1,sY1,sP1 = q.w[16][0],q.w[16][1],q.w[16][2],q.s[16][0],q.s[16][1],q.s[16][2]
                wX3,wY3,wP3,sX3,sY3,sP3 = q.w[18][0],q.w[18][1],q.w[18][2],q.s[18][0],q.s[18][1],q.s[18][2]
    
                walm1 = hp.almxfl(alm1,wY1) # E1
                walm3 = hp.almxfl(alm1,wY3) # E3
                walm2 = hp.almxfl(alm2,wX1) # B2
    
                SpX1, SmX1 = hp.alm2map_spin([walm1,np.zeros_like(walm1)], self.nside, 1, self.lmax1)
                SpX3, SmX3 = hp.alm2map_spin([walm3,np.zeros_like(walm3)], self.nside, 3, self.lmax1)
                SpY2, SmY2 = hp.alm2map_spin([np.zeros_like(walm2),-1j*walm2], self.nside, 2, self.lmax2)
    
                SpZ =  SpY2*(SpX1-SpX3) + SmY2*(SmX1-SmX3)
                SmZ = -SpY2*(SmX1+SmX3) + SmY2*(SpX1+SpX3)
    
                glm_BE,clm_BE = hp.map2alm_spin([SpZ,SmZ],1,self.Lmax)

                nrm = -1
    
                glm_BE = hp.almxfl(glm_BE,nrm*wP1)
                clm_BE = hp.almxfl(clm_BE,nrm*wP1)

                # Sum
                glmsum = glm_TB + glm_EB + glm_BT + glm_BE
                clmsum = clm_TB + clm_EB + clm_BT + glm_BE

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
                        SpX, SmX = hp.alm2map_spin([np.zeros_like(walm1),1j*walm1], self.nside, np.abs(sX), self.lmax1)
                    else:
                        SpX, SmX = hp.alm2map_spin([walm1,np.zeros_like(walm1)], self.nside, np.abs(sX), self.lmax1)
                        
                    X = SpX+1j*SmX # Complex map _{+s}S or _{-s}S
                    
                    if sX<0:
                        X = np.conj(X)*(-1)**(sX)
                    # -----------------------------------------------------------------------------------------------
                    if est[1]=='B':
                        SpY, SmY = hp.alm2map_spin([np.zeros_like(walm2),1j*walm2], self.nside, np.abs(sY), self.lmax2)
                    else:
                        SpY, SmY = hp.alm2map_spin([walm2,np.zeros_like(walm2)], self.nside, np.abs(sY), self.lmax2)
                        
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

            self.retglm += glmsum
            self.retclm += clmsum

        return self.retglm,self.retclm

    def get_source_estimator(self):
        '''
        Compute the source estimator
        '''
        ests = ['TT_GMV_PRF', 'EE_GMV_PRF', 'TE_GMV_PRF']
        idxs = [0, 1, 2]
        for i, est in enumerate(ests):
            print('Doing estimator: %s'%est)
            idx = idxs[i]
            alm1 = self.alm1all[:,idx]
            alm2 = self.alm2all[:,idx]
            q = weights_combined_qestobj.weights(est,max(self.lmax1,self.lmax2),self.config,cltype=self.cltype,u=self.u,totalcls=self.totalcls)
            glmsum = 0                        
            clmsum = 0

            # More traditional quicklens style calculation            
            for i in range(0,q.ntrm):
                wX,wY,wP,sX,sY,sP = q.w[i][0],q.w[i][1],q.w[i][2],q.s[i][0],q.s[i][1],q.s[i][2]
                print("Computing term %d/%d sj = [%d,%d,%d] of est %s"%(i+1,q.ntrm,sX,sY,sP,est))
                walm1 = hp.almxfl(alm1,wX)
                walm2 = hp.almxfl(alm2,wY)
    
                # Input takes in a^+ and a^-, but in this case we are inserting spin-0 maps i.e. tlm,elm,blm
                # -----------------------------------------------------------------------------------------------
                if est[0]=='B':
                    SpX, SmX = hp.alm2map_spin([np.zeros_like(walm1),1j*walm1], self.nside, np.abs(sX), self.lmax1)
                else:
                    SpX, SmX = hp.alm2map_spin([walm1,np.zeros_like(walm1)], self.nside, np.abs(sX), self.lmax1)
                    
                X = SpX+1j*SmX # Complex map _{+s}S or _{-s}S
                
                if sX<0:
                    X = np.conj(X)*(-1)**(sX)
                # -----------------------------------------------------------------------------------------------
                if est[1]=='B':
                    SpY, SmY = hp.alm2map_spin([np.zeros_like(walm2),1j*walm2], self.nside, np.abs(sY), self.lmax2)
                else:
                    SpY, SmY = hp.alm2map_spin([walm2,np.zeros_like(walm2)], self.nside, np.abs(sY), self.lmax2)
                    
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

            self.retglm_prf += glmsum
            self.retclm_prf += clmsum

        return self.retglm_prf,self.retclm_prf

    def get_aresp(self,flm1,flm2):
        '''
        Compute analytical response function

        Parameters
        ----------
        
        Returns
        -------
        aresp:
          Analytical response function
        '''
        print("Analytic response calculation for GMV under construction... Use Abhi's code")

    def harden(self, ee, es, ss):
        '''
        TODO: This function is thrown together by Yuka, and should be fixed;
        in the future, we should have GMV analytic response in here, so that ee, es, ss
        can be calculated here instead of being passed in

        Parameters
        ----------
        ee : Response of est*est
        es : Cross-estimator response of est*src
        ss : Response of src*src

        Returns
        -------
        plm :
          Source hardened glm
        resp :
          Response function
        '''
        plm1 = self.retglm
        plm2 = self.retglm_prf

        weight = -1*es/ss
        plm    = plm1 + hp.almxfl(plm2, weight)
        resp = ee + 2*weight*es + weight**2*ss

        return plm, resp
