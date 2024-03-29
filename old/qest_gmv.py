import sys
import utils
import weights_gmv
import numpy as np
import healpy as hp
np.seterr(all='ignore')

def qest_gmv(Lmax,clfile,totalcls,alm1all,alm2all,ests='all',unl=False):
    '''
    alms are not filtered.
    alm1all and alm2all should be a N x 5 array for each of the 5 estimators.
    '''
    if ests == 'all':
        ests = ['TT_GMV', 'EE_GMV', 'TE_GMV', 'TB_GMV', 'EB_GMV']
        idxs = [0, 1, 2, 3, 4]
    elif ests == 'A':
        ests = ['TT_GMV', 'EE_GMV', 'TE_GMV']
        idxs = [0, 1, 2]
    elif ests == 'B':
        ests = ['TB_GMV', 'EB_GMV']
        idxs = [3, 4]
    nside   = utils.get_nside(Lmax)
    retglm  = 0
    retclm  = 0

    for i, est in enumerate(ests):
        print('doing estimator: %s'%est)
        idx = idxs[i]
        alm1 = alm1all[:,idx]
        alm2 = alm2all[:,idx]
        print("projecting to nside=%d"%nside)
        lmax1    = hp.Alm.getlmax(alm1.shape[0])
        lmax2    = hp.Alm.getlmax(alm2.shape[0])
        q        = weights_gmv.weights(est,max(lmax1,lmax2),clfile,totalcls,unl)
        print("lmax=%d"%max(lmax1,lmax2))
        print("Lmax=%d"%Lmax)
        glmsum = 0
        clmsum = 0
    
        if est=='TB_GMV' or est=='EB_GMV':
            print('WARNING: Currently using a hacky implementation for TB/EB -- should probably revisit!')
            # TB first!
            wX1,wY1,wP1,sX1,sY1,sP1 = q.w[0][0],q.w[0][1],q.w[0][2],q.s[0][0],q.s[0][1],q.s[0][2]
            wX3,wY3,wP3,sX3,sY3,sP3 = q.w[2][0],q.w[2][1],q.w[2][2],q.s[2][0],q.s[2][1],q.s[2][2]
    
            walm1          = hp.almxfl(alm1,wX1) #T1/E1
            walm3          = hp.almxfl(alm1,wX3) #T3/E3
            walm2          = hp.almxfl(alm2,wY1) #B2
    
            SpX1, SmX1   = hp.alm2map_spin( [walm1,np.zeros_like(walm1)], nside , 1,lmax1)
            SpX3, SmX3   = hp.alm2map_spin( [walm3,np.zeros_like(walm3)], nside , 3,lmax1)
            SpY2, SmY2   = hp.alm2map_spin( [np.zeros_like(walm2),-1j*walm2], nside, 2,lmax2)
    
            SpZ =  SpY2*(SpX1-SpX3) + SmY2*(SmX1-SmX3)
            SmZ = -SpY2*(SmX1+SmX3) + SmY2*(SpX1+SpX3)
    
            glm_TB,clm_TB  = hp.map2alm_spin([SpZ,SmZ],1, Lmax)

            nrm = 1
    
            glm_TB = hp.almxfl(glm_TB,nrm*wP1)
            clm_TB = hp.almxfl(clm_TB,nrm*wP1)

            # EB next!
            wX1,wY1,wP1,sX1,sY1,sP1 = q.w[4][0],q.w[4][1],q.w[4][2],q.s[4][0],q.s[4][1],q.s[4][2]
            wX3,wY3,wP3,sX3,sY3,sP3 = q.w[6][0],q.w[6][1],q.w[6][2],q.s[6][0],q.s[6][1],q.s[6][2]
    
            walm1          = hp.almxfl(alm1,wX1) #T1/E1
            walm3          = hp.almxfl(alm1,wX3) #T3/E3
            walm2          = hp.almxfl(alm2,wY1) #B2
    
            SpX1, SmX1   = hp.alm2map_spin( [walm1,np.zeros_like(walm1)], nside , 1,lmax1)
            SpX3, SmX3   = hp.alm2map_spin( [walm3,np.zeros_like(walm3)], nside , 3,lmax1)
            SpY2, SmY2   = hp.alm2map_spin( [np.zeros_like(walm2),-1j*walm2], nside, 2,lmax2)
    
            SpZ =  SpY2*(SpX1-SpX3) + SmY2*(SmX1-SmX3)
            SmZ = -SpY2*(SmX1+SmX3) + SmY2*(SpX1+SpX3)
    
            glm_EB,clm_EB  = hp.map2alm_spin([SpZ,SmZ],1, Lmax)

            nrm = -1
    
            glm_EB = hp.almxfl(glm_EB,nrm*wP1)
            clm_EB = hp.almxfl(clm_EB,nrm*wP1)

            #glmsum = glm_TB + glm_EB
            #clmsum = clm_TB + clm_EB

            # For the BT and BE for TB_GMV...
            # BT
            wX1,wY1,wP1,sX1,sY1,sP1 = q.w[12][0],q.w[12][1],q.w[12][2],q.s[12][0],q.s[12][1],q.s[12][2]
            wX3,wY3,wP3,sX3,sY3,sP3 = q.w[14][0],q.w[14][1],q.w[14][2],q.s[14][0],q.s[14][1],q.s[14][2]
    
            walm1          = hp.almxfl(alm1,wY1) #T1/E1
            walm3          = hp.almxfl(alm1,wY3) #T3/E3
            walm2          = hp.almxfl(alm2,wX1) #B2
    
            SpX1, SmX1   = hp.alm2map_spin( [walm1,np.zeros_like(walm1)], nside , 1,lmax1)
            SpX3, SmX3   = hp.alm2map_spin( [walm3,np.zeros_like(walm3)], nside , 3,lmax1)
            SpY2, SmY2   = hp.alm2map_spin( [np.zeros_like(walm2),-1j*walm2], nside, 2,lmax2)
    
            SpZ =  SpY2*(SpX1-SpX3) + SmY2*(SmX1-SmX3)
            SmZ = -SpY2*(SmX1+SmX3) + SmY2*(SpX1+SpX3)
    
            glm_BT,clm_BT  = hp.map2alm_spin([SpZ,SmZ],1, Lmax)
            nrm = 1
    
            glm_BT = hp.almxfl(glm_BT,nrm*wP1)
            clm_BT = hp.almxfl(clm_BT,nrm*wP1)

            # BE now
            wX1,wY1,wP1,sX1,sY1,sP1 = q.w[16][0],q.w[16][1],q.w[16][2],q.s[16][0],q.s[16][1],q.s[16][2]
            wX3,wY3,wP3,sX3,sY3,sP3 = q.w[18][0],q.w[18][1],q.w[18][2],q.s[18][0],q.s[18][1],q.s[18][2]
    
            walm1          = hp.almxfl(alm1,wY1) #T1/E1
            walm3          = hp.almxfl(alm1,wY3) #T3/E3
            walm2          = hp.almxfl(alm2,wX1) #B2
    
            SpX1, SmX1   = hp.alm2map_spin( [walm1,np.zeros_like(walm1)], nside , 1,lmax1)
            SpX3, SmX3   = hp.alm2map_spin( [walm3,np.zeros_like(walm3)], nside , 3,lmax1)
            SpY2, SmY2   = hp.alm2map_spin( [np.zeros_like(walm2),-1j*walm2], nside, 2,lmax2)
    
            SpZ =  SpY2*(SpX1-SpX3) + SmY2*(SmX1-SmX3)
            SmZ = -SpY2*(SmX1+SmX3) + SmY2*(SpX1+SpX3)
    
            glm_BE,clm_BE  = hp.map2alm_spin([SpZ,SmZ],1, Lmax)
            nrm = -1
    
            glm_BE = hp.almxfl(glm_BE,nrm*wP1)
            clm_BE = hp.almxfl(clm_BE,nrm*wP1)

            # Sum
            glmsum = glm_TB + glm_EB + glm_BT + glm_BE
            clmsum = clm_TB + clm_EB + clm_BT + glm_BE

        else:
            # More traditional quicklens style calculation            
            for i in range (0,q.ntrm):
                wX,wY,wP,sX,sY,sP = q.w[i][0],q.w[i][1],q.w[i][2],q.s[i][0],q.s[i][1],q.s[i][2]
                print("computing term %d/%d sj=[%d,%d,%d]"%(i+1,q.ntrm,sX,sY,sP))
                walm1          = hp.almxfl(alm1,wX)
                walm2          = hp.almxfl(alm2,wY)
    
                ### input takes in a^+ and a^-, but in this case we are inserting spin-0 maps i.e. tlm,elm,blm
                #-----------------------------------------------------------------------------------------------
                if est[0]=='B':
                    SpX, SmX = hp.alm2map_spin( [np.zeros_like(walm1),1j*walm1], nside, np.abs(sX),lmax1)
                else:
                    SpX, SmX = hp.alm2map_spin( [walm1,np.zeros_like(walm1)], nside, np.abs(sX),lmax1)
                    
                X  = SpX+1j*SmX # Complex map _{+s}S or _{-s}S
                
                if sX<0:
                    X = np.conj(X)*(-1)**(sX)
                #-----------------------------------------------------------------------------------------------
                if est[1]=='B':
                    SpY, SmY = hp.alm2map_spin( [np.zeros_like(walm2),1j*walm2], nside, np.abs(sY),lmax2)
                else:
                    SpY, SmY = hp.alm2map_spin( [walm2,np.zeros_like(walm2)], nside, np.abs(sY),lmax2)
                    
                Y  = SpY+1j*SmY
                
                if sY<0:
                    Y = np.conj(Y)*(-1)**(sY)
                #-----------------------------------------------------------------------------------------------
                
                XY = X*Y
                
                if sP<0:
                    XY = np.conj(XY)*(-1)**(sP)
    
                glm,clm  = hp.map2alm_spin([XY.real,XY.imag], np.abs(sP), Lmax)
    
                glmsum += hp.almxfl(glm,0.5*wP)
                clmsum += hp.almxfl(clm,0.5*wP)

        retglm  += glmsum
        retclm  += clmsum

    return retglm,retclm
