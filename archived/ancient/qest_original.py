import sys
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import utils
import weights_gmv
import numpy as np
import healpy as hp
np.seterr(all='ignore')

def qest(est,Lmax,clfile,almbar1,almbar2,unl=False):
    print('estimator used: %s'%est)
    retglm  = 0
    retclm  = 0
    nside    = utils.get_nside(Lmax)
    print("projecting to nside=%d"%nside)
    lmax1    = hp.Alm.getlmax(almbar1.shape[0])
    lmax2    = hp.Alm.getlmax(almbar2.shape[0])
    q        = weights_gmv.weights(est,max(lmax1,lmax2),clfile,unl=unl)
    print("lmax=%d"%max(lmax1,lmax2))
    print("Lmax=%d"%Lmax)

    if est=='TB' or est=='EB':
        # hack to get TB/EB working. currently not understanding some factors of j
        print('WARNING: Currently using a hacky implementation for TB/EB -- should probably revisit!')
        q        = weights_gmv.weights(est,max(lmax1,lmax2),clfile,unl=unl)

        wX1,wY1,wP1,sX1,sY1,sP1 = q.w[0][0],q.w[0][1],q.w[0][2],q.s[0][0],q.s[0][1],q.s[0][2]
        wX3,wY3,wP3,sX3,sY3,sP3 = q.w[2][0],q.w[2][1],q.w[2][2],q.s[2][0],q.s[2][1],q.s[2][2]

        walmbar1          = hp.almxfl(almbar1,wX1) #T1/E1
        walmbar3          = hp.almxfl(almbar1,wX3) #T3/E3
        walmbar2          = hp.almxfl(almbar2,wY1) #B2

        SpX1, SmX1   = hp.alm2map_spin( [walmbar1,np.zeros_like(walmbar1)], nside , 1,lmax1)
        SpX3, SmX3   = hp.alm2map_spin( [walmbar3,np.zeros_like(walmbar3)], nside , 3,lmax1)
        SpY2, SmY2   = hp.alm2map_spin( [np.zeros_like(walmbar2),-1j*walmbar2], nside, 2,lmax2)

        SpZ =  SpY2*(SpX1-SpX3) + SmY2*(SmX1-SmX3)
        SmZ = -SpY2*(SmX1+SmX3) + SmY2*(SpX1+SpX3)

        glm,clm  = hp.map2alm_spin([SpZ,SmZ],1, Lmax)

        if est=='TT' or est=='EE' or est=='TE' or est=='ET':
            nrm=0.5
        elif est=='EB':
            nrm=-1
        else:
            nrm=1

        glm = hp.almxfl(glm,nrm*wP1)
        clm = hp.almxfl(clm,nrm*wP1)
        return glm,clm

    else:
        # More traditional quicklens style calculation
        
        for i in range (0,q.ntrm):
            
            wX,wY,wP,sX,sY,sP = q.w[i][0],q.w[i][1],q.w[i][2],q.s[i][0],q.s[i][1],q.s[i][2]
            print("computing term %d/%d sj=[%d,%d,%d]"%(i+1,q.ntrm,sX,sY,sP))
            walmbar1          = hp.almxfl(almbar1,wX)
            walmbar2          = hp.almxfl(almbar2,wY)

            #print(sP,u[i])

            ### input takes in a^+ and a^-, but in this case we are inserting spin-0 maps i.e. tlm,elm,blm
            #-----------------------------------------------------------------------------------------------
            if est[0]=='B':
                SpX, SmX = hp.alm2map_spin( [np.zeros_like(walmbar1),1j*walmbar1], nside, np.abs(sX),lmax1)
            else:
                SpX, SmX = hp.alm2map_spin( [walmbar1,np.zeros_like(walmbar1)], nside, np.abs(sX),lmax1)
                
            X  = SpX+1j*SmX # Complex map _{+s}S or _{-s}S
            
            if sX<0:
                X = np.conj(X)*(-1)**(sX)
            #-----------------------------------------------------------------------------------------------
            if est[1]=='B':
                SpY, SmY = hp.alm2map_spin( [np.zeros_like(walmbar2),1j*walmbar2], nside, np.abs(sY),lmax2)
            else:
                SpY, SmY = hp.alm2map_spin( [walmbar2,np.zeros_like(walmbar2)], nside, np.abs(sY),lmax2)
                
            Y  = SpY+1j*SmY
            
            if sY<0:
                Y = np.conj(Y)*(-1)**(sY)
            #-----------------------------------------------------------------------------------------------
            
            XY = X*Y
            
            if sP<0:
                XY = np.conj(XY)*(-1)**(sP)

            glm,clm  = hp.map2alm_spin([XY.real,XY.imag], np.abs(sP), Lmax)
            
                

            glm = hp.almxfl(glm,0.5*wP)
            clm = hp.almxfl(clm,0.5*wP)

            retglm  += glm
            retclm  += clm

        return retglm,retclm
