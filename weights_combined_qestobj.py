import numpy as np
import utils 

class weights():
    def __init__(self,est,lmax,config,cltype,u=None,totalcls=None):

        l  = np.arange(lmax+1,dtype=np.float_)
        print('Computing weights')
         
        tdict = {'grad':'gcmb', 'len':'lcmb', 'unl':'ucmb' }
        sl    = {ii:config['cls'][tdict[cltype]][ii] for ii in config['cls'][tdict[cltype]].keys() }

        if totalcls is not None:
            cltt = totalcls[:,0]
            clee = totalcls[:,1]
            clbb = totalcls[:,2]
            clte = totalcls[:,3]

        if est=='TTprf' or est=='TT_GMV_PRF' or est=='EE_GMV_PRF' or est=='TE_GMV_PRF' or est=='TB_GMV_PRF' or est=='EB_GMV_PRF':
            assert u is not None, "must provide u(ell)"
        
        self.lmax=lmax

        if est=='TT_GMV_PRF':
            self.ntrm = 1
            self.w = { i : {} for i in range(0, self.ntrm) }
            self.s = { i : {} for i in range(0, self.ntrm) }
            f1 = u
            f2 = 1/u
            self.w[0][0]=f1*clee[:lmax+1]; self.w[0][1]=f1*clee[:lmax+1]; self.w[0][2]=f2; self.s[0][0]=0; self.s[0][1]=0; self.s[0][2]=0  

        if est=='EE_GMV_PRF':
            self.ntrm = 1
            self.w = { i : {} for i in range(0, self.ntrm) }
            self.s = { i : {} for i in range(0, self.ntrm) }
            f1 = u
            f2 = 1/u
            self.w[0][0]=f1*clte[:lmax+1]; self.w[0][1]=f1*clte[:lmax+1]; self.w[0][2]=f2; self.s[0][0]=0; self.s[0][1]=0; self.s[0][2]=0  

        if est=='TE_GMV_PRF':
            self.ntrm = 2
            self.w = { i : {} for i in range(0, self.ntrm) }
            self.s = { i : {} for i in range(0, self.ntrm) }
            f1 = u
            f2 = 1/u
            self.w[0][0]=f1*clte[:lmax+1]; self.w[0][1]=-1*f1*clee[:lmax+1]; self.w[0][2]=f2; self.s[0][0]=0; self.s[0][1]=0; self.s[0][2]=0  
            self.w[1][1]=f1*clte[:lmax+1]; self.w[1][0]=-1*f1*clee[:lmax+1]; self.w[1][2]=f2; self.s[1][0]=0; self.s[1][1]=0; self.s[1][2]=0  

        if est=='TT_GMV': 
            self.sltt = sl['tt']
            self.slee = sl['ee']
            self.slte = sl['te']
            self.ntrm = 24
            self.w = { i : {} for i in range(0, self.ntrm) }
            self.s = { i : {} for i in range(0, self.ntrm) }
            TT_f1 = -0.5*np.ones_like(l)
            TT_f2 = np.nan_to_num(np.sqrt(l*(l+1)))
            TT_f3 = np.nan_to_num(np.sqrt(l*(l+1)))*self.sltt[:lmax+1]
            EE_f1 = -0.25*np.ones_like(l)
            EE_f2 = +np.nan_to_num(np.sqrt(l*(l+1)))
            EE_f3 = +np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*self.slee[:lmax+1]
            EE_f4 = np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*self.slee[:lmax+1]
            TE_f1 = -0.25*np.ones_like(l,dtype=np.float_)
            TE_f2 =  np.nan_to_num(np.sqrt(l*(l+1)))
            TE_f3 =  np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*self.slte[:lmax+1]
            TE_f4 = +np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*self.slte[:lmax+1]
            TE_f5 = -0.50*np.ones_like(l,dtype=np.float_)
            TE_f6 =  np.nan_to_num(np.sqrt(l*(l+1)))
            TE_f7 =  np.nan_to_num(np.sqrt(l*(l+1)))*self.slte[:lmax+1]
            self.w[0][0]=TT_f3*clee[:lmax+1]; self.w[0][1]=TT_f1*clee[:lmax+1]; self.w[0][2]=TT_f2; self.s[0][0]=+1; self.s[0][1]=+0; self.s[0][2]=+1
            self.w[1][0]=TT_f3*clee[:lmax+1]; self.w[1][1]=TT_f1*clee[:lmax+1]; self.w[1][2]=TT_f2; self.s[1][0]=-1; self.s[1][1]=+0; self.s[1][2]=-1
            self.w[2][0]=TT_f1*clee[:lmax+1]; self.w[2][1]=TT_f3*clee[:lmax+1]; self.w[2][2]=TT_f2; self.s[2][0]=+0; self.s[2][1]=-1; self.s[2][2]=-1
            self.w[3][0]=TT_f1*clee[:lmax+1]; self.w[3][1]=TT_f3*clee[:lmax+1]; self.w[3][2]=TT_f2; self.s[3][0]=+0; self.s[3][1]=+1; self.s[3][2]=+1
            self.w[4][0]=EE_f3*clte[:lmax+1]; self.w[4][1]=EE_f1*clte[:lmax+1]; self.w[4][2]=EE_f2; self.s[4][0]=-1; self.s[4][1]=+2; self.s[4][2]=+1
            self.w[5][0]=EE_f4*clte[:lmax+1]; self.w[5][1]=EE_f1*clte[:lmax+1]; self.w[5][2]=EE_f2; self.s[5][0]=-3; self.s[5][1]=+2; self.s[5][2]=-1
            self.w[6][0]=EE_f4*clte[:lmax+1]; self.w[6][1]=EE_f1*clte[:lmax+1]; self.w[6][2]=EE_f2; self.s[6][0]=+3; self.s[6][1]=-2; self.s[6][2]=+1
            self.w[7][0]=EE_f3*clte[:lmax+1]; self.w[7][1]=EE_f1*clte[:lmax+1]; self.w[7][2]=EE_f2; self.s[7][0]=+1; self.s[7][1]=-2; self.s[7][2]=-1
            self.w[8][0]=EE_f1*clte[:lmax+1]; self.w[8][1]=EE_f3*clte[:lmax+1]; self.w[8][2]=EE_f2; self.s[8][0]=-2; self.s[8][1]=+1; self.s[8][2]=-1
            self.w[9][0]=EE_f1*clte[:lmax+1]; self.w[9][1]=EE_f4*clte[:lmax+1]; self.w[9][2]=EE_f2; self.s[9][0]=-2; self.s[9][1]=+3; self.s[9][2]=+1
            self.w[10][0]=EE_f1*clte[:lmax+1]; self.w[10][1]=EE_f4*clte[:lmax+1]; self.w[10][2]=EE_f2; self.s[10][0]=+2; self.s[10][1]=-3; self.s[10][2]=-1
            self.w[11][0]=EE_f1*clte[:lmax+1]; self.w[11][1]=EE_f3*clte[:lmax+1]; self.w[11][2]=EE_f2; self.s[11][0]=+2; self.s[11][1]=-1; self.s[11][2]=+1
            self.w[12][0]=TE_f3*clee[:lmax+1]; self.w[12][1]=-1*TE_f1*clte[:lmax+1]; self.w[12][2]=TE_f2; self.s[12][0]=-1; self.s[12][1]=+2; self.s[12][2]=+1
            self.w[13][0]=TE_f4*clee[:lmax+1]; self.w[13][1]=-1*TE_f1*clte[:lmax+1]; self.w[13][2]=TE_f2; self.s[13][0]=-3; self.s[13][1]=+2; self.s[13][2]=-1
            self.w[14][0]=TE_f4*clee[:lmax+1]; self.w[14][1]=-1*TE_f1*clte[:lmax+1]; self.w[14][2]=TE_f2; self.s[14][0]=+3; self.s[14][1]=-2; self.s[14][2]=+1
            self.w[15][0]=TE_f3*clee[:lmax+1]; self.w[15][1]=-1*TE_f1*clte[:lmax+1]; self.w[15][2]=TE_f2; self.s[15][0]=+1; self.s[15][1]=-2; self.s[15][2]=-1
            self.w[16][0]=-1*TE_f5*clee[:lmax+1]; self.w[16][1]=TE_f7*clte[:lmax+1]; self.w[16][2]=TE_f6; self.s[16][0]=+0; self.s[16][1]=-1; self.s[16][2]=-1
            self.w[17][0]=-1*TE_f5*clee[:lmax+1]; self.w[17][1]=TE_f7*clte[:lmax+1]; self.w[17][2]=TE_f6; self.s[17][0]=+0; self.s[17][1]=+1; self.s[17][2]=+1
            self.w[18][0]=-1*TE_f1*clte[:lmax+1]; self.w[18][1]=TE_f3*clee[:lmax+1]; self.w[18][2]=TE_f2; self.s[18][0]=+2; self.s[18][1]=-1; self.s[18][2]=+1
            self.w[19][0]=-1*TE_f1*clte[:lmax+1]; self.w[19][1]=TE_f4*clee[:lmax+1]; self.w[19][2]=TE_f2; self.s[19][0]=+2; self.s[19][1]=-3; self.s[19][2]=-1
            self.w[20][0]=-1*TE_f1*clte[:lmax+1]; self.w[20][1]=TE_f4*clee[:lmax+1]; self.w[20][2]=TE_f2; self.s[20][0]=-2; self.s[20][1]=+3; self.s[20][2]=+1
            self.w[21][0]=-1*TE_f1*clte[:lmax+1]; self.w[21][1]=TE_f3*clee[:lmax+1]; self.w[21][2]=TE_f2; self.s[21][0]=-2; self.s[21][1]=+1; self.s[21][2]=-1
            self.w[22][0]=TE_f7*clte[:lmax+1]; self.w[22][1]=-1*TE_f5*clee[:lmax+1]; self.w[22][2]=TE_f6; self.s[22][0]=-1; self.s[22][1]=+0; self.s[22][2]=-1
            self.w[23][0]=TE_f7*clte[:lmax+1]; self.w[23][1]=-1*TE_f5*clee[:lmax+1]; self.w[23][2]=TE_f6; self.s[23][0]=+1; self.s[23][1]=+0; self.s[23][2]=+1

        if est=='EE_GMV':
            self.sltt = sl['tt']
            self.slee = sl['ee']
            self.slte = sl['te']
            self.ntrm = 24
            self.w = { i : {} for i in range(0, self.ntrm) }
            self.s = { i : {} for i in range(0, self.ntrm) }
            TT_f1 = -0.5*np.ones_like(l)
            TT_f2 = np.nan_to_num(np.sqrt(l*(l+1)))
            TT_f3 = np.nan_to_num(np.sqrt(l*(l+1)))*self.sltt[:lmax+1]
            EE_f1 = -0.25*np.ones_like(l)
            EE_f2 = +np.nan_to_num(np.sqrt(l*(l+1)))
            EE_f3 = +np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*self.slee[:lmax+1]
            EE_f4 = np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*self.slee[:lmax+1]
            TE_f1 = -0.25*np.ones_like(l,dtype=np.float_)
            TE_f2 =  np.nan_to_num(np.sqrt(l*(l+1)))
            TE_f3 =  np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*self.slte[:lmax+1]
            TE_f4 = +np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*self.slte[:lmax+1]
            TE_f5 = -0.50*np.ones_like(l,dtype=np.float_)
            TE_f6 =  np.nan_to_num(np.sqrt(l*(l+1)))
            TE_f7 =  np.nan_to_num(np.sqrt(l*(l+1)))*self.slte[:lmax+1]
            self.w[0][0]=TT_f3*clte[:lmax+1]; self.w[0][1]=TT_f1*clte[:lmax+1]; self.w[0][2]=TT_f2; self.s[0][0]=+1; self.s[0][1]=+0; self.s[0][2]=+1
            self.w[1][0]=TT_f3*clte[:lmax+1]; self.w[1][1]=TT_f1*clte[:lmax+1]; self.w[1][2]=TT_f2; self.s[1][0]=-1; self.s[1][1]=+0; self.s[1][2]=-1
            self.w[2][0]=TT_f1*clte[:lmax+1]; self.w[2][1]=TT_f3*clte[:lmax+1]; self.w[2][2]=TT_f2; self.s[2][0]=+0; self.s[2][1]=-1; self.s[2][2]=-1
            self.w[3][0]=TT_f1*clte[:lmax+1]; self.w[3][1]=TT_f3*clte[:lmax+1]; self.w[3][2]=TT_f2; self.s[3][0]=+0; self.s[3][1]=+1; self.s[3][2]=+1
            self.w[4][0]=EE_f3*cltt[:lmax+1]; self.w[4][1]=EE_f1*cltt[:lmax+1]; self.w[4][2]=EE_f2; self.s[4][0]=-1; self.s[4][1]=+2; self.s[4][2]=+1
            self.w[5][0]=EE_f4*cltt[:lmax+1]; self.w[5][1]=EE_f1*cltt[:lmax+1]; self.w[5][2]=EE_f2; self.s[5][0]=-3; self.s[5][1]=+2; self.s[5][2]=-1
            self.w[6][0]=EE_f4*cltt[:lmax+1]; self.w[6][1]=EE_f1*cltt[:lmax+1]; self.w[6][2]=EE_f2; self.s[6][0]=+3; self.s[6][1]=-2; self.s[6][2]=+1
            self.w[7][0]=EE_f3*cltt[:lmax+1]; self.w[7][1]=EE_f1*cltt[:lmax+1]; self.w[7][2]=EE_f2; self.s[7][0]=+1; self.s[7][1]=-2; self.s[7][2]=-1
            self.w[8][0]=EE_f1*cltt[:lmax+1]; self.w[8][1]=EE_f3*cltt[:lmax+1]; self.w[8][2]=EE_f2; self.s[8][0]=-2; self.s[8][1]=+1; self.s[8][2]=-1
            self.w[9][0]=EE_f1*cltt[:lmax+1]; self.w[9][1]=EE_f4*cltt[:lmax+1]; self.w[9][2]=EE_f2; self.s[9][0]=-2; self.s[9][1]=+3; self.s[9][2]=+1
            self.w[10][0]=EE_f1*cltt[:lmax+1]; self.w[10][1]=EE_f4*cltt[:lmax+1]; self.w[10][2]=EE_f2; self.s[10][0]=+2; self.s[10][1]=-3; self.s[10][2]=-1
            self.w[11][0]=EE_f1*cltt[:lmax+1]; self.w[11][1]=EE_f3*cltt[:lmax+1]; self.w[11][2]=EE_f2; self.s[11][0]=+2; self.s[11][1]=-1; self.s[11][2]=+1
            self.w[12][0]=TE_f3*clte[:lmax+1]; self.w[12][1]=-1*TE_f1*cltt[:lmax+1]; self.w[12][2]=TE_f2; self.s[12][0]=-1; self.s[12][1]=+2; self.s[12][2]=+1
            self.w[13][0]=TE_f4*clte[:lmax+1]; self.w[13][1]=-1*TE_f1*cltt[:lmax+1]; self.w[13][2]=TE_f2; self.s[13][0]=-3; self.s[13][1]=+2; self.s[13][2]=-1
            self.w[14][0]=TE_f4*clte[:lmax+1]; self.w[14][1]=-1*TE_f1*cltt[:lmax+1]; self.w[14][2]=TE_f2; self.s[14][0]=+3; self.s[14][1]=-2; self.s[14][2]=+1
            self.w[15][0]=TE_f3*clte[:lmax+1]; self.w[15][1]=-1*TE_f1*cltt[:lmax+1]; self.w[15][2]=TE_f2; self.s[15][0]=+1; self.s[15][1]=-2; self.s[15][2]=-1
            self.w[16][0]=-1*TE_f5*clte[:lmax+1]; self.w[16][1]=TE_f7*cltt[:lmax+1]; self.w[16][2]=TE_f6; self.s[16][0]=+0; self.s[16][1]=-1; self.s[16][2]=-1
            self.w[17][0]=-1*TE_f5*clte[:lmax+1]; self.w[17][1]=TE_f7*cltt[:lmax+1]; self.w[17][2]=TE_f6; self.s[17][0]=+0; self.s[17][1]=+1; self.s[17][2]=+1
            self.w[18][0]=-1*TE_f1*cltt[:lmax+1]; self.w[18][1]=TE_f3*clte[:lmax+1]; self.w[18][2]=TE_f2; self.s[18][0]=+2; self.s[18][1]=-1; self.s[18][2]=+1
            self.w[19][0]=-1*TE_f1*cltt[:lmax+1]; self.w[19][1]=TE_f4*clte[:lmax+1]; self.w[19][2]=TE_f2; self.s[19][0]=+2; self.s[19][1]=-3; self.s[19][2]=-1
            self.w[20][0]=-1*TE_f1*cltt[:lmax+1]; self.w[20][1]=TE_f4*clte[:lmax+1]; self.w[20][2]=TE_f2; self.s[20][0]=-2; self.s[20][1]=+3; self.s[20][2]=+1
            self.w[21][0]=-1*TE_f1*cltt[:lmax+1]; self.w[21][1]=TE_f3*clte[:lmax+1]; self.w[21][2]=TE_f2; self.s[21][0]=-2; self.s[21][1]=+1; self.s[21][2]=-1
            self.w[22][0]=TE_f7*cltt[:lmax+1]; self.w[22][1]=-1*TE_f5*clte[:lmax+1]; self.w[22][2]=TE_f6; self.s[22][0]=-1; self.s[22][1]=+0; self.s[22][2]=-1
            self.w[23][0]=TE_f7*cltt[:lmax+1]; self.w[23][1]=-1*TE_f5*clte[:lmax+1]; self.w[23][2]=TE_f6; self.s[23][0]=+1; self.s[23][1]=+0; self.s[23][2]=+1

        if est=='TE_GMV':
            self.sltt = sl['tt']
            self.slee = sl['ee']
            self.slte = sl['te']
            #self.ntrm = 24
            self.ntrm = 48
            self.w = { i : {} for i in range(0, self.ntrm) }
            self.s = { i : {} for i in range(0, self.ntrm) }
            TT_f1 = -0.5*np.ones_like(l)
            TT_f2 = np.nan_to_num(np.sqrt(l*(l+1)))
            TT_f3 = np.nan_to_num(np.sqrt(l*(l+1)))*self.sltt[:lmax+1]
            EE_f1 = -0.25*np.ones_like(l)
            EE_f2 = +np.nan_to_num(np.sqrt(l*(l+1)))
            EE_f3 = +np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*self.slee[:lmax+1]
            EE_f4 = np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*self.slee[:lmax+1]
            TE_f1 = -0.25*np.ones_like(l,dtype=np.float_)
            TE_f2 =  np.nan_to_num(np.sqrt(l*(l+1)))
            TE_f3 =  np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*self.slte[:lmax+1]
            TE_f4 = +np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*self.slte[:lmax+1]
            TE_f5 = -0.50*np.ones_like(l,dtype=np.float_)
            TE_f6 =  np.nan_to_num(np.sqrt(l*(l+1)))
            TE_f7 =  np.nan_to_num(np.sqrt(l*(l+1)))*self.slte[:lmax+1]
            self.w[0][0]=TT_f3*clee[:lmax+1]; self.w[0][1]=-1*TT_f1*clte[:lmax+1]; self.w[0][2]=TT_f2; self.s[0][0]=+1; self.s[0][1]=+0; self.s[0][2]=+1
            self.w[1][0]=TT_f3*clee[:lmax+1]; self.w[1][1]=-1*TT_f1*clte[:lmax+1]; self.w[1][2]=TT_f2; self.s[1][0]=-1; self.s[1][1]=+0; self.s[1][2]=-1
            self.w[2][0]=-1*TT_f1*clee[:lmax+1]; self.w[2][1]=TT_f3*clte[:lmax+1]; self.w[2][2]=TT_f2; self.s[2][0]=+0; self.s[2][1]=-1; self.s[2][2]=-1
            self.w[3][0]=-1*TT_f1*clee[:lmax+1]; self.w[3][1]=TT_f3*clte[:lmax+1]; self.w[3][2]=TT_f2; self.s[3][0]=+0; self.s[3][1]=+1; self.s[3][2]=+1
            self.w[4][0]=EE_f3*clte[:lmax+1]; self.w[4][1]=-1*EE_f1*cltt[:lmax+1]; self.w[4][2]=EE_f2; self.s[4][0]=-1; self.s[4][1]=+2; self.s[4][2]=+1
            self.w[5][0]=EE_f4*clte[:lmax+1]; self.w[5][1]=-1*EE_f1*cltt[:lmax+1]; self.w[5][2]=EE_f2; self.s[5][0]=-3; self.s[5][1]=+2; self.s[5][2]=-1
            self.w[6][0]=EE_f4*clte[:lmax+1]; self.w[6][1]=-1*EE_f1*cltt[:lmax+1]; self.w[6][2]=EE_f2; self.s[6][0]=+3; self.s[6][1]=-2; self.s[6][2]=+1
            self.w[7][0]=EE_f3*clte[:lmax+1]; self.w[7][1]=-1*EE_f1*cltt[:lmax+1]; self.w[7][2]=EE_f2; self.s[7][0]=+1; self.s[7][1]=-2; self.s[7][2]=-1
            self.w[8][0]=-1*EE_f1*clte[:lmax+1]; self.w[8][1]=EE_f3*cltt[:lmax+1]; self.w[8][2]=EE_f2; self.s[8][0]=-2; self.s[8][1]=+1; self.s[8][2]=-1
            self.w[9][0]=-1*EE_f1*clte[:lmax+1]; self.w[9][1]=EE_f4*cltt[:lmax+1]; self.w[9][2]=EE_f2; self.s[9][0]=-2; self.s[9][1]=+3; self.s[9][2]=+1
            self.w[10][0]=-1*EE_f1*clte[:lmax+1]; self.w[10][1]=EE_f4*cltt[:lmax+1]; self.w[10][2]=EE_f2; self.s[10][0]=+2; self.s[10][1]=-3; self.s[10][2]=-1
            self.w[11][0]=-1*EE_f1*clte[:lmax+1]; self.w[11][1]=EE_f3*cltt[:lmax+1]; self.w[11][2]=EE_f2; self.s[11][0]=+2; self.s[11][1]=-1; self.s[11][2]=+1
            self.w[12][0]=TE_f3*clee[:lmax+1]; self.w[12][1]=TE_f1*cltt[:lmax+1]; self.w[12][2]=TE_f2; self.s[12][0]=-1; self.s[12][1]=+2; self.s[12][2]=+1
            self.w[13][0]=TE_f4*clee[:lmax+1]; self.w[13][1]=TE_f1*cltt[:lmax+1]; self.w[13][2]=TE_f2; self.s[13][0]=-3; self.s[13][1]=+2; self.s[13][2]=-1
            self.w[14][0]=TE_f4*clee[:lmax+1]; self.w[14][1]=TE_f1*cltt[:lmax+1]; self.w[14][2]=TE_f2; self.s[14][0]=+3; self.s[14][1]=-2; self.s[14][2]=+1
            self.w[15][0]=TE_f3*clee[:lmax+1]; self.w[15][1]=TE_f1*cltt[:lmax+1]; self.w[15][2]=TE_f2; self.s[15][0]=+1; self.s[15][1]=-2; self.s[15][2]=-1
            self.w[16][0]=TE_f5*clee[:lmax+1]; self.w[16][1]=TE_f7*cltt[:lmax+1]; self.w[16][2]=TE_f6; self.s[16][0]=+0; self.s[16][1]=-1; self.s[16][2]=-1
            self.w[17][0]=TE_f5*clee[:lmax+1]; self.w[17][1]=TE_f7*cltt[:lmax+1]; self.w[17][2]=TE_f6; self.s[17][0]=+0; self.s[17][1]=+1; self.s[17][2]=+1
            self.w[18][0]=TE_f1*clte[:lmax+1]; self.w[18][1]=TE_f3*clte[:lmax+1]; self.w[18][2]=TE_f2; self.s[18][0]=+2; self.s[18][1]=-1; self.s[18][2]=+1
            self.w[19][0]=TE_f1*clte[:lmax+1]; self.w[19][1]=TE_f4*clte[:lmax+1]; self.w[19][2]=TE_f2; self.s[19][0]=+2; self.s[19][1]=-3; self.s[19][2]=-1
            self.w[20][0]=TE_f1*clte[:lmax+1]; self.w[20][1]=TE_f4*clte[:lmax+1]; self.w[20][2]=TE_f2; self.s[20][0]=-2; self.s[20][1]=+3; self.s[20][2]=+1
            self.w[21][0]=TE_f1*clte[:lmax+1]; self.w[21][1]=TE_f3*clte[:lmax+1]; self.w[21][2]=TE_f2; self.s[21][0]=-2; self.s[21][1]=+1; self.s[21][2]=-1
            self.w[22][0]=TE_f7*clte[:lmax+1]; self.w[22][1]=TE_f5*clte[:lmax+1]; self.w[22][2]=TE_f6; self.s[22][0]=-1; self.s[22][1]=+0; self.s[22][2]=-1
            self.w[23][0]=TE_f7*clte[:lmax+1]; self.w[23][1]=TE_f5*clte[:lmax+1]; self.w[23][2]=TE_f6; self.s[23][0]=+1; self.s[23][1]=+0; self.s[23][2]=+1
            self.w[24][1]=TT_f3*clee[:lmax+1]; self.w[24][0]=-1*TT_f1*clte[:lmax+1]; self.w[24][2]=TT_f2; self.s[24][1]=+1; self.s[24][0]=+0; self.s[24][2]=+1
            self.w[25][1]=TT_f3*clee[:lmax+1]; self.w[25][0]=-1*TT_f1*clte[:lmax+1]; self.w[25][2]=TT_f2; self.s[25][1]=-1; self.s[25][0]=+0; self.s[25][2]=-1
            self.w[26][1]=-1*TT_f1*clee[:lmax+1]; self.w[26][0]=TT_f3*clte[:lmax+1]; self.w[26][2]=TT_f2; self.s[26][1]=+0; self.s[26][0]=-1; self.s[26][2]=-1
            self.w[27][1]=-1*TT_f1*clee[:lmax+1]; self.w[27][0]=TT_f3*clte[:lmax+1]; self.w[27][2]=TT_f2; self.s[27][1]=+0; self.s[27][0]=+1; self.s[27][2]=+1
            self.w[28][1]=EE_f3*clte[:lmax+1]; self.w[28][0]=-1*EE_f1*cltt[:lmax+1]; self.w[28][2]=EE_f2; self.s[28][1]=-1; self.s[28][0]=+2; self.s[28][2]=+1
            self.w[29][1]=EE_f4*clte[:lmax+1]; self.w[29][0]=-1*EE_f1*cltt[:lmax+1]; self.w[29][2]=EE_f2; self.s[29][1]=-3; self.s[29][0]=+2; self.s[29][2]=-1
            self.w[30][1]=EE_f4*clte[:lmax+1]; self.w[30][0]=-1*EE_f1*cltt[:lmax+1]; self.w[30][2]=EE_f2; self.s[30][1]=+3; self.s[30][0]=-2; self.s[30][2]=+1
            self.w[31][1]=EE_f3*clte[:lmax+1]; self.w[31][0]=-1*EE_f1*cltt[:lmax+1]; self.w[31][2]=EE_f2; self.s[31][1]=+1; self.s[31][0]=-2; self.s[31][2]=-1
            self.w[32][1]=-1*EE_f1*clte[:lmax+1]; self.w[32][0]=EE_f3*cltt[:lmax+1]; self.w[32][2]=EE_f2; self.s[32][1]=-2; self.s[32][0]=+1; self.s[32][2]=-1
            self.w[33][1]=-1*EE_f1*clte[:lmax+1]; self.w[33][0]=EE_f4*cltt[:lmax+1]; self.w[33][2]=EE_f2; self.s[33][1]=-2; self.s[33][0]=+3; self.s[33][2]=+1
            self.w[34][1]=-1*EE_f1*clte[:lmax+1]; self.w[34][0]=EE_f4*cltt[:lmax+1]; self.w[34][2]=EE_f2; self.s[34][1]=+2; self.s[34][0]=-3; self.s[34][2]=-1
            self.w[35][1]=-1*EE_f1*clte[:lmax+1]; self.w[35][0]=EE_f3*cltt[:lmax+1]; self.w[35][2]=EE_f2; self.s[35][1]=+2; self.s[35][0]=-1; self.s[35][2]=+1
            self.w[36][1]=TE_f3*clee[:lmax+1]; self.w[36][0]=TE_f1*cltt[:lmax+1]; self.w[36][2]=TE_f2; self.s[36][1]=-1; self.s[36][0]=+2; self.s[36][2]=+1
            self.w[37][1]=TE_f4*clee[:lmax+1]; self.w[37][0]=TE_f1*cltt[:lmax+1]; self.w[37][2]=TE_f2; self.s[37][1]=-3; self.s[37][0]=+2; self.s[37][2]=-1
            self.w[38][1]=TE_f4*clee[:lmax+1]; self.w[38][0]=TE_f1*cltt[:lmax+1]; self.w[38][2]=TE_f2; self.s[38][1]=+3; self.s[38][0]=-2; self.s[38][2]=+1
            self.w[39][1]=TE_f3*clee[:lmax+1]; self.w[39][0]=TE_f1*cltt[:lmax+1]; self.w[39][2]=TE_f2; self.s[39][1]=+1; self.s[39][0]=-2; self.s[39][2]=-1
            self.w[40][1]=TE_f5*clee[:lmax+1]; self.w[40][0]=TE_f7*cltt[:lmax+1]; self.w[40][2]=TE_f6; self.s[40][1]=+0; self.s[40][0]=-1; self.s[40][2]=-1
            self.w[41][1]=TE_f5*clee[:lmax+1]; self.w[41][0]=TE_f7*cltt[:lmax+1]; self.w[41][2]=TE_f6; self.s[41][1]=+0; self.s[41][0]=+1; self.s[41][2]=+1
            self.w[42][1]=TE_f1*clte[:lmax+1]; self.w[42][0]=TE_f3*clte[:lmax+1]; self.w[42][2]=TE_f2; self.s[42][1]=+2; self.s[42][0]=-1; self.s[42][2]=+1
            self.w[43][1]=TE_f1*clte[:lmax+1]; self.w[43][0]=TE_f4*clte[:lmax+1]; self.w[43][2]=TE_f2; self.s[43][1]=+2; self.s[43][0]=-3; self.s[43][2]=-1
            self.w[44][1]=TE_f1*clte[:lmax+1]; self.w[44][0]=TE_f4*clte[:lmax+1]; self.w[44][2]=TE_f2; self.s[44][1]=-2; self.s[44][0]=+3; self.s[44][2]=+1
            self.w[45][1]=TE_f1*clte[:lmax+1]; self.w[45][0]=TE_f3*clte[:lmax+1]; self.w[45][2]=TE_f2; self.s[45][1]=-2; self.s[45][0]=+1; self.s[45][2]=-1
            self.w[46][1]=TE_f7*clte[:lmax+1]; self.w[46][0]=TE_f5*clte[:lmax+1]; self.w[46][2]=TE_f6; self.s[46][1]=-1; self.s[46][0]=+0; self.s[46][2]=-1
            self.w[47][1]=TE_f7*clte[:lmax+1]; self.w[47][0]=TE_f5*clte[:lmax+1]; self.w[47][2]=TE_f6; self.s[47][1]=+1; self.s[47][0]=+0; self.s[47][2]=+1

        if est=='TB_GMV':
            self.slee = sl['ee']
            self.slte = sl['te']
            self.slbb = sl['bb']
            #self.ntrm = 12
            self.ntrm = 24
            self.w = { i : {} for i in range(0, self.ntrm) }
            self.s = { i : {} for i in range(0, self.ntrm) }
            l  = np.arange(lmax+1,dtype=np.float_)
            TB_f1 = -0.25j*np.ones_like(l,dtype=np.float_)
            TB_f2 =  np.nan_to_num(np.sqrt(l*(l+1)))
            TB_f3 =  np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*self.slte[:lmax+1]
            TB_f4 =  np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*self.slte[:lmax+1]
            TB_f5 = +0.25j*np.ones_like(l,dtype=np.float_)
            EB_f1 =  (-0.25j*np.ones_like(l))
            EB_f2 =  (+0.25j*np.ones_like(l))
            EB_f3 = +np.nan_to_num(np.sqrt(l*(l+1)))
            EB_f4 = +np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*self.slee[:lmax+1]
            EB_f5 = +np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*self.slee[:lmax+1]
            EB_f6 = +np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*self.slbb[:lmax+1]
            EB_f7 = +np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*self.slbb[:lmax+1]         
            BT_f1 = -0.25j*np.ones_like(l,dtype=np.float_)
            BT_f2 =  np.nan_to_num(np.sqrt(l*(l+1)))
            BT_f3 =  np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*self.slte[:lmax+1]
            BT_f4 =  np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*self.slte[:lmax+1]
            BT_f5 = +0.25j*np.ones_like(l,dtype=np.float_)
            BE_f1 =  (-0.25j*np.ones_like(l))
            BE_f2 =  (+0.25j*np.ones_like(l))
            BE_f3 = +np.nan_to_num(np.sqrt(l*(l+1)))
            BE_f4 = +np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*self.slee[:lmax+1]
            BE_f5 = +np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*self.slee[:lmax+1]
            BE_f6 = +np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*self.slbb[:lmax+1]
            BE_f7 = +np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*self.slbb[:lmax+1]

            self.w[0][0]=TB_f3*clee[:lmax+1]; self.w[0][1]=TB_f5; self.w[0][2]=TB_f2; self.s[0][0]=-1; self.s[0][1]=+2; self.s[0][2]=+1 # self derived
            self.w[1][0]=TB_f4*clee[:lmax+1]; self.w[1][1]=TB_f5; self.w[1][2]=TB_f2; self.s[1][0]=-3; self.s[1][1]=+2; self.s[1][2]=-1
            self.w[2][0]=TB_f4*clee[:lmax+1]; self.w[2][1]=TB_f1; self.w[2][2]=TB_f2; self.s[2][0]=+3; self.s[2][1]=-2; self.s[2][2]=+1
            self.w[3][0]=TB_f3*clee[:lmax+1]; self.w[3][1]=TB_f1; self.w[3][2]=TB_f2; self.s[3][0]=+1; self.s[3][1]=-2; self.s[3][2]=-1
            # Overleaf version...
            #self.w[0][0]=TB_f3*clee[:lmax+1]; self.w[0][1]=TB_f1; self.w[0][2]=TB_f2; self.s[0][0]=1; self.s[0][1]=-2; self.s[0][2]=+1 # self derived
            #self.w[1][0]=TB_f4*clee[:lmax+1]; self.w[1][1]=TB_f1; self.w[1][2]=TB_f2; self.s[1][0]=3; self.s[1][1]=-2; self.s[1][2]=-1
            #self.w[2][0]=TB_f4*clee[:lmax+1]; self.w[2][1]=TB_f5; self.w[2][2]=TB_f2; self.s[2][0]=-3; self.s[2][1]=2; self.s[2][2]=+1
            #self.w[3][0]=TB_f3*clee[:lmax+1]; self.w[3][1]=TB_f5; self.w[3][2]=TB_f2; self.s[3][0]=-1; self.s[3][1]=2; self.s[3][2]=-1
            
            self.w[4][0]=EB_f4*clte[:lmax+1]; self.w[4][1]=-1*EB_f1; self.w[4][2]=EB_f3; self.s[4][0]=-1; self.s[4][1]=+2; self.s[4][2]=+1
            self.w[5][0]=EB_f5*clte[:lmax+1]; self.w[5][1]=-1*EB_f1; self.w[5][2]=EB_f3; self.s[5][0]=-3; self.s[5][1]=+2; self.s[5][2]=-1
            self.w[6][0]=EB_f5*clte[:lmax+1]; self.w[6][1]=-1*EB_f2; self.w[6][2]=EB_f3; self.s[6][0]=+3; self.s[6][1]=-2; self.s[6][2]=+1
            self.w[7][0]=EB_f4*clte[:lmax+1]; self.w[7][1]=-1*EB_f2; self.w[7][2]=EB_f3; self.s[7][0]=+1; self.s[7][1]=-2; self.s[7][2]=-1
            self.w[8][0]=EB_f6*clte[:lmax+1]; self.w[8][1]=-1*EB_f2; self.w[8][2]=EB_f3; self.s[8][0]=-2; self.s[8][1]=+1; self.s[8][2]=-1
            self.w[9][0]=EB_f7*clte[:lmax+1]; self.w[9][1]=-1*EB_f2; self.w[9][2]=EB_f3; self.s[9][0]=-2; self.s[9][1]=+3; self.s[9][2]=+1
            self.w[10][0]=EB_f7*clte[:lmax+1]; self.w[10][1]=-1*EB_f1; self.w[10][2]=EB_f3; self.s[10][0]=+2; self.s[10][1]=-3; self.s[10][2]=-1
            self.w[11][0]=EB_f6*clte[:lmax+1]; self.w[11][1]=-1*EB_f1; self.w[11][2]=EB_f3; self.s[11][0]=+2; self.s[11][1]=-1; self.s[11][2]=+1
            # Overleaf version...
            #self.w[4][0]=EB_f4*clte[:lmax+1]; self.w[4][1]=-1*EB_f2; self.w[4][2]=EB_f3; self.s[4][0]=-1; self.s[4][1]=+2; self.s[4][2]=+1
            #self.w[5][0]=EB_f5*clte[:lmax+1]; self.w[5][1]=-1*EB_f2; self.w[5][2]=EB_f3; self.s[5][0]=-3; self.s[5][1]=+2; self.s[5][2]=-1
            #self.w[6][0]=EB_f5*clte[:lmax+1]; self.w[6][1]=-1*EB_f1; self.w[6][2]=EB_f3; self.s[6][0]=+3; self.s[6][1]=-2; self.s[6][2]=+1
            #self.w[7][0]=EB_f4*clte[:lmax+1]; self.w[7][1]=-1*EB_f1; self.w[7][2]=EB_f3; self.s[7][0]=+1; self.s[7][1]=-2; self.s[7][2]=-1
            #self.w[8][0]=EB_f1*clte[:lmax+1]; self.w[8][1]=-1*EB_f6; self.w[8][2]=EB_f3; self.s[8][0]=-2; self.s[8][1]=+1; self.s[8][2]=-1
            #self.w[9][0]=EB_f1*clte[:lmax+1]; self.w[9][1]=-1*EB_f7; self.w[9][2]=EB_f3; self.s[9][0]=-2; self.s[9][1]=+3; self.s[9][2]=+1
            #self.w[10][0]=EB_f2*clte[:lmax+1]; self.w[10][1]=-1*EB_f7; self.w[10][2]=EB_f3; self.s[10][0]=+2; self.s[10][1]=-3; self.s[10][2]=-1
            #self.w[11][0]=EB_f2*clte[:lmax+1]; self.w[11][1]=-1*EB_f6; self.w[11][2]=EB_f3; self.s[11][0]=+2; self.s[11][1]=-1; self.s[11][2]=+1

            self.w[12][1]=BT_f3*clee[:lmax+1]; self.w[12][0]=BT_f5; self.w[12][2]=BT_f2; self.s[12][1]=-1; self.s[12][0]=+2; self.s[12][2]=+1 # self derived
            self.w[13][1]=BT_f4*clee[:lmax+1]; self.w[13][0]=BT_f5; self.w[13][2]=BT_f2; self.s[13][1]=-3; self.s[13][0]=+2; self.s[13][2]=-1
            self.w[14][1]=BT_f4*clee[:lmax+1]; self.w[14][0]=BT_f1; self.w[14][2]=BT_f2; self.s[14][1]=+3; self.s[14][0]=-2; self.s[14][2]=+1
            self.w[15][1]=BT_f3*clee[:lmax+1]; self.w[15][0]=BT_f1; self.w[15][2]=BT_f2; self.s[15][1]=+1; self.s[15][0]=-2; self.s[15][2]=-1

            self.w[16][1]=BE_f4*clte[:lmax+1]; self.w[16][0]=-1*BE_f1; self.w[16][2]=BE_f3; self.s[16][1]=-1; self.s[16][0]=+2; self.s[16][2]=+1
            self.w[17][1]=BE_f5*clte[:lmax+1]; self.w[17][0]=-1*BE_f1; self.w[17][2]=BE_f3; self.s[17][1]=-3; self.s[17][0]=+2; self.s[17][2]=-1
            self.w[18][1]=BE_f5*clte[:lmax+1]; self.w[18][0]=-1*BE_f2; self.w[18][2]=BE_f3; self.s[18][1]=+3; self.s[18][0]=-2; self.s[18][2]=+1
            self.w[19][1]=BE_f4*clte[:lmax+1]; self.w[19][0]=-1*BE_f2; self.w[19][2]=BE_f3; self.s[19][1]=+1; self.s[19][0]=-2; self.s[19][2]=-1
            self.w[20][1]=BE_f6*clte[:lmax+1]; self.w[20][0]=-1*BE_f2; self.w[20][2]=BE_f3; self.s[20][1]=-2; self.s[20][0]=+1; self.s[20][2]=-1
            self.w[21][1]=BE_f7*clte[:lmax+1]; self.w[21][0]=-1*BE_f2; self.w[21][2]=BE_f3; self.s[21][1]=-2; self.s[21][0]=+3; self.s[21][2]=+1
            self.w[22][1]=BE_f7*clte[:lmax+1]; self.w[22][0]=-1*BE_f1; self.w[22][2]=BE_f3; self.s[22][1]=+2; self.s[22][0]=-3; self.s[22][2]=-1
            self.w[23][1]=BE_f6*clte[:lmax+1]; self.w[23][0]=-1*BE_f1; self.w[23][2]=BE_f3; self.s[23][1]=+2; self.s[23][0]=-1; self.s[23][2]=+1

        if est=='EB_GMV':
            self.slee = sl['ee']
            self.slte = sl['te']
            self.slbb = sl['bb']
            #self.ntrm = 12
            self.ntrm = 24
            self.w = { i : {} for i in range(0, self.ntrm) }
            self.s = { i : {} for i in range(0, self.ntrm) }
            l  = np.arange(lmax+1,dtype=np.float_)
            TB_f1 = -0.25j*np.ones_like(l,dtype=np.float_)
            TB_f2 =  np.nan_to_num(np.sqrt(l*(l+1)))
            TB_f3 =  np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*self.slte[:lmax+1]
            TB_f4 =  np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*self.slte[:lmax+1]
            TB_f5 = +0.25j*np.ones_like(l,dtype=np.float_)
            EB_f1 =  (-0.25j*np.ones_like(l))
            EB_f2 =  (+0.25j*np.ones_like(l))
            EB_f3 = +np.nan_to_num(np.sqrt(l*(l+1)))
            EB_f4 = +np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*self.slee[:lmax+1]
            EB_f5 = +np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*self.slee[:lmax+1]
            EB_f6 = +np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*self.slbb[:lmax+1]
            EB_f7 = +np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*self.slbb[:lmax+1]         
            BT_f1 = -0.25j*np.ones_like(l,dtype=np.float_)
            BT_f2 =  np.nan_to_num(np.sqrt(l*(l+1)))
            BT_f3 =  np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*self.slte[:lmax+1]
            BT_f4 =  np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*self.slte[:lmax+1]
            BT_f5 = +0.25j*np.ones_like(l,dtype=np.float_)
            BE_f1 =  (-0.25j*np.ones_like(l))
            BE_f2 =  (+0.25j*np.ones_like(l))
            BE_f3 = +np.nan_to_num(np.sqrt(l*(l+1)))
            BE_f4 = +np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*self.slee[:lmax+1]
            BE_f5 = +np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*self.slee[:lmax+1]
            BE_f6 = +np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*self.slbb[:lmax+1]
            BE_f7 = +np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*self.slbb[:lmax+1]

            self.w[0][0]=TB_f3*clte[:lmax+1]; self.w[0][1]=-1*TB_f5; self.w[0][2]=TB_f2; self.s[0][0]=-1; self.s[0][1]=+2; self.s[0][2]=+1 # self derived
            self.w[1][0]=TB_f4*clte[:lmax+1]; self.w[1][1]=-1*TB_f5; self.w[1][2]=TB_f2; self.s[1][0]=-3; self.s[1][1]=+2; self.s[1][2]=-1
            self.w[2][0]=TB_f4*clte[:lmax+1]; self.w[2][1]=-1*TB_f1; self.w[2][2]=TB_f2; self.s[2][0]=+3; self.s[2][1]=-2; self.s[2][2]=+1
            self.w[3][0]=TB_f3*clte[:lmax+1]; self.w[3][1]=-1*TB_f1; self.w[3][2]=TB_f2; self.s[3][0]=+1; self.s[3][1]=-2; self.s[3][2]=-1
            # Overleaf version...
            #self.w[0][0]=TB_f3*clte[:lmax+1]; self.w[0][1]=-1*TB_f1; self.w[0][2]=TB_f2; self.s[0][0]=1; self.s[0][1]=-2; self.s[0][2]=+1 # self derived
            #self.w[1][0]=TB_f4*clte[:lmax+1]; self.w[1][1]=-1*TB_f1; self.w[1][2]=TB_f2; self.s[1][0]=3; self.s[1][1]=-2; self.s[1][2]=-1
            #self.w[2][0]=TB_f4*clte[:lmax+1]; self.w[2][1]=-1*TB_f5; self.w[2][2]=TB_f2; self.s[2][0]=-3; self.s[2][1]=+2; self.s[2][2]=+1
            #self.w[3][0]=TB_f3*clte[:lmax+1]; self.w[3][1]=-1*TB_f5; self.w[3][2]=TB_f2; self.s[3][0]=-1; self.s[3][1]=+2; self.s[3][2]=-1

            self.w[4][0]=EB_f4*cltt[:lmax+1]; self.w[4][1]=EB_f1; self.w[4][2]=EB_f3; self.s[4][0]=-1; self.s[4][1]=+2; self.s[4][2]=+1
            self.w[5][0]=EB_f5*cltt[:lmax+1]; self.w[5][1]=EB_f1; self.w[5][2]=EB_f3; self.s[5][0]=-3; self.s[5][1]=+2; self.s[5][2]=-1
            self.w[6][0]=EB_f5*cltt[:lmax+1]; self.w[6][1]=EB_f2; self.w[6][2]=EB_f3; self.s[6][0]=+3; self.s[6][1]=-2; self.s[6][2]=+1
            self.w[7][0]=EB_f4*cltt[:lmax+1]; self.w[7][1]=EB_f2; self.w[7][2]=EB_f3; self.s[7][0]=+1; self.s[7][1]=-2; self.s[7][2]=-1
            self.w[8][0]=EB_f6*cltt[:lmax+1]; self.w[8][1]=EB_f2; self.w[8][2]=EB_f3; self.s[8][0]=-2; self.s[8][1]=+1; self.s[8][2]=-1
            self.w[9][0]=EB_f7*cltt[:lmax+1]; self.w[9][1]=EB_f2; self.w[9][2]=EB_f3; self.s[9][0]=-2; self.s[9][1]=+3; self.s[9][2]=+1
            self.w[10][0]=EB_f7*cltt[:lmax+1]; self.w[10][1]=EB_f1; self.w[10][2]=EB_f3; self.s[10][0]=+2; self.s[10][1]=-3; self.s[10][2]=-1
            self.w[11][0]=EB_f6*cltt[:lmax+1]; self.w[11][1]=EB_f1; self.w[11][2]=EB_f3; self.s[11][0]=+2; self.s[11][1]=-1; self.s[11][2]=+1
            # Overleaf version...
            #self.w[4][0]=EB_f4*cltt[:lmax+1]; self.w[4][1]=EB_f2; self.w[4][2]=EB_f3; self.s[4][0]=-1; self.s[4][1]=+2; self.s[4][2]=+1
            #self.w[5][0]=EB_f5*cltt[:lmax+1]; self.w[5][1]=EB_f2; self.w[5][2]=EB_f3; self.s[5][0]=-3; self.s[5][1]=+2; self.s[5][2]=-1
            #self.w[6][0]=EB_f5*cltt[:lmax+1]; self.w[6][1]=EB_f1; self.w[6][2]=EB_f3; self.s[6][0]=+3; self.s[6][1]=-2; self.s[6][2]=+1
            #self.w[7][0]=EB_f4*cltt[:lmax+1]; self.w[7][1]=EB_f1; self.w[7][2]=EB_f3; self.s[7][0]=+1; self.s[7][1]=-2; self.s[7][2]=-1
            #self.w[8][0]=EB_f1*cltt[:lmax+1]; self.w[8][1]=EB_f6; self.w[8][2]=EB_f3; self.s[8][0]=-2; self.s[8][1]=+1; self.s[8][2]=-1
            #self.w[9][0]=EB_f1*cltt[:lmax+1]; self.w[9][1]=EB_f7; self.w[9][2]=EB_f3; self.s[9][0]=-2; self.s[9][1]=+3; self.s[9][2]=+1
            #self.w[10][0]=EB_f2*cltt[:lmax+1]; self.w[10][1]=EB_f7; self.w[10][2]=EB_f3; self.s[10][0]=+2; self.s[10][1]=-3; self.s[10][2]=-1
            #self.w[11][0]=EB_f2*cltt[:lmax+1]; self.w[11][1]=EB_f6; self.w[11][2]=EB_f3; self.s[11][0]=+2; self.s[11][1]=-1; self.s[11][2]=+1

            self.w[12][1]=BT_f3*clte[:lmax+1]; self.w[12][0]=-1*BT_f5; self.w[12][2]=BT_f2; self.s[12][1]=-1; self.s[12][0]=+2; self.s[12][2]=+1 # self derived
            self.w[13][1]=BT_f4*clte[:lmax+1]; self.w[13][0]=-1*BT_f5; self.w[13][2]=BT_f2; self.s[13][1]=-3; self.s[13][0]=+2; self.s[13][2]=-1
            self.w[14][1]=BT_f4*clte[:lmax+1]; self.w[14][0]=-1*BT_f1; self.w[14][2]=BT_f2; self.s[14][1]=+3; self.s[14][0]=-2; self.s[14][2]=+1
            self.w[15][1]=BT_f3*clte[:lmax+1]; self.w[15][0]=-1*BT_f1; self.w[15][2]=BT_f2; self.s[15][1]=+1; self.s[15][0]=-2; self.s[15][2]=-1

            self.w[16][1]=BE_f4*cltt[:lmax+1]; self.w[16][0]=BE_f1; self.w[16][2]=BE_f3; self.s[16][1]=-1; self.s[16][0]=+2; self.s[16][2]=+1
            self.w[17][1]=BE_f5*cltt[:lmax+1]; self.w[17][0]=BE_f1; self.w[17][2]=BE_f3; self.s[17][1]=-3; self.s[17][0]=+2; self.s[17][2]=-1
            self.w[18][1]=BE_f5*cltt[:lmax+1]; self.w[18][0]=BE_f2; self.w[18][2]=BE_f3; self.s[18][1]=+3; self.s[18][0]=-2; self.s[18][2]=+1
            self.w[19][1]=BE_f4*cltt[:lmax+1]; self.w[19][0]=BE_f2; self.w[19][2]=BE_f3; self.s[19][1]=+1; self.s[19][0]=-2; self.s[19][2]=-1
            self.w[20][1]=BE_f6*cltt[:lmax+1]; self.w[20][0]=BE_f2; self.w[20][2]=BE_f3; self.s[20][1]=-2; self.s[20][0]=+1; self.s[20][2]=-1
            self.w[21][1]=BE_f7*cltt[:lmax+1]; self.w[21][0]=BE_f2; self.w[21][2]=BE_f3; self.s[21][1]=-2; self.s[21][0]=+3; self.s[21][2]=+1
            self.w[22][1]=BE_f7*cltt[:lmax+1]; self.w[22][0]=BE_f1; self.w[22][2]=BE_f3; self.s[22][1]=+2; self.s[22][0]=-3; self.s[22][2]=-1
            self.w[23][1]=BE_f6*cltt[:lmax+1]; self.w[23][0]=BE_f1; self.w[23][2]=BE_f3; self.s[23][1]=+2; self.s[23][0]=-1; self.s[23][2]=+1

        if est=='TT':
            self.sltt = sl['tt']
            self.ntrm = 4
            self.w = { i : {} for i in range(0, self.ntrm) }
            self.s = { i : {} for i in range(0, self.ntrm) }
            f1 = -0.5*np.ones_like(l)
            f2 = np.nan_to_num(np.sqrt(l*(l+1)))
            f3 = np.nan_to_num(np.sqrt(l*(l+1)))*sl['tt'][:lmax+1]
            self.w[0][0]=f3; self.w[0][1]=f1; self.w[0][2]=f2; self.s[0][0]=+1; self.s[0][1]=+0; self.s[0][2]=+1
            self.w[1][0]=f3; self.w[1][1]=f1; self.w[1][2]=f2; self.s[1][0]=-1; self.s[1][1]=+0; self.s[1][2]=-1
            self.w[2][0]=f1; self.w[2][1]=f3; self.w[2][2]=f2; self.s[2][0]=+0; self.s[2][1]=-1; self.s[2][2]=-1
            self.w[3][0]=f1; self.w[3][1]=f3; self.w[3][2]=f2; self.s[3][0]=+0; self.s[3][1]=+1; self.s[3][2]=+1
        
        if est=='EE':
            self.slee = sl['ee']
            self.ntrm = 8
            self.w = { i : {} for i in range(0, self.ntrm) }
            self.s = { i : {} for i in range(0, self.ntrm) }
            l  = np.arange(lmax+1,dtype=np.float_)
            f1 = -0.25*np.ones_like(l)
            f2 = +np.nan_to_num(np.sqrt(l*(l+1)))
            f3 = +np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*sl['ee'][:lmax+1]
            #f4 = -np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*slee[:lmax+1]
            f4 = np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*sl['ee'][:lmax+1]
            self.w[0][0]=f3; self.w[0][1]=f1; self.w[0][2]=f2; self.s[0][0]=-1; self.s[0][1]=+2; self.s[0][2]=+1
            self.w[1][0]=f4; self.w[1][1]=f1; self.w[1][2]=f2; self.s[1][0]=-3; self.s[1][1]=+2; self.s[1][2]=-1
            self.w[2][0]=f4; self.w[2][1]=f1; self.w[2][2]=f2; self.s[2][0]=+3; self.s[2][1]=-2; self.s[2][2]=+1
            self.w[3][0]=f3; self.w[3][1]=f1; self.w[3][2]=f2; self.s[3][0]=+1; self.s[3][1]=-2; self.s[3][2]=-1
            self.w[4][0]=f1; self.w[4][1]=f3; self.w[4][2]=f2; self.s[4][0]=-2; self.s[4][1]=+1; self.s[4][2]=-1
            self.w[5][0]=f1; self.w[5][1]=f4; self.w[5][2]=f2; self.s[5][0]=-2; self.s[5][1]=+3; self.s[5][2]=+1
            self.w[6][0]=f1; self.w[6][1]=f4; self.w[6][2]=f2; self.s[6][0]=+2; self.s[6][1]=-3; self.s[6][2]=-1
            self.w[7][0]=f1; self.w[7][1]=f3; self.w[7][2]=f2; self.s[7][0]=+2; self.s[7][1]=-1; self.s[7][2]=+1
            
        if est=='TE':
            self.slte = sl['te']
            self.ntrm = 6
            self.w = { i : {} for i in range(0, self.ntrm) }
            self.s = { i : {} for i in range(0, self.ntrm) }
            l  = np.arange(lmax+1,dtype=np.float_)
            f1 = -0.25*np.ones_like(l,dtype=np.float_)
            f2 =  np.nan_to_num(np.sqrt(l*(l+1)))
            f3 =  np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*sl['te'][:lmax+1]
            #f4 = -np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*slte[:lmax+1]
            f4 = +np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*sl['te'][:lmax+1]
            f5 = -0.50*np.ones_like(l,dtype=np.float_)
            f6 =  np.nan_to_num(np.sqrt(l*(l+1)))
            f7 =  np.nan_to_num(np.sqrt(l*(l+1)))*sl['te'][:lmax+1]
            self.w[0][0]=f3; self.w[0][1]=f1; self.w[0][2]=f2; self.s[0][0]=-1; self.s[0][1]=+2; self.s[0][2]=+1
            self.w[1][0]=f4; self.w[1][1]=f1; self.w[1][2]=f2; self.s[1][0]=-3; self.s[1][1]=+2; self.s[1][2]=-1
            self.w[2][0]=f4; self.w[2][1]=f1; self.w[2][2]=f2; self.s[2][0]=+3; self.s[2][1]=-2; self.s[2][2]=+1
            self.w[3][0]=f3; self.w[3][1]=f1; self.w[3][2]=f2; self.s[3][0]=+1; self.s[3][1]=-2; self.s[3][2]=-1
            self.w[4][0]=f5; self.w[4][1]=f7; self.w[4][2]=f6; self.s[4][0]=+0; self.s[4][1]=-1; self.s[4][2]=-1
            self.w[5][0]=f5; self.w[5][1]=f7; self.w[5][2]=f6; self.s[5][0]=+0; self.s[5][1]=+1; self.s[5][2]=+1

        if est=='ET':
            self.slte = sl['te']
            self.ntrm = 6
            self.w = { i : {} for i in range(0, self.ntrm) }
            self.s = { i : {} for i in range(0, self.ntrm) }
            l  = np.arange(lmax+1,dtype=np.float_)
            f1 = -0.25*np.ones_like(l,dtype=np.float_)
            f2 =  np.nan_to_num(np.sqrt(l*(l+1)))
            f3 =  np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*self.slte[:lmax+1]
            #f4 = -np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*self.slte[:lmax+1]
            f4 = +np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*self.slte[:lmax+1]
            f5 = -0.50*np.ones_like(l,dtype=np.float_)
            f6 =  np.nan_to_num(np.sqrt(l*(l+1)))
            f7 =  np.nan_to_num(np.sqrt(l*(l+1)))*self.slte[:lmax+1]
            self.w[0][1]=f3; self.w[0][0]=f1; self.w[0][2]=f2; self.s[0][1]=-1; self.s[0][0]=+2; self.s[0][2]=+1
            self.w[1][1]=f4; self.w[1][0]=f1; self.w[1][2]=f2; self.s[1][1]=-3; self.s[1][0]=+2; self.s[1][2]=-1
            self.w[2][1]=f4; self.w[2][0]=f1; self.w[2][2]=f2; self.s[2][1]=+3; self.s[2][0]=-2; self.s[2][2]=+1
            self.w[3][1]=f3; self.w[3][0]=f1; self.w[3][2]=f2; self.s[3][1]=+1; self.s[3][0]=-2; self.s[3][2]=-1
            self.w[4][1]=f5; self.w[4][0]=f7; self.w[4][2]=f6; self.s[4][1]=+0; self.s[4][0]=-1; self.s[4][2]=-1
            self.w[5][1]=f5; self.w[5][0]=f7; self.w[5][2]=f6; self.s[5][1]=+0; self.s[5][0]=+1; self.s[5][2]=+1

        if est=='TB':
            self.slte = sl['te']
            self.ntrm = 4
            self.w = { i : {} for i in range(0, self.ntrm) }
            self.s = { i : {} for i in range(0, self.ntrm) }
            l  = np.arange(lmax+1,dtype=np.float_)
            f1 = -0.25j*np.ones_like(l,dtype=np.float_)
            f2 =  np.nan_to_num(np.sqrt(l*(l+1)))
            f3 =  np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*sl['te'][:lmax+1]
            f4 =  np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*sl['te'][:lmax+1]
            f5 = +0.25j*np.ones_like(l,dtype=np.float_)
            #self.w[0][0]=f4; self.w[0][1]=f5; self.w[0][2]=f2; self.s[0][0]=+3; self.s[0][1]=-2; self.s[0][2]=+1 #from QL
            #self.w[1][0]=f4; self.w[1][1]=f1; self.w[1][2]=f2; self.s[1][0]=-3; self.s[1][1]=+2; self.s[1][2]=-1
            #self.w[2][0]=f3; self.w[2][1]=f1; self.w[2][2]=f2; self.s[2][0]=-1; self.s[2][1]=+2; self.s[2][2]=+1
            #self.w[3][0]=f3; self.w[3][1]=f5; self.w[3][2]=f2; self.s[3][0]=+1; self.s[3][1]=-2; self.s[3][2]=-1
            self.w[0][0]=f3; self.w[0][1]=f5; self.w[0][2]=f2; self.s[0][0]=-1; self.s[0][1]=+2; self.s[0][2]=+1 # self derived
            self.w[1][0]=f4; self.w[1][1]=f5; self.w[1][2]=f2; self.s[1][0]=-3; self.s[1][1]=+2; self.s[1][2]=-1
            self.w[2][0]=f4; self.w[2][1]=f1; self.w[2][2]=f2; self.s[2][0]=+3; self.s[2][1]=-2; self.s[2][2]=+1
            self.w[3][0]=f3; self.w[3][1]=f1; self.w[3][2]=f2; self.s[3][0]=+1; self.s[3][1]=-2; self.s[3][2]=-1

        if est=='BT':
            self.slte = sl['te']
            self.ntrm = 4
            self.w = { i : {} for i in range(0, self.ntrm) }
            self.s = { i : {} for i in range(0, self.ntrm) }
            l  = np.arange(lmax+1,dtype=np.float_)
            #f1 = +0.25*np.ones_like(l,dtype=np.float_)
            #f2 =  np.nan_to_num(np.sqrt(l*(l+1)))
            #f3 =  np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*sl['te'][:lmax+1]
            #f4 = -np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*sl['te'][:lmax+1]
            #f5 = -0.25*np.ones_like(l,dtype=np.float_)
            #self.w[0][1]=f4; self.w[0][0]=f1; self.w[0][2]=f2; self.s[0][1]=+3; self.s[0][0]=-2; self.s[0][2]=+1
            #self.w[1][1]=f4; self.w[1][0]=f5; self.w[1][2]=f2; self.s[1][1]=-3; self.s[1][0]=+2; self.s[1][2]=-1
            #self.w[2][1]=f3; self.w[2][0]=f5; self.w[2][2]=f2; self.s[2][1]=-1; self.s[2][0]=+2; self.s[2][2]=+1
            #self.w[3][1]=f3; self.w[3][0]=f1; self.w[3][2]=f2; self.s[3][1]=+1; self.s[3][0]=-2; self.s[3][2]=-1
            # Yuuki's weights, 3/1/2023
            f1 = -0.25j*np.ones_like(l,dtype=np.float_)
            f2 =  np.nan_to_num(np.sqrt(l*(l+1)))
            f3 =  np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*self.slte[:lmax+1]
            f4 =  np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*self.slte[:lmax+1]
            f5 = +0.25j*np.ones_like(l,dtype=np.float_)
            #self.w[0][0]=f4; self.w[0][1]=f5; self.w[0][2]=f2; self.s[0][0]=+3; self.s[0][1]=-2; self.s[0][2]=+1 #from QL
            #self.w[1][0]=f4; self.w[1][1]=f1; self.w[1][2]=f2; self.s[1][0]=-3; self.s[1][1]=+2; self.s[1][2]=-1
            #self.w[2][0]=f3; self.w[2][1]=f1; self.w[2][2]=f2; self.s[2][0]=-1; self.s[2][1]=+2; self.s[2][2]=+1
            #self.w[3][0]=f3; self.w[3][1]=f5; self.w[3][2]=f2; self.s[3][0]=+1; self.s[3][1]=-2; self.s[3][2]=-1
            self.w[0][1]=f3; self.w[0][0]=f5; self.w[0][2]=f2; self.s[0][1]=-1; self.s[0][0]=+2; self.s[0][2]=+1 # self derived
            self.w[1][1]=f4; self.w[1][0]=f5; self.w[1][2]=f2; self.s[1][1]=-3; self.s[1][0]=+2; self.s[1][2]=-1
            self.w[2][1]=f4; self.w[2][0]=f1; self.w[2][2]=f2; self.s[2][1]=+3; self.s[2][0]=-2; self.s[2][2]=+1
            self.w[3][1]=f3; self.w[3][0]=f1; self.w[3][2]=f2; self.s[3][1]=+1; self.s[3][0]=-2; self.s[3][2]=-1

        '''
        if est=='EB':
            self.slee = slee
            self.ntrm = 4
            self.w = { i : {} for i in range(0, self.ntrm) }
            self.s = { i : {} for i in range(0, self.ntrm) }
            f1 =  -0.25j*np.ones_like(l)
            f2 =  +0.25j*np.ones_like(l)
            f3 = +np.nan_to_num(np.sqrt(l*(l+1)))
            f4 = +np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*slee[:lmax+1]
            f5 = -np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*slee[:lmax+1]
            self.w[0][0]=f4; self.w[0][1]=f1; self.w[0][2]=f3; self.s[0][0]=+1; self.s[0][1]=-2; self.s[0][2]=-1
            self.w[1][0]=f5; self.w[1][1]=f1; self.w[1][2]=f3; self.s[1][0]=+3; self.s[1][1]=-2; self.s[1][2]=+1
            self.w[2][0]=f5; self.w[2][1]=f2; self.w[2][2]=f3; self.s[2][0]=-3; self.s[2][1]=+2; self.s[2][2]=-1
            self.w[3][0]=f4; self.w[3][1]=f2; self.w[3][2]=f3; self.s[3][0]=-1; self.s[3][1]=+2; self.s[3][2]=+1
        '''
        '''
        if est=='EB':
            self.slee = slee
            self.ntrm = 8
            self.w = { i : {} for i in range(0, self.ntrm) }
            self.s = { i : {} for i in range(0, self.ntrm) }
            f1 =  (+1/(4j)*np.ones_like(l))
            f2 =  (+1/(4j)*np.ones_like(l))
            f3 = +np.nan_to_num(np.sqrt(l*(l+1)))
            f4 = +np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*slee[:lmax+1]
            f5 = -np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*slee[:lmax+1]
            f6 = +np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*slbb[:lmax+1]
            f7 = -np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*slbb[:lmax+1]         
            self.w[0][0]=f4; self.w[0][1]=f1; self.w[0][2]=f3; self.s[0][0]=-1; self.s[0][1]=+2; self.s[0][2]=+1
            self.w[1][0]=f5; self.w[1][1]=f1; self.w[1][2]=f3; self.s[1][0]=-3; self.s[1][1]=+2; self.s[1][2]=-1
            self.w[2][0]=f5; self.w[2][1]=f2; self.w[2][2]=f3; self.s[2][0]=+3; self.s[2][1]=-2; self.s[2][2]=+1
            self.w[3][0]=f4; self.w[3][1]=f2; self.w[3][2]=f3; self.s[3][0]=+1; self.s[3][1]=-2; self.s[3][2]=-1
            self.w[4][0]=f6; self.w[4][1]=f2; self.w[4][2]=f3; self.s[4][0]=-2; self.s[4][1]=+1; self.s[4][2]=-1
            self.w[5][0]=f7; self.w[5][1]=f2; self.w[5][2]=f3; self.s[5][0]=-2; self.s[5][1]=+3; self.s[5][2]=+1
            self.w[6][0]=f7; self.w[6][1]=f1; self.w[6][2]=f3; self.s[6][0]=+2; self.s[6][1]=-3; self.s[6][2]=-1
            self.w[7][0]=f6; self.w[7][1]=f1; self.w[7][2]=f3; self.s[7][0]=+2; self.s[7][1]=-1; self.s[7][2]=+1
        '''

        if est=='EB':
            self.slee = sl['ee']
            self.ntrm = 8
            self.w = { i : {} for i in range(0, self.ntrm) }
            self.s = { i : {} for i in range(0, self.ntrm) }
            f1 =  (-0.25j*np.ones_like(l))
            f2 =  (+0.25j*np.ones_like(l))
            f3 = +np.nan_to_num(np.sqrt(l*(l+1)))
            f4 = +np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*sl['ee'][:lmax+1]
            f5 = +np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*sl['ee'][:lmax+1]
            f6 = +np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*sl['bb'][:lmax+1]
            f7 = +np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*sl['bb'][:lmax+1]         
            self.w[0][0]=f4; self.w[0][1]=f1; self.w[0][2]=f3; self.s[0][0]=-1; self.s[0][1]=+2; self.s[0][2]=+1
            self.w[1][0]=f5; self.w[1][1]=f1; self.w[1][2]=f3; self.s[1][0]=-3; self.s[1][1]=+2; self.s[1][2]=-1
            self.w[2][0]=f5; self.w[2][1]=f2; self.w[2][2]=f3; self.s[2][0]=+3; self.s[2][1]=-2; self.s[2][2]=+1
            self.w[3][0]=f4; self.w[3][1]=f2; self.w[3][2]=f3; self.s[3][0]=+1; self.s[3][1]=-2; self.s[3][2]=-1
            self.w[4][0]=f6; self.w[4][1]=f2; self.w[4][2]=f3; self.s[4][0]=-2; self.s[4][1]=+1; self.s[4][2]=-1
            self.w[5][0]=f7; self.w[5][1]=f2; self.w[5][2]=f3; self.s[5][0]=-2; self.s[5][1]=+3; self.s[5][2]=+1
            self.w[6][0]=f7; self.w[6][1]=f1; self.w[6][2]=f3; self.s[6][0]=+2; self.s[6][1]=-3; self.s[6][2]=-1
            self.w[7][0]=f6; self.w[7][1]=f1; self.w[7][2]=f3; self.s[7][0]=+2; self.s[7][1]=-1; self.s[7][2]=+1

        if est=='BE':
            self.slee = sl['ee']
            self.slbb = sl['bb']
            self.ntrm = 8
            self.w = { i : {} for i in range(0, self.ntrm) }
            self.s = { i : {} for i in range(0, self.ntrm) }
            f1 =  (-0.25j*np.ones_like(l))
            f2 =  (+0.25j*np.ones_like(l))
            f3 = +np.nan_to_num(np.sqrt(l*(l+1)))
            f4 = +np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*self.slee[:lmax+1]
            f5 = +np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*self.slee[:lmax+1]
            f6 = +np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*self.slbb[:lmax+1]
            f7 = +np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*self.slbb[:lmax+1]
            self.w[0][1]=f4; self.w[0][0]=f1; self.w[0][2]=f3; self.s[0][1]=-1; self.s[0][0]=+2; self.s[0][2]=+1
            self.w[1][1]=f5; self.w[1][0]=f1; self.w[1][2]=f3; self.s[1][1]=-3; self.s[1][0]=+2; self.s[1][2]=-1
            self.w[2][1]=f5; self.w[2][0]=f2; self.w[2][2]=f3; self.s[2][1]=+3; self.s[2][0]=-2; self.s[2][2]=+1
            self.w[3][1]=f4; self.w[3][0]=f2; self.w[3][2]=f3; self.s[3][1]=+1; self.s[3][0]=-2; self.s[3][2]=-1
            self.w[4][1]=f6; self.w[4][0]=f2; self.w[4][2]=f3; self.s[4][1]=-2; self.s[4][0]=+1; self.s[4][2]=-1
            self.w[5][1]=f7; self.w[5][0]=f2; self.w[5][2]=f3; self.s[5][1]=-2; self.s[5][0]=+3; self.s[5][2]=+1
            self.w[6][1]=f7; self.w[6][0]=f1; self.w[6][2]=f3; self.s[6][1]=+2; self.s[6][0]=-3; self.s[6][2]=-1
            self.w[7][1]=f6; self.w[7][0]=f1; self.w[7][2]=f3; self.s[7][1]=+2; self.s[7][0]=-1; self.s[7][2]=+1

        if est=="bEP":
            #self.slee = slee
            #self.slpp = uslpp
            self.ntrm = 2 
            self.w = { i : {} for i in range(0, self.ntrm) }
            self.s = { i : {} for i in range(0, self.ntrm) }
            f1 =  (-1*np.ones_like(l))
            f2 =  (+1*np.ones_like(l))
            f3 = +np.nan_to_num(np.sqrt(l*(l+1))) #*uslpp[:lmax+1] #TBD take WF E/phi or invvar E/phi
            f4 = +np.nan_to_num(np.sqrt((l+2.)*(l-1.))) #*slee[:lmax+1]
            f5 = +np.nan_to_num(np.sqrt((l+3.)*(l-2.))) #*slee[:lmax+1]
            self.w[0][0]=f5; self.w[0][1]=f3; self.w[0][2]=f2; self.s[0][0]=+3; self.s[0][1]=-1; self.s[0][2]=+2
            self.w[1][0]=f4; self.w[1][1]=f3; self.w[1][2]=f2; self.s[1][0]=+1; self.s[1][1]=+1; self.s[1][2]=+2

        if est=="srcTT" or est=="TTsrc":
            self.ntrm = 1
            self.w = { i : {} for i in range(0, self.ntrm) }
            self.s = { i : {} for i in range(0, self.ntrm) }
            f1 = np.ones_like(l)
            f2 = 0.5*np.ones_like(l)
            self.w[0][0]=f1; self.w[0][1]=f1; self.w[0][2]=f2; self.s[0][0]=0; self.s[0][1]=0; self.s[0][2]=0  

        if est=="TTprf":
            self.ntrm = 1
            self.w = { i : {} for i in range(0, self.ntrm) }
            self.s = { i : {} for i in range(0, self.ntrm) }
            #f1 = np.ones_like(l)
            f1 = u
            f2 = 1/u
            self.w[0][0]=f1; self.w[0][1]=f1; self.w[0][2]=f2; self.s[0][0]=0; self.s[0][1]=0; self.s[0][2]=0  

'''
def weights_TT(idx,sltt,lmax):
    f1 = -0.5*np.ones_like(l)
    f2 = np.sqrt(l*(l+1))
    f3 = np.sqrt( l*(l+1) )*sltt[:lmax+1]; f3[:3]=0
    if idx==0: w1=f3; w2=f1; wL=f2; s1=+1; s2=+0; sL=+1
    if idx==1: w1=f3; w2=f1; wL=f2; s1=-1; s2=+0; sL=-1
    if idx==2: w1=f3; w2=f1; wL=f2; s1=+0; s2=-1; sL=-1
    if idx==3: w1=f3; w2=f1; wL=f2; s1=+0; s2=+1; sL=+1
    return w1,w2,wL,s1,s2,sL
    
def weights_EE(idx,slee,lmax):
    l  = np.arange(lmax+1,dtype=np.float_)
    f1 = -0.25*np.ones_like(l)
    f2 = +np.sqrt(l*(l+1))
    f3 = +np.sqrt((l+2.)*(l-1.))*slee[:lmax+1]; f3[:3]=0
    f4 = -np.sqrt((l+3.)*(l-2.))*slee[:lmax+1]; f4[:3]=0
    if idx==0: w1=f3; w2=f1; wL=f2; s1=-1; s2=+2; sL=+1
    if idx==1: w1=f4; w2=f1; wL=f2; s1=-3; s2=+2; sL=-1
    if idx==2: w1=f4; w2=f1; wL=f2; s1=+3; s2=-2; sL=+1
    if idx==3: w1=f3; w2=f1; wL=f2; s1=+1; s2=-2; sL=-1
    if idx==4: w1=f1; w2=f3; wL=f2; s1=-2; s2=+1; sL=-1
    if idx==5: w1=f1; w2=f4; wL=f2; s1=-2; s2=+3; sL=+1
    if idx==6: w1=f1; w2=f4; wL=f2; s1=+2; s2=-3; sL=-1
    if idx==7: w1=f1; w2=f3; wL=f2; s1=+2; s2=-1; sL=+1
    return w1,w2,wL,s1,s2,sL

def weights_TE(idx,slte,lmax):
    l  =  np.arange(lmax+1,dtype=np.float_)
    f1 = -0.25*np.ones_like(l)
    f2 =  np.sqrt(l*(l+1))
    f3 =  np.sqrt((l+2.)*(l-1.))*slte[:lmax+1]; f3[:3]=0
    f4 = -np.sqrt((l+3.)*(l-2.))*slte[:lmax+1]; f4[:3]=0
    f5 = -0.5*np.ones_like(l,dtype=np.float_)
    f6 =  np.sqrt(l*(l+1))
    f7 =  np.sqrt(l*(l+1))*slte[:lmax+1]
    if idx==0: w1=f3; w2=f1; wL=f2; s1=-1; s2=+2; sL=+1
    if idx==1: w1=f4; w2=f1; wL=f2; s1=-3; s2=+2; sL=+1
    if idx==2: w1=f4; w2=f1; wL=f2; s1=+3; s2=-2; sL=+1
    if idx==3: w1=f3; w2=f1; wL=f2; s1=+1; s2=-2; sL=-1
    if idx==4: w1=f5; w2=f7; wL=f6; s1=+0; s2=-1; sL=-1
    if idx==5: w1=f5; w2=f7; wL=f6; s1=+0; s2=+1; sL=+1
    return w1,w2,wL,s1,s2,sL
''' 
