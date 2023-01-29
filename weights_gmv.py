import numpy as np
import utils 

class weights():
    def __init__(self,est,lmax,clfile,totalcls=None,unlclfile=None):
        l  = np.arange(lmax+1,dtype=np.float_)
        ell,sltt,slee,slbb,slte = utils.get_lensedcls(clfile,lmax=lmax)
        if totalcls is not None:
            cltt = totalcls[:,0]
            clee = totalcls[:,1]
            clbb = totalcls[:,2]
            clte = totalcls[:,3]
        if unlclfile is not None:
            uell,usltt,uslee,uslbb,uslte,uslpp,usltp,uslep = utils.get_unlensedcls(unlclfile, lmax=lmax) 
        self.lmax=lmax

        if est=='TT_GMV': 
            self.sltt = sltt
            self.slee = slee
            self.slte = slte
            self.ntrm = 24
            self.w = { i : {} for i in range(0, self.ntrm) }
            self.s = { i : {} for i in range(0, self.ntrm) }
            #TODO: is this right
            TT_f1 = -0.5*np.ones_like(l)
            TT_f2 = np.nan_to_num(np.sqrt(l*(l+1)))
            TT_f3 = np.nan_to_num(np.sqrt(l*(l+1)))*sltt[:lmax+1]
            EE_f1 = -0.25*np.ones_like(l)
            EE_f2 = +np.nan_to_num(np.sqrt(l*(l+1)))
            EE_f3 = +np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*slee[:lmax+1]
            EE_f4 = np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*slee[:lmax+1]
            TE_f1 = -0.25*np.ones_like(l,dtype=np.float_)
            TE_f2 =  np.nan_to_num(np.sqrt(l*(l+1)))
            TE_f3 =  np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*slte[:lmax+1]
            TE_f4 = +np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*slte[:lmax+1]
            TE_f5 = -0.50*np.ones_like(l,dtype=np.float_)
            TE_f6 =  np.nan_to_num(np.sqrt(l*(l+1)))
            TE_f7 =  np.nan_to_num(np.sqrt(l*(l+1)))*slte[:lmax+1]
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
            self.sltt = sltt
            self.slee = slee
            self.slte = slte
            self.ntrm = 24
            self.w = { i : {} for i in range(0, self.ntrm) }
            self.s = { i : {} for i in range(0, self.ntrm) }
            TT_f1 = -0.5*np.ones_like(l)
            TT_f2 = np.nan_to_num(np.sqrt(l*(l+1)))
            TT_f3 = np.nan_to_num(np.sqrt(l*(l+1)))*sltt[:lmax+1]
            EE_f1 = -0.25*np.ones_like(l)
            EE_f2 = +np.nan_to_num(np.sqrt(l*(l+1)))
            EE_f3 = +np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*slee[:lmax+1]
            EE_f4 = np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*slee[:lmax+1]
            TE_f1 = -0.25*np.ones_like(l,dtype=np.float_)
            TE_f2 =  np.nan_to_num(np.sqrt(l*(l+1)))
            TE_f3 =  np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*slte[:lmax+1]
            TE_f4 = +np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*slte[:lmax+1]
            TE_f5 = -0.50*np.ones_like(l,dtype=np.float_)
            TE_f6 =  np.nan_to_num(np.sqrt(l*(l+1)))
            TE_f7 =  np.nan_to_num(np.sqrt(l*(l+1)))*slte[:lmax+1]
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
            self.sltt = sltt
            self.slee = slee
            self.slte = slte
            self.ntrm = 24
            self.w = { i : {} for i in range(0, self.ntrm) }
            self.s = { i : {} for i in range(0, self.ntrm) }
            TT_f1 = -0.5*np.ones_like(l)
            TT_f2 = np.nan_to_num(np.sqrt(l*(l+1)))
            TT_f3 = np.nan_to_num(np.sqrt(l*(l+1)))*sltt[:lmax+1]
            EE_f1 = -0.25*np.ones_like(l)
            EE_f2 = +np.nan_to_num(np.sqrt(l*(l+1)))
            EE_f3 = +np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*slee[:lmax+1]
            EE_f4 = np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*slee[:lmax+1]
            TE_f1 = -0.25*np.ones_like(l,dtype=np.float_)
            TE_f2 =  np.nan_to_num(np.sqrt(l*(l+1)))
            TE_f3 =  np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*slte[:lmax+1]
            TE_f4 = +np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*slte[:lmax+1]
            TE_f5 = -0.50*np.ones_like(l,dtype=np.float_)
            TE_f6 =  np.nan_to_num(np.sqrt(l*(l+1)))
            TE_f7 =  np.nan_to_num(np.sqrt(l*(l+1)))*slte[:lmax+1]
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

        if est=='TB_GMV':
            self.slte = slte
            self.slee = slee
            self.ntrm = 12
            self.w = { i : {} for i in range(0, self.ntrm) }
            self.s = { i : {} for i in range(0, self.ntrm) }
            l  = np.arange(lmax+1,dtype=np.float_)
            TB_f1 = -0.25j*np.ones_like(l,dtype=np.float_)
            TB_f2 =  np.nan_to_num(np.sqrt(l*(l+1)))
            TB_f3 =  np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*slte[:lmax+1]
            TB_f4 =  np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*slte[:lmax+1]
            TB_f5 = +0.25j*np.ones_like(l,dtype=np.float_)
            EB_f1 =  (-0.25j*np.ones_like(l))
            EB_f2 =  (+0.25j*np.ones_like(l))
            EB_f3 = +np.nan_to_num(np.sqrt(l*(l+1)))
            EB_f4 = +np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*slee[:lmax+1]
            EB_f5 = +np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*slee[:lmax+1]
            EB_f6 = +np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*slbb[:lmax+1]
            EB_f7 = +np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*slbb[:lmax+1]         
            self.w[0][0]=TB_f3*clee[:lmax+1]; self.w[0][1]=TB_f5; self.w[0][2]=TB_f2; self.s[0][0]=-1; self.s[0][1]=+2; self.s[0][2]=+1 # self derived
            self.w[1][0]=TB_f4*clee[:lmax+1]; self.w[1][1]=TB_f5; self.w[1][2]=TB_f2; self.s[1][0]=-3; self.s[1][1]=+2; self.s[1][2]=-1
            self.w[2][0]=TB_f4*clee[:lmax+1]; self.w[2][1]=TB_f1; self.w[2][2]=TB_f2; self.s[2][0]=+3; self.s[2][1]=-2; self.s[2][2]=+1
            self.w[3][0]=TB_f3*clee[:lmax+1]; self.w[3][1]=TB_f1; self.w[3][2]=TB_f2; self.s[3][0]=+1; self.s[3][1]=-2; self.s[3][2]=-1
            self.w[4][0]=EB_f4*clte[:lmax+1]; self.w[4][1]=-1*EB_f1; self.w[4][2]=EB_f3; self.s[4][0]=-1; self.s[4][1]=+2; self.s[4][2]=+1
            self.w[5][0]=EB_f5*clte[:lmax+1]; self.w[5][1]=-1*EB_f1; self.w[5][2]=EB_f3; self.s[5][0]=-3; self.s[5][1]=+2; self.s[5][2]=-1
            self.w[6][0]=EB_f5*clte[:lmax+1]; self.w[6][1]=-1*EB_f2; self.w[6][2]=EB_f3; self.s[6][0]=+3; self.s[6][1]=-2; self.s[6][2]=+1
            self.w[7][0]=EB_f4*clte[:lmax+1]; self.w[7][1]=-1*EB_f2; self.w[7][2]=EB_f3; self.s[7][0]=+1; self.s[7][1]=-2; self.s[7][2]=-1
            self.w[8][0]=EB_f6*clte[:lmax+1]; self.w[8][1]=-1*EB_f2; self.w[8][2]=EB_f3; self.s[8][0]=-2; self.s[8][1]=+1; self.s[8][2]=-1
            self.w[9][0]=EB_f7*clte[:lmax+1]; self.w[9][1]=-1*EB_f2; self.w[9][2]=EB_f3; self.s[9][0]=-2; self.s[9][1]=+3; self.s[9][2]=+1
            self.w[10][0]=EB_f7*clte[:lmax+1]; self.w[10][1]=-1*EB_f1; self.w[10][2]=EB_f3; self.s[10][0]=+2; self.s[10][1]=-3; self.s[10][2]=-1
            self.w[11][0]=EB_f6*clte[:lmax+1]; self.w[11][1]=-1*EB_f1; self.w[11][2]=EB_f3; self.s[11][0]=+2; self.s[11][1]=-1; self.s[11][2]=+1

        if est=='EB_GMV':
            self.slte = slte
            self.slee = slee
            self.ntrm = 12
            self.w = { i : {} for i in range(0, self.ntrm) }
            self.s = { i : {} for i in range(0, self.ntrm) }
            l  = np.arange(lmax+1,dtype=np.float_)
            TB_f1 = -0.25j*np.ones_like(l,dtype=np.float_)
            TB_f2 =  np.nan_to_num(np.sqrt(l*(l+1)))
            TB_f3 =  np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*slte[:lmax+1]
            TB_f4 =  np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*slte[:lmax+1]
            TB_f5 = +0.25j*np.ones_like(l,dtype=np.float_)
            EB_f1 =  (-0.25j*np.ones_like(l))
            EB_f2 =  (+0.25j*np.ones_like(l))
            EB_f3 = +np.nan_to_num(np.sqrt(l*(l+1)))
            EB_f4 = +np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*slee[:lmax+1]
            EB_f5 = +np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*slee[:lmax+1]
            EB_f6 = +np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*slbb[:lmax+1]
            EB_f7 = +np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*slbb[:lmax+1]         
            self.w[0][0]=TB_f3*clte[:lmax+1]; self.w[0][1]=-1*TB_f5; self.w[0][2]=TB_f2; self.s[0][0]=-1; self.s[0][1]=+2; self.s[0][2]=+1 # self derived
            self.w[1][0]=TB_f4*clte[:lmax+1]; self.w[1][1]=-1*TB_f5; self.w[1][2]=TB_f2; self.s[1][0]=-3; self.s[1][1]=+2; self.s[1][2]=-1
            self.w[2][0]=TB_f4*clte[:lmax+1]; self.w[2][1]=-1*TB_f1; self.w[2][2]=TB_f2; self.s[2][0]=+3; self.s[2][1]=-2; self.s[2][2]=+1
            self.w[3][0]=TB_f3*clte[:lmax+1]; self.w[3][1]=-1*TB_f1; self.w[3][2]=TB_f2; self.s[3][0]=+1; self.s[3][1]=-2; self.s[3][2]=-1
            self.w[4][0]=EB_f4*cltt[:lmax+1]; self.w[4][1]=EB_f1; self.w[4][2]=EB_f3; self.s[4][0]=-1; self.s[4][1]=+2; self.s[4][2]=+1
            self.w[5][0]=EB_f5*cltt[:lmax+1]; self.w[5][1]=EB_f1; self.w[5][2]=EB_f3; self.s[5][0]=-3; self.s[5][1]=+2; self.s[5][2]=-1
            self.w[6][0]=EB_f5*cltt[:lmax+1]; self.w[6][1]=EB_f2; self.w[6][2]=EB_f3; self.s[6][0]=+3; self.s[6][1]=-2; self.s[6][2]=+1
            self.w[7][0]=EB_f4*cltt[:lmax+1]; self.w[7][1]=EB_f2; self.w[7][2]=EB_f3; self.s[7][0]=+1; self.s[7][1]=-2; self.s[7][2]=-1
            self.w[8][0]=EB_f6*cltt[:lmax+1]; self.w[8][1]=EB_f2; self.w[8][2]=EB_f3; self.s[8][0]=-2; self.s[8][1]=+1; self.s[8][2]=-1
            self.w[9][0]=EB_f7*cltt[:lmax+1]; self.w[9][1]=EB_f2; self.w[9][2]=EB_f3; self.s[9][0]=-2; self.s[9][1]=+3; self.s[9][2]=+1
            self.w[10][0]=EB_f7*cltt[:lmax+1]; self.w[10][1]=EB_f1; self.w[10][2]=EB_f3; self.s[10][0]=+2; self.s[10][1]=-3; self.s[10][2]=-1
            self.w[11][0]=EB_f6*cltt[:lmax+1]; self.w[11][1]=EB_f1; self.w[11][2]=EB_f3; self.s[11][0]=+2; self.s[11][1]=-1; self.s[11][2]=+1

        if est=='TT':
            self.sltt = sltt
            self.ntrm = 4
            self.w = { i : {} for i in range(0, self.ntrm) }
            self.s = { i : {} for i in range(0, self.ntrm) }
            f1 = -0.5*np.ones_like(l)
            f2 = np.nan_to_num(np.sqrt(l*(l+1)))
            f3 = np.nan_to_num(np.sqrt(l*(l+1)))*sltt[:lmax+1]
            self.w[0][0]=f3; self.w[0][1]=f1; self.w[0][2]=f2; self.s[0][0]=+1; self.s[0][1]=+0; self.s[0][2]=+1
            self.w[1][0]=f3; self.w[1][1]=f1; self.w[1][2]=f2; self.s[1][0]=-1; self.s[1][1]=+0; self.s[1][2]=-1
            self.w[2][0]=f1; self.w[2][1]=f3; self.w[2][2]=f2; self.s[2][0]=+0; self.s[2][1]=-1; self.s[2][2]=-1
            self.w[3][0]=f1; self.w[3][1]=f3; self.w[3][2]=f2; self.s[3][0]=+0; self.s[3][1]=+1; self.s[3][2]=+1
        
        if est=='EE':
            self.slee = slee
            self.ntrm = 8
            self.w = { i : {} for i in range(0, self.ntrm) }
            self.s = { i : {} for i in range(0, self.ntrm) }
            l  = np.arange(lmax+1,dtype=np.float_)
            f1 = -0.25*np.ones_like(l)
            f2 = +np.nan_to_num(np.sqrt(l*(l+1)))
            f3 = +np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*slee[:lmax+1]
            #f4 = -np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*slee[:lmax+1]
            f4 = np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*slee[:lmax+1]
            self.w[0][0]=f3; self.w[0][1]=f1; self.w[0][2]=f2; self.s[0][0]=-1; self.s[0][1]=+2; self.s[0][2]=+1
            self.w[1][0]=f4; self.w[1][1]=f1; self.w[1][2]=f2; self.s[1][0]=-3; self.s[1][1]=+2; self.s[1][2]=-1
            self.w[2][0]=f4; self.w[2][1]=f1; self.w[2][2]=f2; self.s[2][0]=+3; self.s[2][1]=-2; self.s[2][2]=+1
            self.w[3][0]=f3; self.w[3][1]=f1; self.w[3][2]=f2; self.s[3][0]=+1; self.s[3][1]=-2; self.s[3][2]=-1
            self.w[4][0]=f1; self.w[4][1]=f3; self.w[4][2]=f2; self.s[4][0]=-2; self.s[4][1]=+1; self.s[4][2]=-1
            self.w[5][0]=f1; self.w[5][1]=f4; self.w[5][2]=f2; self.s[5][0]=-2; self.s[5][1]=+3; self.s[5][2]=+1
            self.w[6][0]=f1; self.w[6][1]=f4; self.w[6][2]=f2; self.s[6][0]=+2; self.s[6][1]=-3; self.s[6][2]=-1
            self.w[7][0]=f1; self.w[7][1]=f3; self.w[7][2]=f2; self.s[7][0]=+2; self.s[7][1]=-1; self.s[7][2]=+1
            
        if est=='TE':
            self.slte = slte
            self.ntrm = 6
            self.w = { i : {} for i in range(0, self.ntrm) }
            self.s = { i : {} for i in range(0, self.ntrm) }
            l  = np.arange(lmax+1,dtype=np.float_)
            f1 = -0.25*np.ones_like(l,dtype=np.float_)
            f2 =  np.nan_to_num(np.sqrt(l*(l+1)))
            f3 =  np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*slte[:lmax+1]
            #f4 = -np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*slte[:lmax+1]
            f4 = +np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*slte[:lmax+1]
            f5 = -0.50*np.ones_like(l,dtype=np.float_)
            f6 =  np.nan_to_num(np.sqrt(l*(l+1)))
            f7 =  np.nan_to_num(np.sqrt(l*(l+1)))*slte[:lmax+1]
            self.w[0][0]=f3; self.w[0][1]=f1; self.w[0][2]=f2; self.s[0][0]=-1; self.s[0][1]=+2; self.s[0][2]=+1
            self.w[1][0]=f4; self.w[1][1]=f1; self.w[1][2]=f2; self.s[1][0]=-3; self.s[1][1]=+2; self.s[1][2]=-1
            self.w[2][0]=f4; self.w[2][1]=f1; self.w[2][2]=f2; self.s[2][0]=+3; self.s[2][1]=-2; self.s[2][2]=+1
            self.w[3][0]=f3; self.w[3][1]=f1; self.w[3][2]=f2; self.s[3][0]=+1; self.s[3][1]=-2; self.s[3][2]=-1
            self.w[4][0]=f5; self.w[4][1]=f7; self.w[4][2]=f6; self.s[4][0]=+0; self.s[4][1]=-1; self.s[4][2]=-1
            self.w[5][0]=f5; self.w[5][1]=f7; self.w[5][2]=f6; self.s[5][0]=+0; self.s[5][1]=+1; self.s[5][2]=+1

        if est=='TB':
            self.slte = slte
            self.ntrm = 4
            self.w = { i : {} for i in range(0, self.ntrm) }
            self.s = { i : {} for i in range(0, self.ntrm) }
            l  = np.arange(lmax+1,dtype=np.float_)
            f1 = -0.25j*np.ones_like(l,dtype=np.float_)
            f2 =  np.nan_to_num(np.sqrt(l*(l+1)))
            f3 =  np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*slte[:lmax+1]
            f4 =  np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*slte[:lmax+1]
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
            self.slte = slte
            self.ntrm = 4
            self.w = { i : {} for i in range(0, self.ntrm) }
            self.s = { i : {} for i in range(0, self.ntrm) }
            l  = np.arange(lmax+1,dtype=np.float_)
            f1 = +0.25*np.ones_like(l,dtype=np.float_)
            f2 =  np.nan_to_num(np.sqrt(l*(l+1)))
            f3 =  np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*slte[:lmax+1]
            f4 = -np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*slte[:lmax+1]
            f5 = -0.25*np.ones_like(l,dtype=np.float_)
            self.w[0][1]=f4; self.w[0][0]=f1; self.w[0][2]=f2; self.s[0][1]=+3; self.s[0][0]=-2; self.s[0][2]=+1
            self.w[1][1]=f4; self.w[1][0]=f5; self.w[1][2]=f2; self.s[1][1]=-3; self.s[1][0]=+2; self.s[1][2]=-1
            self.w[2][1]=f3; self.w[2][0]=f5; self.w[2][2]=f2; self.s[2][1]=-1; self.s[2][0]=+2; self.s[2][2]=+1
            self.w[3][1]=f3; self.w[3][0]=f1; self.w[3][2]=f2; self.s[3][1]=+1; self.s[3][0]=-2; self.s[3][2]=-1
        if est=='EB':
            self.slee = slee
            self.ntrm = 8
            self.w = { i : {} for i in range(0, self.ntrm) }
            self.s = { i : {} for i in range(0, self.ntrm) }
            f1 =  (-0.25j*np.ones_like(l))
            f2 =  (+0.25j*np.ones_like(l))
            f3 = +np.nan_to_num(np.sqrt(l*(l+1)))
            f4 = +np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*slee[:lmax+1]
            f5 = +np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*slee[:lmax+1]
            f6 = +np.nan_to_num(np.sqrt((l+2.)*(l-1.)))*slbb[:lmax+1]
            f7 = +np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*slbb[:lmax+1]         
            self.w[0][0]=f4; self.w[0][1]=f1; self.w[0][2]=f3; self.s[0][0]=-1; self.s[0][1]=+2; self.s[0][2]=+1
            self.w[1][0]=f5; self.w[1][1]=f1; self.w[1][2]=f3; self.s[1][0]=-3; self.s[1][1]=+2; self.s[1][2]=-1
            self.w[2][0]=f5; self.w[2][1]=f2; self.w[2][2]=f3; self.s[2][0]=+3; self.s[2][1]=-2; self.s[2][2]=+1
            self.w[3][0]=f4; self.w[3][1]=f2; self.w[3][2]=f3; self.s[3][0]=+1; self.s[3][1]=-2; self.s[3][2]=-1
            self.w[4][0]=f6; self.w[4][1]=f2; self.w[4][2]=f3; self.s[4][0]=-2; self.s[4][1]=+1; self.s[4][2]=-1
            self.w[5][0]=f7; self.w[5][1]=f2; self.w[5][2]=f3; self.s[5][0]=-2; self.s[5][1]=+3; self.s[5][2]=+1
            self.w[6][0]=f7; self.w[6][1]=f1; self.w[6][2]=f3; self.s[6][0]=+2; self.s[6][1]=-3; self.s[6][2]=-1
            self.w[7][0]=f6; self.w[7][1]=f1; self.w[7][2]=f3; self.s[7][0]=+2; self.s[7][1]=-1; self.s[7][2]=+1

        if est=="bEP":
            #self.slee = slee
            #self.slpp = uslpp
            self.ntrm = 2 
            self.w = { i : {} for i in range(0, self.ntrm) }
            self.s = { i : {} for i in range(0, self.ntrm) }
            f1 =  (-0.5*np.ones_like(l))
            f2 =  (+0.5*np.ones_like(l))
            f3 = +np.nan_to_num(np.sqrt(l*(l+1))) #*uslpp[:lmax+1] #TBD take WF E/phi or invvar E/phi
            f4 = +np.nan_to_num(np.sqrt((l+2.)*(l-1.))) #*slee[:lmax+1]
            f5 = +np.nan_to_num(np.sqrt((l+3.)*(l-2.)))*-1.0 #*slee[:lmax+1]
            self.w[0][0]=f5; self.w[0][1]=f3; self.w[0][2]=f1; self.s[0][0]=+3; self.s[0][1]=-1; self.s[0][2]=+2
            self.w[1][0]=f4; self.w[1][1]=f3; self.w[1][2]=f1; self.s[1][0]=+1; self.s[1][1]=+1; self.s[1][2]=+2

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
''';    
