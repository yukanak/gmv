import healpy as hp
import sys
import matplotlib.pyplot as plt
sys.path.append('/home/users/yukanaka/healqest/healqest/src/')
import healqest_utils as utils

# Mask
# Mask is where the bright sources are, so look at where the mask is 0
#mask = hp.read_map('/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mask8192_mdpl2_v0.7_spt3g_150ghz_lenmag_cibmap_radmap_fluxcut6.0mjy_singlepix.fits')
mask = hp.read_map('/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mask8192_agora_ptsrc_spt3g_6mjy_singlepix.fits')

# Check radio source-only map to make it easier to see
#t95_lrad,q95_lrad,u95_lrad = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_95ghz_lradNG_uk.fits',field=[0,1,2])
t150_lrad,q150_lrad,u150_lrad = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_150ghz_lradNG_uk.fits',field=[0,1,2])
#t220_lrad,q220_lrad,u220_lrad = hp.read_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/mdpl2_spt3g_220ghz_lradNG_uk.fits',field=[0,1,2])
#t95 = t95_lrad*mask; q95 = q95_lrad*mask; u95 = u95_lrad*mask
t150 = t150_lrad*mask; q150 = q150_lrad*mask; u150 = u150_lrad*mask
#t220 = t220_lrad*mask; q220 = q220_lrad*mask; u220 = u220_lrad*mask
#hp.fitsfunc.write_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_95ghz_map_lrad_masked.fits',[t95,q95,u95])
#hp.fitsfunc.write_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_150ghz_map_lrad_masked.fits',[t150,q150,u150])
#hp.fitsfunc.write_map(f'/oak/stanford/orgs/kipac/users/yukanaka/agora_sims_20240904/agora_spt3g_220ghz_map_lrad_masked.fits',[t220,q220,u220])

# Plot maps
config = utils.parse_yaml('test_yuka.yaml')
dir_out = config['dir_out']
scale = 700
plt.figure(0)
plt.clf()
#hp.gnomview(t150,title='Agora Radio Sources-Only 150 GHz T Map, Masked',rot=[0,-60],notext=True,xsize=300,ysize=300,reso=3,min=-1*scale,max=scale,cmap='RdBu_r')#,unit="uK")
hp.gnomview(t150,title='Agora Radio Sources-Only 150 GHz T Map, Masked',rot=[0,-60],notext=True,xsize=300,ysize=300,reso=3,cmap='RdBu_r')#,unit="uK")
plt.savefig(dir_out+f'/figs/agora_t150_lrad_map_masked.png',bbox_inches='tight')

plt.clf()
#hp.gnomview(t150_lrad,title='Agora Radio Sources-Only 150 GHz T Map, Unmasked',rot=[0,-60],notext=True,xsize=300,ysize=300,reso=3,min=-1*scale,max=scale,cmap='RdBu_r')
hp.gnomview(t150_lrad,title='Agora Radio Sources-Only 150 GHz T Map, Unmasked',rot=[0,-60],notext=True,xsize=300,ysize=300,reso=3,cmap='RdBu_r')
plt.savefig(dir_out+f'/figs/agora_t150_lrad_map_unmasked.png',bbox_inches='tight')
