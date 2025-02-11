from scipy import integrate
import numpy as np
from scipy import interpolate
import os 
import pandas as pan
from astropy.io import fits
import bin_range

from cosmology import *

a=bin_range.zmin
b=bin_range.zmax


try:
    script_dir= os.path.dirname(os.path.abspath(__file__))                     # Current directory 
except NameError:
    script_dir=os.getcwd()

############################
############################
#
#  Chi^2 funcions
#
############################
############################


############################
############################
#
# Chi square for CC's data
#
############################
############################

# Import the data 

CC=np.loadtxt(script_dir+"/Data/CC/CC.dat")                # CC data
cov_matCC=np.load(script_dir+"/Data/CC/cov_matCC.npy")     # Covariance matrix 
z_CC=np.array(CC[:,0])                                     # Redshifts  
H_obser=np.array(CC[:,1])                                  # H(z) data

mask_CC=(z_CC<=b) & (z_CC>a)                               # Create a mask according to the bin

# Bin the data using the mask

zbin_CC=z_CC[mask_CC]
Hbin_obser=H_obser[mask_CC]
covbin_CC=cov_matCC[mask_CC, :][:, mask_CC]                  # Reconstruct the covariance matrix according to the binned data                     
inv_covCC=np.linalg.inv(covbin_CC)                           # Inverse of the covariance matrix 


# Theoretical function for H(z)
def H_theo(Om0,H0,w0,wa): 
    return H(zbin_CC,Om0,H0,w0,wa)  


# Construct the chi^2 for CC
def chi2_CC(Om0,H0,w0,wa):
    chi2=0
    delta=H_theo(Om0,H0,w0,wa)-Hbin_obser
    chi2=np.dot(delta,np.dot(inv_covCC,delta))
    return chi2

############################
############################
#
# Chi square for SNe Ia data - Union3 
#
############################
############################

# Import the data 

file_Union3=fits.open(script_dir+"/Data/union3_release/mu_mat_union3_cosmo=2_mu.fits")
data_Union3=file_Union3[0].data
zcmb_Union3=data_Union3[0,1:]                                                              # Redshift corrected for CMB
zhel_Union3=data_Union3[0,1:]                                                              # Heliocentric redshift 
mb_Union3=data_Union3[1:,0]                                                                # data of the apparent magnitude mb
cov_matUnion3 = np.linalg.inv(data_Union3[1:,1:])                                          # Covariance matrix

mask_Union3=(zcmb_Union3<=b) & (zcmb_Union3>a)                   # mask to separate data

# Bin the data using the mask

zcmb_bin_Union3=zcmb_Union3[mask_Union3]
zhel_bin_Union3=zhel_Union3[mask_Union3]
mb_bin_Union3=mb_Union3[mask_Union3]
covbin_mat_Union3=cov_matUnion3[mask_Union3, :][:, mask_Union3]  # Reconstruct the covariance matrix according to the binned data
inv_cov_Union3=np.linalg.inv(covbin_mat_Union3)                  # Inverse of the covariance matrix                     


# Theoreticall comoving angular diameter distance
def DM_theo_Union3(Om0,H0,w0,wa):
    return DM(zcmb_bin_Union3,Om0,H0,w0,wa)

# Theoretical luminous distance 
def Dl_theo_Union3(Om0,H0,w0,wa):
    return (1+zhel_bin_Union3)*DM_theo_Union3(Om0,H0,w0,wa)

# Theoretical apparent magnitude 
def mb_theo_Union3(Om0,H0,w0,wa,M):
    return 5*np.log10(Dl_theo_Union3(Om0,H0,w0,wa))+25+M

# Construct the chi^2 for SNe - Union3 
def chi2_Union3(Om0,H0,w0,wa,M):
    chi2=0
    delta=mb_bin_Union3-mb_theo_Union3(Om0,H0,w0,wa,M)
    chi2=np.dot(delta,np.dot(inv_cov_Union3,delta))
    return chi2


############################
############################
#
# Chi square for SNe Ia data - DES 
#
############################
############################

Data_path=script_dir+"/Data/DES-SN5YR/4_DISTANCES_COVMAT/DES-SN5YR_HD+MetaData.csv"
data_DES=pan.read_csv(Data_path,comment='#')

# Import the data 

mb_DES=data_DES.MU.values                         # Data of the apparent magnitude mb
mberr_DES=data_DES.MUERR_FINAL.values             # Error in mb 
zcmb_DES=data_DES.zCMB.values                     # Redshift corrected for CMB
zhel_DES=data_DES.zHEL.values                     # Heliocentric redshift  
zHD_DES=data_DES.zHD.values                       # Hubble Diagram Redshift (with CMB and VPEC corrections) 

# The file format for the covariance has the first line as an integer
# indicating the number of covariance elements, and the the subsequent
# lines being the elements.
# This data file is just the systematic component of the covariance - 
 # we also need to add in the statistical error on the magnitudes
# that we loaded earlier
covmat_DES=np.loadtxt(script_dir+"/Data/DES-SN5YR/4_DISTANCES_COVMAT/STAT+SYS.txt",skiprows=1).reshape((1829,1829))
np.fill_diagonal(covmat_DES,covmat_DES.diagonal()+mberr_DES**2)             

# Bin the data using the mask

mask_DES=(zHD_DES<=b) & (zHD_DES>a)          # mask to separate data


# Bin the data using the mask

zcmb_bin_DES=zHD_DES[mask_DES]
zhel_bin_DES=zhel_DES[mask_DES]
mb_bin_DES=mb_DES[mask_DES]
covmat_bin_DES=covmat_DES[mask_DES, :][:, mask_DES]
inv_cov_DES=np.linalg.inv(covmat_bin_DES)                 # Inverse covariance matrix 

# Due to the large amount of data, we made an interpolation among them to reduce the computing time. 

try:
    logzcmb_inter_DES = np.linspace(np.log(min(zcmb_bin_DES))-0.5,np.log(max(zcmb_bin_DES))+0.5,600)
except:
    logzcmb_inter_DES=np.array([])
zcmb_inter_DES = np.exp(logzcmb_inter_DES)

# Theoretical comoving angular diameter distance (Interpolated)
def DM_anali_DES(Om0,H0,w0,wa):
    DM_interpole=0
    DM_interpole=DM(zcmb_inter_DES,Om0,H0,w0,wa)
    try:
        return np.interp(zcmb_bin_DES,zcmb_inter_DES,DM_interpole)
    except:
        return np.array([])

# Theoretical luminous distance 
def Dl_anali_DES(Om0,H0,w0,wa):
    return (1+zhel_bin_DES)*DM_anali_DES(Om0,H0,w0,wa)

# Theoretical apparent magnitude
def mb_anali_DES(Om0,H0,w0,wa,M):
    return 5*np.log10(Dl_anali_DES(Om0,H0,w0,wa))+25+M
 
# Construct the chi^2 for SNe - DES 
def chi2_DES(Om0,H0,w0,wa,M):
    chi2=0
    delta=mb_bin_DES-mb_anali_DES(Om0,H0,w0,wa,M)
    chi2=np.dot(delta,np.dot(inv_cov_DES,delta))
    return chi2

############################
############################
#
# Chi square for SNe Ia data - Pantheon+
#
############################
############################

# Import the data  
data_Pantheon=np.loadtxt(script_dir+"/Data/PantheonPlusSH0ES/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat",skiprows=1,usecols=(6,2,8))
zhel_Pantheon=np.array(data_Pantheon[:,0])
zHD_Pantheon=np.array(data_Pantheon[:,1])
mb_Pantheon=np.array(data_Pantheon[:,2])

# Cov+Sys covariance matrix
covmat_Pantheon=np.loadtxt(script_dir+"/Data/PantheonPlusSH0ES/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES_STAT+SYS.cov",skiprows=1).reshape((1701,1701))

mask_Pantheon=(zHD_Pantheon<=b) & (zHD_Pantheon>a)                   # mask to separate data

# Bin the data using the mask

zcmb_bin_Pantheon=zHD_Pantheon[mask_Pantheon]
zhel_bin_Pantheon=zhel_Pantheon[mask_Pantheon]
mb_bin_Pantheon=mb_Pantheon[mask_Pantheon]
covmat_bin_Pantheon=covmat_Pantheon[mask_Pantheon, :][:,mask_Pantheon]

 # Inverse covariance matrix 
inv_cov_Pantheon=np.linalg.inv(covmat_bin_Pantheon)                 

# Due to the large amount of data, we made an interpolation among them to reduce the computing time. 

try:
    logzcmb_inter_Pantheon = np.linspace(np.log(min(zcmb_bin_Pantheon))-0.5,np.log(max(zcmb_bin_Pantheon))+0.5,600)
except: 
    logzcmb_inter_Pantheon = np.array([])
    
zcmb_inter_Pantheon = np.exp(logzcmb_inter_Pantheon)

# Theoretical comoving angular diameter distance (Interpolated)
def DM_anali_Pantheon(Om0,H0,w0,wa):
    DM_interpole=0
    DM_interpole=DM(zcmb_inter_Pantheon,Om0,H0,w0,wa)
    try:
        return np.interp(zcmb_bin_Pantheon,zcmb_inter_Pantheon,DM_interpole)
    except: 
        return np.array([])

# Theoretical luminous distance 
def Dl_anali_Pantheon(Om0,H0,w0,wa):
    return (1+zhel_bin_Pantheon)*DM_anali_Pantheon(Om0,H0,w0,wa)

# Theoretical apparent magnitude
def mb_anali_Pantheon(Om0,H0,w0,wa,M):
    return 5*np.log10(Dl_anali_Pantheon(Om0,H0,w0,wa))+25+M
 
# Construct the chi^2 for SNe - Pantheon+
def chi2_Pantheon(Om0,H0,w0,wa,M):
    chi2=0
    delta=mb_bin_Pantheon-mb_anali_Pantheon(Om0,H0,w0,wa,M)
    chi2=np.dot(delta,np.dot(inv_cov_Pantheon,delta))
    return chi2

############################
############################
#
# Chi square for the Megamasers
#
############################
############################

# Import the data
DA_maser=np.genfromtxt(script_dir+"/Data/Megamasers/megamaser.dat", usecols=1)          # Angular diameter distance data
err_DA_maser=np.genfromtxt(script_dir+"/Data/Megamasers/megamaser.dat", usecols=2)      # error in D_A
vel_maser=np.genfromtxt(script_dir+"/Data/Megamasers/megamaser.dat", usecols=3)         # velocity data
err_vel_maser=np.genfromtxt(script_dir+"/Data/Megamasers/megamaser.dat", usecols=4)     # error in velocity 
err_pec_maser=250                                                                       # error in peculiar velocity 

#Covariance matrix for D_A that only includes statistical errors
cov_DA_maser=np.diag(err_DA_maser**2) 
invcov_DA_maser=np.linalg.inv(cov_DA_maser)

#Covariance matrix for the velocity that includes statistical errors and an error due the peculiar velocity
cov_vel_maser=np.diag(err_vel_maser**2+err_pec_maser**2)
invcov_vel_maser=np.linalg.inv(cov_vel_maser)

# Array for the Theoretical velocities (nuisance parameters)
def vel_anali(v1,v2,v3,v4,v5,v6):
    return np.array([v1,v2,v3,v4,v5,v6])

# Construct the chi^2 for Megamasers
def chi2_maser(v1,v2,v3,v4,v5,v6,Om0,H0,w0,wa):
    if a<=0.01 :                                                   # Megamasers are at low z's
        chi2=0
        delta_vel=vel_anali(v1,v2,v3,v4,v5,v6)-vel_maser
        zs = vel_anali(v1,v2,v3,v4,v5,v6)/c                               # "Theoretical" redshift
        delta_DA=DA(zs,Om0,H0,w0,wa)-DA_maser
        chi2=np.dot(delta_vel,np.dot(invcov_vel_maser,delta_vel)) + np.dot(delta_DA,np.dot(invcov_DA_maser,delta_DA))
    else:
        chi2=0.0
    return chi2

############################
############################
#
# Chi square for DESI-BAO BGS (z=0.30)
#
############################
############################

# Import the data
z_BGS=np.loadtxt(script_dir+"/Data/BAO-DESI/desi_2024_gaussian_bao_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4_mean.txt",usecols=0)  # Redshift
Dv_BGS=np.loadtxt(script_dir+"/Data/BAO-DESI/desi_2024_gaussian_bao_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4_mean.txt",usecols=1) # Spherically averaged distance
cov_BGS=np.loadtxt(script_dir+"/Data/BAO-DESI/desi_2024_gaussian_bao_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4_cov.txt",usecols=0) # Error in Dv

invcov_BGS=cov_BGS**(-1)

# Construct the chi^2 for DESI-BAO BGS 
def chi2_BGS(Om0,H0,w0,wa,rd):
    if a<0.3 and b>0.3:                                                 
        chi2=0
        delta= Dv(z_BGS,Om0,H0,w0,wa,)/rd-Dv_BGS
        chi2=np.dot(delta,np.dot(invcov_BGS,delta))
    else:
        chi2=0.0
    return chi2

############################
############################
#
# Chi square for DESI-BAO LRG (z=0.51,0.71)
#
############################
############################

# Import the data
data_LRG=np.genfromtxt(script_dir+"/Data/BAO-DESI/desi_2024_gaussian_bao_ALL_GCcomb_mean.txt")  # data
cov_LRG=np.genfromtxt(script_dir+"/Data/BAO-DESI/desi_2024_gaussian_bao_ALL_GCcomb_cov.txt")    # Covariance Matrix

mask_split=(data_LRG[:,0]==0.51) | (data_LRG[:,0]==0.706)                                          # mask to separate the combined data

data_LRG=data_LRG[mask_split]
z_LRG=data_LRG[:,0]                                                                        # Redshift
dist_LRG=data_LRG[:,1]                                                                     # Distances D_H and D_M

cov_LRG=cov_LRG[mask_split,:][:,mask_split]                                # Total covariance Matrix                                   

# Separate the D_M/rd data 
mask_DM=(True,False,True,False)
mask_DM=np.array(mask_DM)
z_LRG=z_LRG[mask_DM]
DM_LRG=dist_LRG[mask_DM]                                               # Here, D_M is divided by the sound horizon r_d 

# Separate the D_H/rd data
mask_DH=(False,True,False,True)
mask_DH=np.array(mask_DH)
DH_LRG=dist_LRG[mask_DH]                                               # Here, D_H is divided by the sound horizon r_d


# Split the data according to the bin used 
mask_LRG=(z_LRG<=b) & (z_LRG>a)

z_bin_LRG=z_LRG[mask_LRG]
DM_bin_LRG=DM_LRG[mask_LRG]
DH_bin_LRG=DH_LRG[mask_LRG]

# Split the covariance matrix

masc_for_cov =[mask_LRG[0],mask_LRG[0],mask_LRG[1],mask_LRG[1]]
cov_bin_LRG=cov_LRG[masc_for_cov,:][:,masc_for_cov]

try:
    inv_cov_LRG=np.linalg.inv(cov_bin_LRG)
except:
    inv_cov_LRG = np.array([])
    
# Theoretical comoving angular diameter distance divided by r_d
def DM_anali_LRG(Om0,H0,w0,wa,rd):
    return DM(z_bin_LRG,Om0,H0,w0,wa)/rd

# Theoretical Hubble distance divided by r_d
def DH_anali_LRG(Om0,H0,w0,wa,rd):
    return DH(z_bin_LRG,Om0,H0,w0,wa)/rd

# Construct the chi^2 for DESI-BAO LRG
def chi2_LRG(Om0,H0,w0,wa,rd):
    chi2=0
    deltaDM_LRG=DM_bin_LRG-DM_anali_LRG(Om0,H0,w0,wa,rd)
    deltaDH_LRG=DH_bin_LRG-DH_anali_LRG(Om0,H0,w0,wa,rd)
    shape = np.shape(deltaDM_LRG)
    if shape == (2,):
        deltatot_LRG=np.array([deltaDM_LRG[0],deltaDH_LRG[0],deltaDM_LRG[1],deltaDH_LRG[1]])
        chi2=np.dot(deltatot_LRG,np.dot(inv_cov_LRG,deltatot_LRG))
    elif shape == (1,):
        deltatot_LRG=np.array([deltaDM_LRG[0],deltaDH_LRG[0]])
        chi2=np.dot(deltatot_LRG,np.dot(inv_cov_LRG,deltatot_LRG))
    elif shape == (0,):
        chi2 = 0
    return chi2


############################
############################
#
# Chi square for DESI-BAO LRG + ELG (z=0.93)
#
############################
############################

z_LE=np.loadtxt(script_dir+"/Data/BAO-DESI/desi_2024_gaussian_bao_LRG+ELG_LOPnotqso_GCcomb_z0.8-1.1_mean.txt",usecols=0)[0]  #redshift data
dist_LE=np.loadtxt(script_dir+"/Data/BAO-DESI/desi_2024_gaussian_bao_LRG+ELG_LOPnotqso_GCcomb_z0.8-1.1_mean.txt",usecols=1)
DM_LE=dist_LE[0]               # Distance D_M
DH_LE=dist_LE[1]               # Distance D_H 
cov_LE=np.loadtxt(script_dir+"/Data/BAO-DESI/desi_2024_gaussian_bao_LRG+ELG_LOPnotqso_GCcomb_z0.8-1.1_cov.txt")    # covariance matrix 
inv_covLE=np.linalg.inv(cov_LE)                # Inverse covariance matrix

# Theoretical comoving angular diameter distance divided by r_d
def DM_anali_LE(Om0,H0,w0,wa,rd):
    return DM(z_LE,Om0,H0,w0,wa)/rd

# Theoretical Hubble distance divided by r_d
def DH_anali_LE(Om0,H0,w0,wa,rd):
    return DH(z_LE,Om0,H0,w0,wa)/rd

# Construct the chi^2 for DESI-BAO LRG+ELG
def chi2_LE(Om0,H0,w0,wa,rd):
    if a<0.93 and b>0.93:  
        chi2=0                                                 
        deltaDM_LE=DM_LE-DM_anali_LE(Om0,H0,w0,wa,rd)
        deltaDH_LE=DH_LE-DH_anali_LE(Om0,H0,w0,wa,rd)
        deltatot_LE=np.array([deltaDM_LE,deltaDH_LE])
        chi2=np.dot(deltatot_LE,np.dot(inv_covLE,deltatot_LE))
    else:
        chi2=0.0
    return chi2


############################
############################
#
# Chi square for DESI-BAO ELG (z=1.32)
#
############################
############################

z_ELG=np.loadtxt(script_dir+"/Data/BAO-DESI/desi_2024_gaussian_bao_ELG_LOPnotqso_GCcomb_z1.1-1.6_mean.txt",usecols=0)[0] #redshift data
dist_ELG=np.loadtxt(script_dir+"/Data/BAO-DESI/desi_2024_gaussian_bao_ELG_LOPnotqso_GCcomb_z1.1-1.6_mean.txt",usecols=1)
DM_ELG=dist_ELG[0]                # Distance D_M
DH_ELG=dist_ELG[1]                # Distance D_H
cov_ELG=np.loadtxt(script_dir+"/Data/BAO-DESI/desi_2024_gaussian_bao_ELG_LOPnotqso_GCcomb_z1.1-1.6_cov.txt")    # covariance matrix
inv_covELG=np.linalg.inv(cov_ELG)      # Inverse covariance matrix

# Theoretical comoving angular diameter distance divided by r_d
def DM_analiELG(Om0,H0,w0,wa,rd):
    return DM(z_ELG,Om0,H0,w0,wa)/rd

# Theoretical Hubble distance divided by r_d
def DH_analiELG(Om0,H0,w0,wa,rd):
    return DH(z_ELG,Om0,H0,w0,wa)/rd

# Construct the chi^2 for DESI-BAO ELG
def chi2_ELG(Om0,H0,w0,wa,rd):
    if a<1.32 and b>1.32:  
        deltaDM_ELG=DM_ELG-DM_analiELG(Om0,H0,w0,wa,rd)
        deltaDH_ELG=DH_ELG-DH_analiELG(Om0,H0,w0,wa,rd)
        deltatot_ELG=np.array([deltaDM_ELG,deltaDH_ELG])
        chi2=np.dot(deltatot_ELG,np.dot(inv_covELG,deltatot_ELG))
    else:
        chi2=0.0
    return chi2

############################
############################
#
# Chi square for DESI-BAO QSO (z=1.49)
#
############################
############################    


z_QSO=np.loadtxt(script_dir+"/Data/BAO-DESI/desi_2024_gaussian_bao_QSO_GCcomb_z0.8-2.1_mean.txt",usecols=0)    #redshift data
Dv_QSO=np.loadtxt(script_dir+"/Data/BAO-DESI/desi_2024_gaussian_bao_QSO_GCcomb_z0.8-2.1_mean.txt",usecols=1)   # Distance D_V
cov_QSO=np.loadtxt(script_dir+"/Data/BAO-DESI/desi_2024_gaussian_bao_QSO_GCcomb_z0.8-2.1_cov.txt")             # covariance matrix
inv_covQSO=cov_QSO**(-1)      # Inverse covariance matrix

# Construct the chi^2 for DESI-BAO QSO
def chi2_QSO(Om0,H0,w0,wa,rd):
    if a<1.49 and b>1.49:  
        chi2=0
        delta=Dv(z_QSO,Om0,H0,w0,wa)/rd-Dv_QSO
        chi2=np.dot(delta,np.dot(inv_covQSO,delta))
    else:
        chi2=0.0
    return chi2

############################
############################
#
# Chi square for DESI-BAO Lya-QSO (z=2.33)
#
############################
############################  


z_Ly=np.loadtxt(script_dir+"/Data/BAO-DESI/desi_2024_gaussian_bao_Lya_GCcomb_mean.txt",usecols=0)[0]    #redshift data
dist_Ly=np.loadtxt(script_dir+"/Data/BAO-DESI/desi_2024_gaussian_bao_Lya_GCcomb_mean.txt",usecols=1)    
DH_Ly=dist_Ly[0]                   # Distance D_H
DM_Ly=dist_Ly[1]                   # Distance D_M
cov_Ly=np.loadtxt(script_dir+"/Data/BAO-DESI/desi_2024_gaussian_bao_Lya_GCcomb_cov.txt")                # covariance matrix
inv_covLy=np.linalg.inv(cov_Ly)    # Inverse covariance matrix


# Theoretical comoving angular diameter distance divided by r_d
def DM_analiLy(Om0,H0,w0,wa,rd):
    return DM(z_Ly,Om0,H0,w0,wa)/rd

# Theoretical Hubble distance divided by r_d
def DH_analiLy(Om0,H0,w0,wa,rd):
    return DH(z_Ly,Om0,H0,w0,wa)/rd

# Construct the chi^2 for DESI-BAO Lya-QSO
def chi2_Ly(Om0,H0,w0,wa,rd):
    if a<2.33 and b>=2.33:  
        chi2=0
        deltaDH_Ly=DH_Ly-DH_analiLy(Om0,H0,w0,wa,rd)
        deltaDM_Ly=DM_Ly-DM_analiLy(Om0,H0,w0,wa,rd)
        deltatot_Ly=np.array([deltaDH_Ly,deltaDM_Ly])
        chi2=np.dot(deltatot_Ly,np.dot(inv_covLy,deltatot_Ly))
    else:
        chi2=0.0
    return chi2


############################
############################
#
# Total Chi square for DESI data
#
############################
############################ 

def chi2_DESI(Om0,H0,w0,wa,rd):
    return chi2_BGS(Om0,H0,w0,wa,rd)+chi2_LRG(Om0,H0,w0,wa,rd)+chi2_LE(Om0,H0,w0,wa,rd)+chi2_ELG(Om0,H0,w0,wa,rd)+chi2_QSO(Om0,H0,w0,wa,rd)+chi2_Ly(Om0,H0,w0,wa,rd)

############################
############################
#
# Total Chi square for Base+Sne
#
############################
############################ 


# Base+PantheonPlus
def chi2_Base_PantheonPlus(arg):
    v1,v2,v3,v4,v5,v6,Om0,H0,w0,wa,M,rd=arg
    return chi2_maser(v1,v2,v3,v4,v5,v6,Om0,H0,w0,wa)+chi2_CC(Om0,H0,w0,wa)+chi2_Pantheon(Om0,H0,w0,wa,M)+chi2_DESI(Om0,H0,w0,wa,rd)

# Base+Union3
def chi2_Base_Union3(arg):
    v1,v2,v3,v4,v5,v6,Om0,H0,w0,wa,M,rd=arg
    return chi2_maser(v1,v2,v3,v4,v5,v6,Om0,H0,w0,wa)+chi2_CC(Om0,H0,w0,wa)+chi2_Union3(Om0,H0,w0,wa,M)+chi2_DESI(Om0,H0,w0,wa,rd)
    
# Base+DES
def chi2_Base_DES(arg):
    v1,v2,v3,v4,v5,v6,Om0,H0,w0,wa,M,rd=arg
    return chi2_maser(v1,v2,v3,v4,v5,v6,Om0,H0,w0,wa)+chi2_CC(Om0,H0,w0,wa)+chi2_DES(Om0,H0,w0,wa,M)+chi2_DESI(Om0,H0,w0,wa,rd)


