from scipy import integrate
import numpy as np
from scipy import interpolate
import os 
import pandas as pan
from astropy.io import fits
import bin_range
from numpy.polynomial.legendre import leggauss

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

CC        = np.loadtxt(script_dir+"/Data/CC/CC_data.dat")           # CC data
cov_matCC = np.load(script_dir+"/Data/CC/cov_matCC.npy")     # Covariance matrix 
z_CC      = np.array(CC[:,0])                                     # Redshifts  
H_obser   = np.array(CC[:,1])                                  # H(z) data

mask_CC = (z_CC<=b) & (z_CC>a)                               # Create a mask according to the bin

# Bin the data using the mask

zbin_CC     = z_CC[mask_CC]
Hbin_obser  = H_obser[mask_CC]
covbin_CC   = cov_matCC[mask_CC, :][:, mask_CC]                  # Reconstruct the covariance matrix according to the binned data                     
inv_covCC   = np.linalg.inv(covbin_CC)                           # Inverse of the covariance matrix 


# Theoretical function for H(z)
def H_theo(Om0,H0,w0,wa): 
    return H(zbin_CC,Om0,H0,w0,wa)  

# Construct the chi^2 for CC
def chi2_CC(Om0,H0,w0,wa):
    chi2  = 0
    delta = H_theo(Om0,H0,w0,wa)-Hbin_obser
    chi2  = delta @ (inv_covCC @ delta) 
    return chi2

############################
############################
#
# Chi square for SNe Ia data - Union3 
#
############################
############################

# Import the data 

file_Union3   = fits.open(script_dir+"/Data/union3_release/mu_mat_union3_cosmo=2_mu.fits")
data_Union3   = file_Union3[0].data
zcmb_Union3   = data_Union3[0,1:]                                                              # Redshift corrected for CMB
zhel_Union3   = data_Union3[0,1:]                                                              # Heliocentric redshift 
mb_Union3     = data_Union3[1:,0]                                                                # data of the apparent magnitude mb
cov_matUnion3 = np.linalg.inv(data_Union3[1:,1:])                                          # Covariance matrix

mask_Union3 = (zcmb_Union3<=b) & (zcmb_Union3>a)                   # mask to separate data

# Bin the data using the mask

zcmb_bin_Union3   = zcmb_Union3[mask_Union3]
mb_bin_Union3     = mb_Union3[mask_Union3]
covbin_mat_Union3 = cov_matUnion3[mask_Union3, :][:, mask_Union3]  # Reconstruct the covariance matrix according to the binned data
inv_cov_Union3    = np.linalg.inv(covbin_mat_Union3)                  # Inverse of the covariance matrix

# Theoretical apparent magnitude 
def mb_theo_Union3(Om0,H0,w0,wa,M):
    return mb(zcmb_bin_Union3,Om0,H0,w0,wa,M)    

# NOTE: In the Union3 and DES samples, the fitted magnitude parameter represents 
# an offset ΔM relative to the absolute magnitude calibration (MB) rather than MB itself. 
# Therefore, M here should be interpreted as ΔM, not as the absolute magnitude.

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

Data_path = script_dir+"/Data/DES-SN5YR/4_DISTANCES_COVMAT/DES-SN5YR_HD+MetaData.csv"
data_DES  = pan.read_csv(Data_path,comment='#')

# Import the data 

mb_DES    = data_DES.MU.values                         # Data of the apparent magnitude mb
mberr_DES = data_DES.MUERR_FINAL.values             # Error in mb 
zcmb_DES  = data_DES.zCMB.values                     # Redshift corrected for CMB
zhel_DES  = data_DES.zHEL.values                     # Heliocentric redshift  
zHD_DES   = data_DES.zHD.values                       # Hubble Diagram Redshift (with CMB and VPEC corrections) 

# ===============================================================
# The file format for the covariance has the first line as an integer
# indicating the number of covariance elements, and the the subsequent
# lines being the elements.
# This data file is just the systematic component of the covariance - 
# we also need to add in the statistical error on the magnitudes
# that we loaded earlier
# ===============================================================

covmat_DES = np.loadtxt(script_dir+"/Data/DES-SN5YR/4_DISTANCES_COVMAT/STAT+SYS.txt",skiprows=1).reshape((1829,1829))
np.fill_diagonal(covmat_DES,covmat_DES.diagonal()+mberr_DES**2)             

# Bin the data using the mask

mask_DES=(zHD_DES<=b) & (zHD_DES>a)          # mask to separate data


# Bin the data using the mask

zcmb_bin_DES   = zHD_DES[mask_DES]
zhel_bin_DES   = zhel_DES[mask_DES]
mb_bin_DES     = mb_DES[mask_DES]
covmat_bin_DES = covmat_DES[mask_DES, :][:, mask_DES]
inv_cov_DES    = np.linalg.inv(covmat_bin_DES)                 # Inverse covariance matrix 

# ===============================================================
# We calculate the comoving distance integral using 
# Gauss–Legendre quadrature for each supernova individually.
# For each SN with redshift z_i, we precompute NGL_DES nodes (z_nodes_DES)
# and weights (w_scaled_DES) scaled to the interval [0, z_i].
# During each model evaluation, D_M(z_i) is obtained by summing
# c * Σ_k [ w_scaled_DES[i,k] / H(z_nodes_DES[i,k]) ].
# This approach greatly speeds up the computation while maintaining high accuracy.
# ===============================================================

NGL_DES = 80  # 
xk_DES, wk_DES = leggauss(NGL_DES)  # 

# z_nodes_DES and w_scaled_DES, shape (#SNe, NGL_DES)

if zcmb_bin_DES.size > 0:
    z_nodes_DES  = 0.5 * zcmb_bin_DES[:, None] * (xk_DES[None, :] + 1.0)
    w_scaled_DES = 0.5 * zcmb_bin_DES[:, None] *  wk_DES[None, :]
else:
    # empty bin
    z_nodes_DES  = np.empty((0, NGL_DES))
    w_scaled_DES = np.empty((0, NGL_DES))

def DM_anali_DES(Om0, H0, w0, wa):
    if z_nodes_DES.size == 0:
        return np.array([])
    Hz_nodes = H(z_nodes_DES, Om0, H0, w0, wa)        # 
    DM = c * np.sum(w_scaled_DES / Hz_nodes, axis=1)  # shape (#SNe,)
    return DM

def Dl_anali_DES(Om0, H0, w0, wa):
    return (1.0 + zhel_bin_DES) * DM_anali_DES(Om0, H0, w0, wa)

def mb_anali_DES(Om0, H0, w0, wa, M):
    return 5.0 * np.log10(Dl_anali_DES(Om0, H0, w0, wa)) + 25.0 + M

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
data_Pantheon = np.loadtxt(script_dir+"/Data/PantheonPlusSH0ES/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat",skiprows=1,usecols=(6,2,8))
zhel_Pantheon = np.array(data_Pantheon[:,0])
zHD_Pantheon  = np.array(data_Pantheon[:,1])
mb_Pantheon   = np.array(data_Pantheon[:,2])

# Cov+Sys covariance matrix
covmat_Pantheon = np.loadtxt(script_dir+"/Data/PantheonPlusSH0ES/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES_STAT+SYS.cov",skiprows=1).reshape((1701,1701))

mask_Pantheon = (zHD_Pantheon<=b) & (zHD_Pantheon>a)                   # mask to separate data

# Bin the data using the mask

zcmb_bin_Pantheon   = zHD_Pantheon[mask_Pantheon]
zhel_bin_Pantheon   = zhel_Pantheon[mask_Pantheon]
mb_bin_Pantheon     = mb_Pantheon[mask_Pantheon]
covmat_bin_Pantheon = covmat_Pantheon[mask_Pantheon, :][:,mask_Pantheon]

 # Inverse covariance matrix 
inv_cov_Pantheon = np.linalg.inv(covmat_bin_Pantheon)                 

# ===============================================================
# Same methodology as in the DES section:
# We compute the comoving distance integral for each supernova
# using Gauss–Legendre quadrature.
# ===============================================================

NGL_PAN = 80 
xk_PAN, wk_PAN = leggauss(NGL_PAN) 

if zcmb_bin_Pantheon.size > 0:
    z_nodes_PAN  = 0.5 * zcmb_bin_Pantheon[:, None] * (xk_PAN[None, :] + 1.0)
    w_scaled_PAN = 0.5 * zcmb_bin_Pantheon[:, None] *  wk_PAN[None, :]
else:
    z_nodes_PAN  = np.empty((0, NGL_PAN))
    w_scaled_PAN = np.empty((0, NGL_PAN))

def DM_anali_Pantheon(Om0, H0, w0, wa):
    if z_nodes_PAN.size == 0:
        return np.array([])
    Hz_nodes = H(z_nodes_PAN, Om0, H0, w0, wa)        
    DM = c * np.sum(w_scaled_PAN / Hz_nodes, axis=1)  # (#SNe,)
    return DM

def Dl_anali_Pantheon(Om0, H0, w0, wa):
    return (1.0 + zhel_bin_Pantheon) * DM_anali_Pantheon(Om0, H0, w0, wa)

def mb_anali_Pantheon(Om0, H0, w0, wa, M):
    return 5.0 * np.log10(Dl_anali_Pantheon(Om0, H0, w0, wa)) + 25.0 + M
 
# Construct the chi^2 for SNe - Pantheon+
def chi2_Pantheon(Om0,H0,w0,wa,M):
    chi2=0
    delta=mb_bin_Pantheon-mb_anali_Pantheon(Om0,H0,w0,wa,M)
    chi2=delta @ (inv_cov_Pantheon @ delta)
    return chi2

############################
############################
#
# Chi square for the Megamasers
#
############################
############################

# Import the data
DA_maser      = np.genfromtxt(script_dir+"/Data/Megamasers/megamaser.dat", usecols=1)          # Angular diameter distance data
err_DA_maser  = np.genfromtxt(script_dir+"/Data/Megamasers/megamaser.dat", usecols=2)      # error in D_A
vel_maser     = np.genfromtxt(script_dir+"/Data/Megamasers/megamaser.dat", usecols=3)         # velocity data
err_vel_maser = np.genfromtxt(script_dir+"/Data/Megamasers/megamaser.dat", usecols=4)     # error in velocity 
err_pec_maser = 250                                                                       # error in peculiar velocity 

#Covariance matrix for D_A that only includes statistical errors
cov_DA_maser    = np.diag(err_DA_maser**2) 
invcov_DA_maser = np.linalg.inv(cov_DA_maser)

#Covariance matrix for the velocity that includes statistical errors and an error due the peculiar velocity
cov_vel_maser    = np.diag(err_vel_maser**2+err_pec_maser**2)
invcov_vel_maser = np.linalg.inv(cov_vel_maser)

# Array for the Theoretical velocities (nuisance parameters)
def vel_anali(v1,v2,v3,v4,v5,v6):
    return np.array([v1,v2,v3,v4,v5,v6])

# Construct the chi^2 for Megamasers
def chi2_maser(v1,v2,v3,v4,v5,v6,Om0,H0,w0,wa):
    if a<=0.01 :                                                   # Megamasers are at low z's
        chi2=0
        delta_vel = vel_anali(v1,v2,v3,v4,v5,v6)-vel_maser
        zs        = vel_anali(v1,v2,v3,v4,v5,v6)/c                               # "Theoretical" redshift
        delta_DA  = DA(zs,Om0,H0,w0,wa)-DA_maser
        chi2      = delta_vel @ (invcov_vel_maser @ delta_vel) + delta_DA @ (invcov_DA_maser @ delta_DA)
    else:
        chi2=0.0
    return chi2


############################
############################
#
# Chi square for DESI-BAO DR2
#
############################
############################

# Data
arr        = np.genfromtxt(script_dir+"/Data/BAO_data/desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_mean.txt", 
                           skip_header=1, dtype=None, encoding=None)  
z_desi     = np.array([d[0] for d in arr], dtype=float)
data_desi  = np.array([d[1] for d in arr], dtype=float)
qty        = np.array([d[2] for d in arr])

# Covariance matrix 
covmat_desi = np.loadtxt(script_dir+"/Data/BAO_data/desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_cov.txt")

mask_DESI = (z_desi <= b ) & (z_desi > a)

def make_chi2_DESI_with_mask(mask):
    z_sel   = z_desi[mask_DESI]
    d_sel   = data_desi[mask_DESI]
    qty_sel = qty[mask_DESI]
    C_sel   = covmat_desi[mask_DESI,:][:, mask_DESI]
    invC_sel = np.linalg.inv(C_sel)


    def chi2_DESI(Om0, H0, w0, wa, rs):
        DMv = DM(z_sel,Om0,H0,w0,wa)   # D_M(z)
        DHv = DH(z_sel,Om0,H0,w0,wa)   # D_H(z)
        model_vec = np.empty_like(d_sel)

        for i in range(len(z_sel)):
            if   qty_sel[i] == "DM_over_rs":
                model_vec[i] = DMv[i] / rs
            elif qty_sel[i] == "DH_over_rs":
                model_vec[i] = DHv[i] / rs
            elif qty_sel[i] == "DV_over_rs":
                model_vec[i] = (z_sel[i] * DHv[i] * DMv[i]**2)**(1.0/3.0) / rs
            else:
                raise ValueError("Cantidad no reconocida: " + qty_sel[i])

        r = d_sel - model_vec
        return float(r @ (invC_sel @ r))

    return chi2_DESI

chi2_DESI = make_chi2_DESI_with_mask(mask_DESI)

############################
############################
#
# Total Chi square for Base+SNe
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


