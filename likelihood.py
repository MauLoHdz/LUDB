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
zcmb_Union3   = data_Union3[0,1:]                                                          # Redshift corrected for CMB
zhel_Union3   = data_Union3[0,1:]                                                          # Heliocentric redshift 
mb_Union3     = data_Union3[1:,0]                                                          # data of the apparent magnitude mb
cov_matUnion3 = np.linalg.inv(data_Union3[1:,1:])                                          # Covariance matrix

mask_Union3 = (zcmb_Union3<=b) & (zcmb_Union3>a)                   # mask to separate data

# Bin the data using the mask

zcmb_bin_Union3   = zcmb_Union3[mask_Union3]
mb_bin_Union3     = mb_Union3[mask_Union3]
covbin_mat_Union3 = cov_matUnion3[mask_Union3, :][:, mask_Union3]     # Reconstruct the covariance matrix according to the binned data
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
# Chi square for SNe Ia data - DES-Dovekie
#
############################
############################

def load_dovekie_ascii(path):
    """
    Reads the DES-Dovekie_HD.csv SNANA-style ASCII file.
    """
    with open(path, "r") as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        if line.startswith("VARNAMES:"):
            header_line = i
            break
    else:
        raise RuntimeError("VARNAMES: not found in file")
    columns = (
        lines[header_line]
        .replace("VARNAMES:", "")
        .strip()
        .split()
    )

    df = pan.read_csv(
        path,
        comment="#",
        skiprows=header_line + 1,
        sep=r"\s+",         
        names=columns,
        engine="python"
    )

    return df
    
path = script_dir + "/Data/DES-SN5YR/4_DISTANCES_COVMAT/DES-Dovekie_HD.csv"
df_DES = load_dovekie_ascii(path)

# Import the data 

mu_DES   = df_DES.MU.values     # modulus distance
zhel_DES = df_DES.zHEL.values   # Heliocentric redshift  
zHD_DES  = df_DES.zHD.values    # Hubble Diagram Redshift (CMB+Vpec corrected)

# Import covmat file 
inv_cov_npz = np.load(script_dir + "/Data/DES-SN5YR/4_DISTANCES_COVMAT/STAT+SYS.npz")  
inv_cov_vector = inv_cov_npz["cov"]    # Flattened upper triangle of the inverse covariance matrix 

# Dimension N
N = int((-1 + np.sqrt(1 + 8 * len(inv_cov_vector))) // 2)

# Built the full matrix
inv_covmat_DES = np.zeros((N, N))
inv_covmat_DES[np.triu_indices(N)] = inv_cov_vector

i_lower = np.tril_indices(N, -1)
inv_covmat_DES[i_lower] = inv_covmat_DES.T[i_lower]


# Bin the data using the mask

mask_DES=(zHD_DES<=b) & (zHD_DES>a)          # mask to separate data

# Bin the data using the mask

zcmb_bin_DES   = zHD_DES[mask_DES]
zhel_bin_DES   = zhel_DES[mask_DES]
mu_bin_DES     = mu_DES[mask_DES]
inv_cov_DES    = inv_covmat_DES[mask_DES, :][:, mask_DES]

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

def mu_anali_DES(Om0, H0, w0, wa, M):
    return 5.0 * np.log10(Dl_anali_DES(Om0, H0, w0, wa)) + 25.0 + M  # Offset to absorbs the change in H0

# Construct the chi^2 for SNe - DES 
def chi2_DES(Om0,H0,w0,wa,M):
    chi2  = 0
    delta = mu_bin_DES-mu_anali_DES(Om0,H0,w0,wa,M)
    chi2  = np.dot(delta,np.dot(inv_cov_DES,delta))
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
# Chi square for Planck Distance Priors (DP)
#
############################
############################

# Internal constants and tolerances for the DP block
_dp_Tcmb = 2.7255     # Kelvin
_DP_EPSABS = 1e-8
_DP_EPSREL = 1e-8

# --- Helper functions for DP only ------

def _dp_z_eq(Om0, h):
    """Redshift at matter-radiation equality ."""
    return 2.5e4 * Om0 * h**2 * (_dp_Tcmb / 2.7)**(-4)

def _dp_Omega_r(Om0, h):
    """Radiation density parameter today (using z_eq)."""
    return Om0 / (1.0 + _dp_z_eq(Om0, h))

def _dp_z_decou(Omega_bh2, Om0h2):
    """Photon decoupling redshift z*."""
    g1 = 0.0738 * Omega_bh2**(-0.238) / (1.0 + 39.5 * Omega_bh2**0.763)
    g2 = 0.560 / (1.0 + 21.1 * Omega_bh2**1.81)
    return 1048.0 * (1.0 + 0.00124 * Omega_bh2**(-0.738)) * (1.0 + g1 * Om0h2**g2)

def _dp_Ez(z, Om0, Or, Ode0, w0, wa):
    """
    E(z) = H(z)/H0 for flat w0waCDM (CPL).
    Dark energy factor: f_de(a) = a^{-3(1+w0+wa)} * exp[3*wa*(a-1)] with a = 1/(1+z).
    """
    a = 1.0 / (1.0 + z)
    f_de = a**(-3.0 * (1.0 + w0 + wa)) * np.exp(3.0 * wa * (a - 1.0))
    return np.sqrt(Or * (1.0 + z)**4 + Om0 * (1.0 + z)**3 + Ode0 * f_de)

def _dp_comoving_distance(z, Om0, Or, Ode0, H0, w0, wa):
    """Line-of-sight comoving distance chi(z) in Mpc for w0waCDM."""
    integrand = lambda zp: 1.0 / _dp_Ez(zp, Om0, Or, Ode0, w0, wa)
    val, _ = integrate.quad(integrand, 0.0, z, epsabs=_DP_EPSABS, epsrel=_DP_EPSREL)
    return (c / H0) * val

def _dp_DA(z, Om0, Or, Ode0, H0, w0, wa):
    """Angular-diameter distance D_A(z) = chi(z)/(1+z) for w0waCDM."""
    return _dp_comoving_distance(z, Om0, Or, Ode0, H0, w0, wa) / (1.0 + z)

# Photon density for r_s
_dp_Omega_gamma_h2 = (3.0 / (4.0 * 31500.0)) * (_dp_Tcmb / 2.7)**4

def _dp_r_sound(z, Omega_bh2, Om0, Or, Ode0, H0, w0, wa):
    """
    Sound horizon r_s(z) in Mpc. Dark energy is negligible at early times, but included for consistency.
    Implemented in scale factor a, where 1+z = 1/a.
    """
    def E_a(a):
        f_de = a**(-3.0 * (1.0 + w0 + wa)) * np.exp(3.0 * wa * (a - 1.0))
        return np.sqrt(Or * a**(-4) + Om0 * a**(-3) + Ode0 * f_de)

    def integrand(a):
        Rb = 3.0 * Omega_bh2 / (4.0 * _dp_Omega_gamma_h2) * a
        return 1.0 / (a**2 * E_a(a) * np.sqrt(3.0 * (1.0 + Rb)))

    a_max = 1.0 / (1.0 + z)
    val, _ = integrate.quad(integrand, 0.0, a_max, epsabs=_DP_EPSABS, epsrel=_DP_EPSREL)
    return (c / H0) * val

def _dp_compute_R_lA(Om0, H0, w0, wa, Omega_bh2):
    """
    Compute (R, l_A) at the photon decoupling redshift z* for flat w0waCDM.
    R    = (1+z*) * D_A(z*) * H0 * sqrt(Om0) / c
    l_A  = (1+z*) * pi * D_A(z*) / r_s(z*)
    """
    h = H0 / 100.0
    Om0h2 = Om0 * h**2
    zstar = _dp_z_decou(Omega_bh2, Om0h2)
    Or = _dp_Omega_r(Om0, h)
    Ode0 = 1.0 - Om0 - Or  # flatness
    Da = _dp_DA(zstar, Om0, Or, Ode0, H0, w0, wa)
    rs = _dp_r_sound(zstar, Omega_bh2, Om0, Or, Ode0, H0, w0, wa)
    Rval = (1.0 + zstar) * Da * (H0 * np.sqrt(Om0)) / c
    l_A  = (1.0 + zstar) * np.pi * Da / rs
    return Rval, l_A

# Planck 2018 (base LCDM) distance-prior means and covariance (R, l_A, Omega_b h^2)
_R_mean,  _sig_R     = 1.7502,  0.0046
_lA_mean, _sig_lA    = 301.471, 0.0895
_Obh2_mean, _sig_Obh2 = 0.02236, 0.00015

_corr_DP = np.array([
    [ 1.00,  0.46, -0.66 ],
    [ 0.46,  1.00, -0.33 ],
    [-0.66, -0.33,  1.00 ],
])
_sigmas_DP = np.array([_sig_R, _sig_lA, _sig_Obh2])
_cov_DP = np.outer(_sigmas_DP, _sigmas_DP) * _corr_DP
_inv_cov_DP = np.linalg.inv(_cov_DP)
_dvec_DP = np.array([_R_mean, _lA_mean, _Obh2_mean])

def chi2_PlanckDP(Om0, H0, w0, wa, Omega_bh2):
    """
    Planck distance priors chi-squared:
    """
    if not bin_range.Planck_distance_priors:
        return 0.0
    R_mod, lA_mod = _dp_compute_R_lA(Om0, H0, w0, wa, Omega_bh2)
    x = np.array([R_mod, lA_mod, Omega_bh2])
    r = x - _dvec_DP
    return float(r @ (_inv_cov_DP @ r))



############################
############################
#
# Dynamic total chi2 builder (auto-selects params)
#
############################
############################ 


def build_total_chi2(which_sne="PantheonPlus"):
    """
    Create a total chi2(theta) callable and the ordered list of parameter names
    based on which datasets are active under the current bin [a,b] and toggles.

    which_sne in {"PantheonPlus", "Union3", "DES"} selects which SN sample to use.
    Returns:
      chi2_total(theta): callable that sums only active datasets
      param_names: list[str] with the exact parameter order expected by chi2_total
    """

    # ---- 1) Detect active datasets under current bin ----
    HAS_CC        = (zbin_CC.size > 0)
    HAS_PANTHEON  = (which_sne == "PantheonPlus") and ('zcmb_bin_Pantheon' in globals()) and (zcmb_bin_Pantheon.size > 0)
    HAS_UNION3    = (which_sne == "Union3")       and ('zcmb_bin_Union3'   in globals()) and (zcmb_bin_Union3.size   > 0)
    HAS_DES_SN    = (which_sne == "DES")          and ('zcmb_bin_DES'      in globals()) and (zcmb_bin_DES.size      > 0)
    HAS_MASER     = (a <= 0.01)                   # tu chi2_maser ya retorna 0 si no aplica
    HAS_DESI      = bool(np.any(mask_DESI))
    HAS_DP        = bool(getattr(bin_range, "Planck_distance_priors", False))

    # Sanity: exactamente un sample SN
    if sum([HAS_PANTHEON, HAS_UNION3, HAS_DES_SN]) != 1:
        raise ValueError("Pick exactly one SN sample via which_sne={'PantheonPlus','Union3','DES'} and ensure it is active in this bin.")

    # ---- 2) Build the parameter ordering ----
    param_names = []

    # (i) Megamasers nuisance velocities if needed
    if HAS_MASER:
        param_names += ["v1","v2","v3","v4","v5","v6"]

    # (ii) Core cosmological parameters (always needed if there is any dataset)
    param_names += ["Om0","H0","w0","wa"]

    # (iii) SN absolute-magnitude offset (M) if any SN dataset is used
    if HAS_PANTHEON or HAS_UNION3 or HAS_DES_SN:
        param_names += ["M"]

    # (iv) DESI BAO sound horizon if DESI is active
    if HAS_DESI:
        param_names += ["rd"]

    # (v) Planck DP Omega_b h^2 if DP is toggled on
    if HAS_DP:
        param_names += ["Omega_bh2"]

    # ---- 3) Pick the SN chi2 to use ----
    def _sn_term(Om0,H0,w0,wa,M):
        if HAS_PANTHEON:
            return chi2_Pantheon(Om0,H0,w0,wa,M)
        if HAS_UNION3:
            return chi2_Union3(Om0,H0,w0,wa,M)
        if HAS_DES_SN:
            return chi2_DES(Om0,H0,w0,wa,M)
        return 0.0  # should not happen

    # ---- 4) Build the callable that unpacks theta by param_names and sums active terms ----
    name_to_idx = {name:i for i,name in enumerate(param_names)}

    def chi2_total(theta):
        # Megamasers
        if HAS_MASER:
            v = [theta[name_to_idx[f"v{i}"]] for i in range(1,7)]
        # Core
        Om0 = theta[name_to_idx["Om0"]]
        H0  = theta[name_to_idx["H0"]]
        w0  = theta[name_to_idx["w0"]]
        wa  = theta[name_to_idx["wa"]]
        # SN offset
        M   = theta[name_to_idx["M"]] if ("M" in name_to_idx) else None
        # DESI r_d
        rd  = theta[name_to_idx["rd"]] if ("rd" in name_to_idx) else None
        # Planck DP Ω_b h^2
        Obh2 = theta[name_to_idx["Omega_bh2"]] if ("Omega_bh2" in name_to_idx) else None

        chi2_sum = 0.0

        if HAS_MASER:
            chi2_sum += chi2_maser(*v, Om0, H0, w0, wa)

        if HAS_CC:
            chi2_sum += chi2_CC(Om0, H0, w0, wa)

        if (HAS_PANTHEON or HAS_UNION3 or HAS_DES_SN) and (M is not None):
            chi2_sum += _sn_term(Om0, H0, w0, wa, M)

        if HAS_DESI and (rd is not None):
            chi2_sum += chi2_DESI(Om0, H0, w0, wa, rd)

        if HAS_DP and (Obh2 is not None):
            chi2_sum += chi2_PlanckDP(Om0, H0, w0, wa, Obh2)

        return float(chi2_sum)

    return chi2_total, param_names