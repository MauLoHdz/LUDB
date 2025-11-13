# Example to evaluate the likelihood in each dataset. Each parameter has an assigned value, feel free to change 
# them as you like.

import likelihood 
import bin_range
import numpy as np

# Maser galaxy velocities:
v1 = 3319.9
v2 = 10192.
v3 = 7801.5
v4 = 8525.7
v5 = 7172.2
v6 = 679.3

Omega_m    = 0.3       # Matter density parameter 
H0         = 70        # Hubble constant
w0         = -1        # Present-day equation of state of Dark energy (CPL model)
wa         =  0        # EoS evolution parameter in a CPL context
M_pantheon = -19.3     # Absolute magnitude (offset interpretation varies by sample; see notes in your code)
M_Union3   = 0         # See arXiv:2311.12098
M_DES      = 0         # See arXiv:2401.02929
rd         = 144       # Sound Horizon 
Omega_bh2  = 0.02      # Omega_b * h^2 

print("\n\033[1;31mEvaluating the individual chi2 from zmin=\033[0m", bin_range.zmin, "\033[1;31mto zmax=\033[0m", bin_range.zmax)

print("\nChi2 for CC data", likelihood.chi2_CC(Omega_m, H0, w0, wa))

print("\nChi2 for Megamaser data", likelihood.chi2_maser(v1, v2, v3, v4, v5, v6, Omega_m, H0, w0, wa))

print("\nChi2 for Union3 data", likelihood.chi2_Union3(Omega_m, H0, w0, wa, M_Union3))

print("\nChi2 for Pantheon data", likelihood.chi2_Pantheon(Omega_m, H0, w0, wa, M_pantheon))

print("\nChi2 for DES data", likelihood.chi2_DES(Omega_m, H0, w0, wa, M_DES))

print("\nChi2 for BAO-DESI data", likelihood.chi2_DESI(Omega_m, H0, w0, wa, rd))

print("\nChi2 for Planck Distance Priors", likelihood.chi2_PlanckDP(Omega_m, H0, w0, wa, Omega_bh2))

print("\n\033[1;31mEvaluating the full chi2\033[0m")

# Helper: build theta vector in the param order returned by build_total_chi2
def assemble_theta(param_names, which_sne):
    # Common parameter dictionary with provided values
    p = {
        "v1": v1, "v2": v2, "v3": v3, "v4": v4, "v5": v5, "v6": v6,
        "Om0": Omega_m, "H0": H0, "w0": w0, "wa": wa,
        "rd": rd, "Omega_bh2": Omega_bh2,
        # M depends on the SN sample chosen below
        "M": M_pantheon if which_sne == "PantheonPlus" else (M_Union3 if which_sne == "Union3" else M_DES),
    }
    # Only pick those required, in order
    return np.array([p[name] for name in param_names], dtype=float)

# Base + PantheonPlus
chi2_fn_PP, names_PP = likelihood.build_total_chi2(which_sne="PantheonPlus")
theta_PP = assemble_theta(names_PP, which_sne="PantheonPlus")
print("\nChi2 total for Base+PantheonPlus", chi2_fn_PP(theta_PP))

# Base + Union3
chi2_fn_U3, names_U3 = likelihood.build_total_chi2(which_sne="Union3")
theta_U3 = assemble_theta(names_U3, which_sne="Union3")
print("\nChi2 total for Base+Union3", chi2_fn_U3(theta_U3))

# Base + DES
chi2_fn_DES, names_DES = likelihood.build_total_chi2(which_sne="DES")
theta_DES = assemble_theta(names_DES, which_sne="DES")
print("\nChi2 total for Base+DES", chi2_fn_DES(theta_DES), "\n")