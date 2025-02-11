# Example to evaluate the likelihood in each dataset. Each parameter has an assigned value, feel free to change 
# them as you like.

import likelihood 
import bin_range

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
M_pantheon = -19.3     # Absolut magnitude for SNe Ia in pantheon
M_Union3   = 0         # Absolut magnitude for SNe Ia in Union3. See arXiv:2311.12098
                       # for about how they treat the absolute magnitude 
M_DES      = 0         # Absolut magnitude for SNe Ia in DES. See arXiv:2401.02929
                       # for about how they treat the absolute magnitude
rd         = 144       # Sound Horizon 

print("\n\033[1;31mEvaluating the individual chi2 from zmin=\033[0m", bin_range.zmin, "\033[1;31mto zmax=\033[0m", bin_range.zmax)

print("\nChi2 for CC data", likelihood.chi2_CC(Omega_m,H0,w0,wa))

print("\nChi2 for Megamaser data",likelihood.chi2_maser(v1,v2,v3,v4,v5,v6,Omega_m,H0,w0,wa))

print("\nChi2 for Union3 data", likelihood.chi2_Union3(Omega_m,H0,w0,wa,M_Union3))

print("\nChi2 for Pantheon data", likelihood.chi2_Pantheon(Omega_m,H0,w0,wa,M_pantheon))

print("\nChi2 for DES data", likelihood.chi2_DES(Omega_m,H0,w0,wa,M_DES))

print("\nChi2 for BAO-DESI data", likelihood.chi2_DESI(Omega_m,H0,w0,wa,rd))

print("\n\033[1;31mEvaluating the full chi2\033[0m")

print("\nChi2 total for Base+PantheonPlus", likelihood.chi2_Base_PantheonPlus([v1,v2,v3,v4,v5,v6,Omega_m,H0,w0,wa,M_pantheon,rd]))

print("\nChi2 total for Base+Union3", likelihood.chi2_Base_Union3([v1,v2,v3,v4,v5,v6,Omega_m,H0,w0,wa,M_Union3,rd]))

print("\nChi2 total for Base+DES", likelihood.chi2_Base_DES([v1,v2,v3,v4,v5,v6,Omega_m,H0,w0,wa,M_DES,rd]), "\n")