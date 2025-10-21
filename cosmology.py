from scipy import integrate
import numpy as np


######################
#
# Cosmological functions
#
######################

#CPL Model 
c = 299792.458 #km/s                                                           # Speed of light 


def f(z,w0,wa):
    return ((z+1)**(3*(w0+wa+1)))*np.exp((-3*wa*z)/(z+1))                      # CPL parameterization 


def H(z,Om0,H0,w0,wa):
    return (H0)*np.sqrt(Om0*(1+z)**3 + (1-Om0)*f(z,w0,wa))                     # Friedmann equation for the CPL model


def integrand(z,Om0,H0,w0,wa):
    return 1/H(z,Om0,H0,w0,wa)


def DA(z,Om0,H0,w0,wa):                                                        # Angular diameter distance 
    return (c/(1+z))*integrate.quad(integrand,0,z,args=(Om0,H0,w0,wa))[0]    


DA=np.vectorize(DA,otypes=[np.float64])


def DM(z,Om0,H0,w0,wa):                                                        # Comoving angular diameter distance
    return (1+z)*DA(z,Om0,H0,w0,wa)                                           


def DH(z,Om0,H0,w0,wa):                                                        # Hubble distance 
    return c*integrand(z,Om0,H0,w0,wa)  


def Dv(z,Om0,H0,w0,wa):
    return (z*(DM(z,Om0,H0,w0,wa)**2)*c*integrand(z,Om0,H0,w0,wa))**(1/3)     # Spherically averaged distance


def Dl(z,Om0,H0,w0,wa):                                                        # Luminosity distance 
    return ((1+z)**2)*DA(z,Om0,H0,w0,wa)


def mu(z,Om0,H0,w0,wa):                                                         # Distance modulus 
    return 5*np.log10(Dl(z,Om0,H0,w0,wa))+25 


def mb(z,Om0,H0,w0,wa,M):                                                      # Apparent magnitude of SNe 
       return mu(z,Om0,H0,w0,wa) + M                                    

