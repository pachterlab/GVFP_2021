# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 20:30:47 2020

@author: johnv
"""

import numpy as np
from scipy.special import factorial as fact
from scipy.fft import ifft





# Helper function for computing CIR generating function. 
# Pochhammer-like symbol (plus signs).
def poch_plus(n, ratio):
    result = 1
    for i in range(1, n+1):
        result = result*(i + ratio)
    return result

# Helper function for computing CIR generating function. 
# Pochhammer-like symbol (minus signs).
def poch_minus(n, ratio):
    result = 1
    for i in range(1, n+1):
        result = result*(i - ratio)
    return result

# Helper function for computing CIR generating function. 
# Coefficient involving the ratio of two infinite series.
def get_B(Q, ratio, n_max):
    num = 0
    for n in range(0, n_max + 1):
        num = num + ((Q**n)*n)/(fact(n)*poch_minus(n, ratio))
    
    denom = 0
    for n in range(0, n_max + 1):
        denom = denom + ((Q**n)*(n + ratio))/(fact(n)*poch_plus(n, ratio))
    
    result = -num/denom
    return result

# Helper function for computing CIR generating function. 
# Infinite series involving plus sign Pochhammers.
def get_series_plus(Q, ratio, n_max):
    result = 0
    for n in range(0, n_max + 1):
        result = result + ((Q**n))/(fact(n)*poch_plus(n, ratio))      
    return result

# Helper function for computing CIR generating function. 
# Infinite series involving minus sign Pochhammers.
def get_series_minus(Q, ratio, n_max):
    result = 0
    for n in range(0, n_max + 1):
        result = result + ((Q**n))/(fact(n)*poch_minus(n, ratio))    
    return result


# Computes generating function for CIR-prod rate.  Gen func = sum_x   g^x   P(x)
def get_gf_CIR(g, params, n_max):
    gamma, lamb, theta, k = params[0], params[1], params[2], params[3]     # get parameters
    

    Q = -((lamb*theta)/(gamma**2))*(g - 1)
    
    ratio = lamb/gamma
    B = get_B(Q, ratio, n_max)
    
    series_plus = get_series_plus(Q, ratio, n_max)
    series_minus = get_series_minus(Q, ratio, n_max)
    
    my_sum = series_minus + B*series_plus
    
    
    # Final answer
    gf = my_sum**k
    
    return gf



# Computes Pss for CIR-prod rate.
def get_Pss_CIR(my_x, params, n_max):
    my_g = np.exp(- (2*np.pi*1j)*(my_x/(len(my_x))))         # Choose values of g on unit circle
    gf = get_gf_CIR(my_g, params, n_max)                     # Get generating function
    Pss = np.real(ifft(gf))                                  # Get Pss by inverse fast Fourier transform
    
    return Pss
