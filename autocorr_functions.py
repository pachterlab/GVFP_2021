# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 17:06:53 2021

@author: johnv
"""
import numpy as np

# =====================

# Autocorrelation for 1 species birth-death, i.e. 0 -> X -> 0
def get_auto_bd_1sp(tau, params):
    gamma, kappa, theta, k = params
    return np.exp(-gamma*tau)

# Autocorrelation for 1 species CIR
def get_auto_CIR_1sp(tau, params):
    gamma, kappa, theta, k = params
    
    prefactor = (gamma*theta)/(gamma + theta + kappa)
    auto = np.exp(-gamma*tau) + prefactor*( (np.exp(-kappa*tau) - np.exp(-gamma*tau) )/(gamma - kappa)  )
    return auto

# Autocorrelation for 2 species birth-death, i.e.   0 -> X0 -> X1 -> 0
def get_auto_bd_2sp(tau, params):
    beta_0, beta_1, kappa, theta, k = params
    
    auto_0 = np.exp(-beta_0*tau)
    auto_1 = np.exp(-beta_1*tau)
    return auto_0, auto_1

# Autocorrelation for 2 species CIR
def get_auto_CIR_2sp(tau, params):
    b0, b1, kappa, theta, k = params
    
    auto_0 = get_auto_CIR_1sp(tau, [b0, kappa, theta, k])
    
    r = (b0 + b1 + kappa)/(b0 + b1)
    denom = (b0 + kappa)*(b1 + kappa) + theta*b0*r
    
    term0 = np.exp(-b1*tau)
    term1 = (b0/(b1 - b0))*((theta*b1*r)/denom)*( np.exp(-b0*tau) - np.exp(-b1*tau) )
    term2 = (b0/(b1 - b0))*((theta*b0*b1)/denom)*( (np.exp(-b1*tau))/(b1 - kappa) - (np.exp(-b0*tau))/(b0 - kappa)  )
    term3 = (b0*theta*b0*b1)/( (b0-kappa)*(b1-kappa)*denom )*np.exp(-kappa*tau)
    
    auto_1 = term0 + term1 + term2 + term3
    
    return auto_0, auto_1