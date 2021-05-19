# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 08:38:51 2020

@author: johnv
"""
from scipy.fft import ifft, ifft2
import numpy as np


# =================================

# Get 1 species CIR generating function using ODE method
def get_gf_CIR_1sp_ODE(g, params):
    gamma, kappa, theta, k = params[0], params[1], params[2], params[3]     # get parameters
    
    max_fudge = 5                                                 # Determine integration time scale / length
    t_max = np.max([1/kappa, 1/gamma])*max_fudge
    min_fudge = 0.1
    dt = np.min([1/kappa, 1/gamma])*min_fudge
    num_tsteps = int(np.ceil(t_max/dt))

    q = np.zeros((g.shape[0], num_tsteps + 1), dtype=np.complex64)    # initialize array to store ODE
    
    # Solve ODE using basic Euler method (seems to be OK)
    for i in range(0, num_tsteps):
        t = i*dt
        q_cur = q[:,i]
        q[:,i+1] = q_cur + dt*( - kappa*q_cur - q_cur*q_cur - kappa*theta*( (g-1)*np.exp(-gamma*t) ) )
    
    integral = np.trapz(q, dx=dt, axis=1)     # integrate ODE solution
    gf = np.exp(- k * integral)               # get generating function
        
    return gf



# Computes Pss for 1 species CIR.
def get_Pss_CIR_1sp_ODE(my_x, params):
    my_g = np.exp(- (2*np.pi*1j)*(my_x/(len(my_x))))         # Choose values of g on unit circle
    gf = get_gf_CIR_1sp_ODE(my_g, params)                     # Get generating function
    Pss = np.real(ifft(gf))  
    
    Pss = np.abs(Pss)/np.sum(np.abs(Pss))
    return Pss



# ======================================


# Get 2 species CIR generating function using ODE method
def get_gf_CIR_2sp_ODE(g0, g1, params):
    beta_0, beta_1, kappa, theta, k = params[0], params[1], params[2], params[3], params[4]     # get parameters
    
    q0 = (g0 - 1) + (beta_0/(beta_1 - beta_0))*(g1 - 1)       # q coefficients
    q1 = - (beta_0/(beta_1 - beta_0))*(g1 - 1)   
    
    max_fudge = 5                                                 # Determine integration time scale / length
    t_max = np.max([1/kappa, 1/beta_0, 1/beta_1])*max_fudge
    min_fudge = 0.1
    dt = np.min([1/kappa, 1/beta_0, 1/beta_1])*min_fudge
    num_tsteps = int(np.ceil(t_max/dt))
    
    
    q = np.zeros((g0.shape[0], g1.shape[1], num_tsteps + 1), dtype=np.complex64)    # initialize array to store ODE
    
    # Solve ODE using basic Euler method (seems to be OK)
    for i in range(0, num_tsteps):
        t = i*dt
        q_cur = q[:,:,i]
        q[:,:,i+1] = q_cur + dt*( - kappa*q_cur - q_cur*q_cur - kappa*theta*( q0*np.exp(-beta_0*t) + q1*np.exp(-beta_1*t)  ) )
    
    integral = np.trapz(q, dx=dt, axis=2)     # integrate ODE solution
    gf = np.exp(- k * integral)               # get generating function
    return gf


# Computes Pss for 2 species CIR.
def get_Pss_CIR_2sp_ODE(X0, X1, params):
    g0 = np.exp(- (2*np.pi*1j)*(X0/(len(X0))))         # Choose values of g on unit circle
    g1 = np.exp(- (2*np.pi*1j)*(X1/(len(X1))))         # Choose values of g on unit circle
    gf = get_gf_CIR_2sp_ODE(g0, g1, params)                     # Get generating function
    Pss = np.real(ifft2(gf))                                  # Get Pss by inverse fast Fourier transform
    
    Pss = np.abs(Pss)/np.sum(np.abs(Pss))
    return Pss

# ====================================================