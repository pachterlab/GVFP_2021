# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 08:38:51 2020

@author: johnv
"""
from scipy.fft import irfft, irfftn
import numpy as np


# =================================


# ODE for 1 species model
def f_1sp(q, t, g, params):
    gamma, kappa, theta, k = params
    
    result = - kappa*q - q*q - kappa*theta*( (g)*np.exp(-gamma*t) )
    return result


# RK4 implementation for 2 species model
def RK4_1sp(q, f, t, g, step_size, param):
    j1 = f(q, t, g, param)
    j2 = f(q + (step_size/2)*j1, t + (step_size/2), g, param)   
    j3 = f(q + (step_size/2)*j2, t + (step_size/2), g, param)   
    j4 = f(q + (step_size)*j3, t + (step_size), g, param)  
    q_new = q + (step_size/6)*(j1 + 2*j2 + 2*j3 + j4)
  
    return q_new


# Get 1 species CIR generating function using ODE method
def get_gf_CIR_1sp_ODE(g, params):
    gamma, kappa, theta, k = params[0], params[1], params[2], params[3]     # get parameters
    
    max_fudge = 10                                                 # Determine integration time scale / length
    t_max = np.max([1/kappa, 1/gamma])*max_fudge
    min_fudge = 0.1
    dt = np.min([1/kappa, 1/gamma])*min_fudge
    num_tsteps = int(np.ceil(t_max/dt))

    q = np.zeros((g.shape[0], num_tsteps + 1), dtype=np.complex64)    # initialize array to store ODE
    
    # Solve ODE using basic Euler method (seems to be OK)
    for i in range(0, num_tsteps):
        t = i*dt
        q[:,i+1] = RK4_1sp(q[:,i], f_1sp, t, g, dt, params)
        #q_cur = q[:,i]
        #q[:,i+1] = q_cur + dt*( - kappa*q_cur - q_cur*q_cur - kappa*theta*( (g)*np.exp(-gamma*t) ) )
    
    integral = np.trapz(q, dx=dt, axis=1)     # integrate ODE solution
    gf = np.exp(- k * integral)               # get generating function
        
    return gf



# Computes Pss for 1 species CIR.
def get_Pss_CIR_1sp_ODE(mx, params):
    half = mx[0]//2 + 1
    l = np.arange(half) 
    u = np.exp(-2j*np.pi*l/mx[0])-1                # Choose values of g on unit circle
       
    gf = get_gf_CIR_1sp_ODE(u, params)             # Get generating function
 
    Pss = irfft(gf, n=mx[0])                       # inverse rFFT
    Pss = np.abs(Pss)/np.sum(np.abs(Pss))          # normalize
    return Pss



# ======================================

# ODE for 2 species model
def f_2sp(q, t, c0, c1, params):
    beta_0, beta_1, kappa, theta, k = params
    
    result = - kappa*q - q*q - kappa*theta*( c0*np.exp(-beta_0*t) + c1*np.exp(-beta_1*t)  ) 
    return result


# RK4 implementation for 2 species model
def RK4_2sp(q, f, t, c0, c1, step_size, param):
    j1 = f(q, t, c0, c1, param)
    j2 = f(q + (step_size/2)*j1, t + (step_size/2), c0, c1, param)   
    j3 = f(q + (step_size/2)*j2, t + (step_size/2), c0, c1, param)   
    j4 = f(q + (step_size)*j3, t + (step_size), c0, c1, param)  
    q_new = q + (step_size/6)*(j1 + 2*j2 + 2*j3 + j4)
  
    return q_new



# Get 2 species CIR generating function using ODE method
def get_gf_CIR_2sp_ODE(g0, g1, params):
    beta_0, beta_1, kappa, theta, k = params     # get parameters
    
    c0 = (g0) + (beta_0/(beta_1 - beta_0))*(g1)       #  relevant linear combinations of g_i
    c1 = - (beta_0/(beta_1 - beta_0))*(g1)   
    
    max_fudge = 10                                                 # Determine integration time scale / length
    t_max = np.max([1/kappa, 1/beta_0, 1/beta_1])*max_fudge
    min_fudge = 0.1
    dt = np.min([1/kappa, 1/beta_0, 1/beta_1])*min_fudge
    num_tsteps = int(np.ceil(t_max/dt))
    
    
    q = np.zeros((g0.shape[0], num_tsteps + 1), dtype=np.complex64)    # initialize array to store ODE
    
    # Solve ODE using RK4 method 
    for i in range(0, num_tsteps):
        t = i*dt
        q[:,i+1] = RK4_2sp(q[:,i], f_2sp, t, c0, c1, dt, params)
        
    integral = np.trapz(q, dx=dt, axis=1)     # integrate ODE solution
    gf = np.exp(- k * integral)               # get generating function
    return gf



# Get Pss for 2 species model (unspliced and spliced) via ODE method
def get_Pss_CIR_2sp_ODE(mx, params):

    # Get generating function argument
    u = []
    half = mx[:]
    half[-1] = mx[-1]//2 + 1
    for i in range(len(mx)):
        l = np.arange(half[i])
        u_ = np.exp(-2j*np.pi*l/mx[i])-1
        u.append(u_)
    g = np.meshgrid(*[u_ for u_ in u], indexing='ij')
    for i in range(len(mx)):
        g[i] = g[i].flatten()
    
    # Get generating function
    gf = get_gf_CIR_2sp_ODE(g[0], g[1], params)                    
    gf = gf.reshape(tuple(half))
    
                              
    Pss = irfftn(gf, s=mx)                        # Get Pss by inverse fast Fourier transform
    Pss = np.abs(Pss)/np.sum(np.abs(Pss))           # Normalize

    return Pss

# ====================================================