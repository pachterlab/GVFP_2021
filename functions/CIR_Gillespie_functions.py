#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:36:44 2020

@author: Meichen Fang


"""

import numpy as np
from math import log, exp, sqrt, ceil
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
import scipy.io as sio


def CIR(r0, ts, te, h, args, random_seed = None):
    """
    sample CIR processes using exact method shown in    
    Shao, A. (2012). A fast and exact simulation for CIR process (Doctoral dissertation, University of Florida).
     
    Parameters
    ----------
    ro : float
        Initial value 
    ts : float
        Start time 
    te : float
        End time
    h : float
        time step  
    args : tuple of floats
        args = (alpha, beta, sigma_2) is the parameter defining a CIR process: 
            dr(t) = alpha (beta-r(t)) dt+ sqrt{sigma_2} sqrt{r(t)} dW(t)
    random_seed : int
        set numpy.random.seed
        
    Returns
    -------
    T : ndarray
        time points    
    R : ndarray
        The value off CIR process at time points in T


    """
    
    # initial array and unfold parameters of CIR process
    N=int(ceil((te-ts)/h))
    T=np.arange(ts, te+h, h)
    R=np.zeros(N+1)
    R[0]=r0
    alpha, beta, sigma_2 = args
  
    # precompute parameters
    exph = exp(-alpha*h)
    l_ = 4*alpha*exph / (sigma_2 * (1-exph)) #non-centrality patameter
    d = 4*beta*alpha/sigma_2 #degree of freedom
    c = sigma_2*(1-exph)/(4*alpha)

    # generate random variables
    if isinstance(random_seed, int):
        np.random.seed(random_seed)
    Z = np.random.normal(loc=0,scale=1, size=[N,2])
    log_U = np.log(np.random.uniform(low=0.0, high=1.0, size=N))
    chi = np.random.gamma(shape=d/2, scale=2, size=N)

    for i in range(N):
        s = l_ * R[i] + 2*log_U[i]
        if s <= 0:
            Y = 0
        else:
            Y = (Z[i,0] + sqrt(s))**2 + Z[i,1]**2
        R[i+1] = c * (chi[i] + Y)

    return T, R


def Gillespie_CIR_1D(ts, te, dt, r0, x0, gamma, args_CIR, random_seed = None):
    """
    Gillespie algorithm for the system:
        null -> x: production rate CIR process
        x -> null: degradation rate gamma
     
    Parameters
    ----------
    ts : float
        Start time 
    te : float
        End time
    dt : float
        time step  
    ro : float
        Initial value of CIR
    x0 : float
        Initial value
    gamma: float
        Degradation rate of 
    args : tuple of floats
        args = (alpha, beta, sigma_2) is the parameter defining a CIR process: 
            dr(t) = alpha (beta-r(t)) dt+ sqrt{sigma_2 * r(t)} dW(t)
    random_seed : int
        set numpy.random.seed
        
    Returns
    -------
    T : ndarray
        Time points  
    X : ndarray
        The value of CIR process at time points in T
    Tr : ndarray
        Time points of the CIR process
    R : ndarray
        The value of CIR process at time points in Tr

    """
    # np.random.seed is global
    if isinstance(random_seed, int):
        np.random.seed(random_seed)

    def waiting_time(R,x,t0,dt,N,gamma):
        i=int(ceil(t0/dt))
        rv=np.random.uniform(0,1,1) 
        int_next=(i*dt-t0)*((R[i]+R[i-1])/2+gamma*x) #trapezoidal rule
        int_sum=-log(rv)
        
        if int_sum<=int_next:
            return i, int_sum/int_next*(i*dt-t0)
        
        while int_sum>int_next:
            int_sum=int_sum-int_next
            i=i+1
            if i>N: # next reaction happens after te
              return 0, dt*N
            int_next=dt*((R[i]+R[i-1])/2+gamma*x)
        
        u=int_sum/int_next*dt
        
        return i, (i-1)*dt-t0+u

    #np.random.seed()    
    Tr, R = CIR(r0, ts, te, dt, args_CIR)
    N=int(ceil((te-ts)/dt))
    t=ts
    x=x0 #system state
    v=[1,-1] # stoichiometry vector
    T=[]
    X=[]

    while t<te:
        T.append(t)
        X.append(x)

        ti,tau=waiting_time(R,x,t,dt,N,gamma)

        if ti==0: # next reaction happens after te
            t=te
            break  
            
        t=t+tau    
        r=np.random.uniform(0,1)
        a_cumsum=[(R[ti]+R[ti-1])/2, (R[ti]+R[ti-1])/2+gamma*x]
        a_normalized=a_cumsum/a_cumsum[-1]
        if r<= a_normalized[0]:
            idx=0
        else:
            idx=1
        x=x+v[idx]

    T.append(t)
    X.append(x)

    return np.array(T), np.array(X), Tr, R

def Gillespie_CIR_2D(ts, te, dt, r0, x0, c, gamma, args_CIR, random_seed = None):
    """
     Gillespie algorithm for the system:
        null -> x1: production rate CIR process
        x1 -> x2: splicing rate c
        x2 -> null: degradation rate gamma
    
    Parameters
    ----------
    ts : float
        Start time 
    te : float
        End time
    dt : float
        time step  
    r0 : float
        Initial value of CIR
    x0 : float
        Initial value
    c : float
        transition rate of x0 to x1
    gamma: float
        Degradation rate of 
    args : tuple of floats
        args = (alpha, beta, sigma_2) is the parameter defining a CIR process: 
            dr(t) = alpha (beta-r(t)) dt+ sigma_2 sqrt{sigma_2 * r(t)} dW(t)
    random_seed : int
        set numpy.random.seed
        
    Returns
    -------
    T : ndarray
        Time points  
    X : ndarray
        The value of CIR process at time points in T
    Tr : ndarray
        Time points of the CIR process
    R : ndarray
        The value of CIR process at time points in Tr
    
    """
    # np.random.seed is global
    if isinstance(random_seed, int):
        np.random.seed(random_seed)
        
    def waiting_time(R,x,t0,dt,N,c,gamma):
        i=int(np.ceil(t0/dt))
        rv=np.random.uniform(0,1,1) 
        int_next=(i*dt-t0)*((R[i]+R[i-1])/2+c*x[0]+gamma*x[1]) #trapezoidal rule
        int_sum=-log(rv)
        
        if int_sum<=int_next:
            return i, int_sum/int_next*(i*dt-t0)
        
        while int_sum>int_next:
            int_sum=int_sum-int_next
            i=i+1
            if i>N: # next reaction happens after te
              return 0, dt*N
            int_next=dt*((R[i]+R[i-1])/2+c*x[0]+gamma*x[1])
        
        u=int_sum/int_next*dt
        
        return i, (i-1)*dt-t0+u

    #np.random.seed()    
    N = int(np.ceil((te-ts)/dt))
    Tr, R = CIR(r0, ts, te, dt, args_CIR)
    t = ts
    x = np.array(x0) #system state
    v = [[1,0],[-1,1],[0,-1]]
    T = []
    X = []

    while t<te:
        T.append(t)
        X.append(x)

        ti,tau=waiting_time(R,x,t,dt,N,c,gamma)

        if ti==0: # next reaction happens after te
            t=te
            break  
            
        t=t+tau    
        r=np.random.uniform(0,1)
        a_cumsum=[(R[ti]+R[ti-1])/2, (R[ti]+R[ti-1])/2+c*x[0], (R[ti]+R[ti-1])/2+c*x[0]+gamma*x[1]]
        a_normalized=a_cumsum/a_cumsum[-1]
        
        idx=0
        while r>a_normalized[idx]:
            idx=idx+1
            
        x=x+v[idx]

    T.append(t)
    X.append(x)

    return np.array(T), np.array(X), Tr, R

def Gillespie_CIR_2D_tau(args):
    """
    wrapper function of Gillespie_CIR_2D for multiprocessing
    return X in tvec and CIR process with given index
    
    """
    
    te, dt, r0, x0, beta, gamma, args_CIR, tvec, idx =  args
    T, X, Tr, R = Gillespie_CIR_2D(0, te, dt, r0, x0, beta, gamma, args_CIR)
    SDE = R[idx]
    
    i=len(tvec)-1
    j=len(T)-1
    Xs=np.zeros([len(tvec),2])
    while i>=0:        
        while T[j]>tvec[i]:
            j=j-1
        Xs[i,:]=X[j]
        i=i-1
    
    return Xs, SDE
    
def Gillespie_CIR_2D_data(beta, gamma, kappa, alpha, eta, T, lag, nCell, n_threads, filename):
    meta = ('alpha, eta: shape and rate parameters of gamma distribution\t' + 'beta, gamma: splicing and degradation rate\n' 
            + 'kappa: mean-reversion rate\t' + 'T_: simulation timescale end. Tmax = T/min([kappa, gamma, alpha*kappa, eta])\n' 
            + 'nCell: Number of cells\t' + 'dt: integration step size\t' + 'runtime: Runtime (seconds)\n'
            + 'nT: number of time points\t' + 'Tmax: End time\t' + 'tvec: Time vector of SDE\n' 
            + 'X_s: 2D array of molecule counts in each cell at Tmax (nCell, 2)\n'
            + 'SDE_t: 100 samples of simulated CIR process (100, nT)\t' + 'SDE_mean: mean of all CIR processes (not SDE_t)')
    nT = 500 #CIR sample number
    h = 0.001 #step size threshold
    x0 = [0,0]
    trun = time.time()
    # load parameters
    args_CIR = [kappa, alpha/eta, 2*kappa/eta]
   
    #initial value
    r0 = np.random.gamma(alpha, 1/eta, size = nCell)
    te = T+lag
    tvec_mol = np.linspace(T, te, 100)
    tvec_sde = np.linspace(0, T, nT)
    dt = T/(nT-1)
    idx = 1
    while dt > h:
        dt = dt/2
        idx = idx*2
    idx = idx*np.arange(0,nT)
    idx = idx.astype(int)
    
    # Pool
    input_args = [(te, dt, r0_, x0, beta, gamma, args_CIR, tvec_mol, idx) for r0_ in r0]       
    with Pool(n_threads) as pool:      
        X_s, SDE = zip(*pool.map(Gillespie_CIR_2D_tau, input_args))
    
    # write data    
    SDE_mean = np.mean(SDE, axis = 0)            
    trun = time.time() - trun 
    mdict={'runtime': trun, 'Ncell': nCell, 'Tmax': te, 'dt': dt, 'T': T, 'tvec_mol': tvec_mol, 'tvec_sde': tvec_sde,
           'X_s': np.array(X_s), 'SDE_t': np.array(SDE[0:100]), 'SDE_mean': SDE_mean,
           'alpha': alpha, 'eta': eta, 'kappa': kappa, 'beta': beta, 'gamma': gamma,
           'metadata': meta}
    sio.savemat(filename, mdict, do_compression = True)
    return trun




if __name__ == "__main__":
    kappa,L,eta,beta,gamma = 0.60440829, 0.24280198, 0.17960711, 2.44193867, 0.2120487
    alpha = L/kappa
    T = 23.579489051335848
    lag = 10
    nCell = 10000
    n_threads = 40
    filename = "data/CIR_output/20210122/CIR_7_.mat"
    trun = Gillespie_CIR_2D_data(beta, gamma, kappa, alpha, eta, T, lag, nCell, n_threads, filename)
    print(trun)
