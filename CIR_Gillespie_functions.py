#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 14:36:44 2020

@author: Meichen Fang


"""

import numpy as np
from math import log, exp, sqrt, ceil
import matplotlib.pyplot as plt
import scipy.io as sio
import time

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
    l_ = 4*alpha*exph / (sigma_2 * (1-exph))
    d = 4*beta*alpha/sigma_2
    c = sigma_2*(1-exph)/(4*alpha)

    # generate random variables
    if isinstance(random_seed, int):
        np.random.seed(random_seed)
    Z = np.random.normal(loc=0,scale=1,size=[N,2])
    log_U = np.log(np.random.uniform(low=0.0, high=1.0, size=N))
    chi = np.random.gamma(shape=d/2, scale=2,size=N)

    for i in range(N):
        s = l_ * R[i] + 2*log_U[i]
        if s <= 0:
            Y = 0
        else:
            Y = (Z[i,0] + sqrt(s))**2 + Z[i,1]**2
        R[i+1] = c * (chi[i] + Y)

    return T, R

def CIR_Milstein(r0, ts, te, h, args, random_seed = None):
    """
    sample CIR processes using Milstein method 
     
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
        args = (alpha, beta, sigma) is the parameter defining a CIR process: 
            dr(t) = alpha (beta-r(t)) dt+ sigma sqrt{r(t)} dW(t)
    random_seed : int
        set numpy.random.seed
        
    Returns
    -------
    T : ndarray
        time points    
    R : ndarray
        The value off CIR process at time points in T

    """
    alpha, beta, sigma = args
    # initial y_i
    N=int(ceil((te-ts)/h))
    R=np.zeros(N+1)
    T=h*np.arange(N+1)
    R[0]=r0
    
    # generate random variables
    if isinstance(random_seed, int):
        np.random.seed(random_seed)
    dW=np.random.normal(0,np.sqrt(h),N)  

    # update R[i]
    for i in range(N):
        R[i+1]=R[i]+ alpha*(beta-R[i]) * h + sigma * sqrt(R[i]) * dW[i-1] + sigma**2/4 * (dW[i-1]**2-h)

    return T, R

'''
def CIR_int(h, R, k, args):
  """
  sample integration of CIR processes using method shown in Chan and Joshi, 2010

  Input: R, args
  h: time step
  R: R[i] = R(T[i]) (T[i] = i*h)
  k: truncation level
  args: kappa, theta, epsilon as in dr(t) = kappa (theta-r(t)) dt+ epsilon sqrt{r(t)} dW(t)

  Output: IntR
  IntR: IntR[i] = \int_{T_i}^{T_{i+1}} R(t) dt
  """
  # parameters and initiate
  N = len(R)-1
  IntR = np.zeros(N)
  kappa, theta, epsilon = args
  delta = 4*kappa*theta/epsilon**2
  nu = delta/2 - 1
  J = np.arange(1,k+1)
  gamma = ((kappa*h)**2+4*np.pi**2*J**2)/(2*(epsilon*h)**2)
  lambda_star = 16*np.pi**2*J**2/( epsilon*2**h*((kappa*h)**2+4*np.pi**2*J**2) )

  # generate random variables
  U_eta = np.random.uniform(0, 1, size = N)
  Z_LN = np.random.normal(0, 1, size = N)

  # precompute C_n for Bessel random variables
  C = [1]
  D = ( 2*kappa/epsilon**2/ np.sinh(kappa*h/2) )**2

  for i in range(10):
    C.append(D*C[i]/(4*(i+1)*(i+1+nu)))
  
  # for each step
  for i in range(N):
    
    # sample the Bessel random variable η
    x = kappa*h/2
    r = R[i]* R[i+1]
    z = 2*kappa/epsilon**2/np.sinh(x)*np.sqrt(r)
    n = 0
    entry = C[0]
    C_sum = C[0]
    while entry/C_sum >= 0.0001:
      n = n+1
      if n>=10:
        print('C does not converge')
        return 0
      entry = C[n] * np.power(r,n)
      C_sum = C_sum + entry
    print('truncation level of C to approximate p0: ',n)

    p = 1/C_sum
    eta = 0
    p_sum = 0
    while p_sum <= U_eta[i]:
      eta = eta+1
      p = p*z**2/(4*eta*(eta+nu))
      
      p_sum = p_sum + p
    # now eta is the simulated Bessel random variable
  
    gamma_sum = 0
    r = R[i] + R[i+1]
    for j in range(k):
      # generate poisson random variable N_l
      N_l = np.random.poisson(lambda_star[j])
      # # generate gamma random variable Gamma
      Gamma = np.random.gamma(2*eta+N_l+delta/2,1)
      # compute the sum
      gamma_sum = gamma_sum + Gamma/gamma[j]
    
    # sample the lognormal random variable
    # calculate the mean and the variance for the remainder of the truncated series
    coth = (np.exp(2*x)+1)/(np.exp(2*x)-1)
    csch_2 = (2*np.exp(x)/(np.exp(2*x)-1))**2
    mu_X1_star = 1/kappa*coth - h/2*csch_2
    sigma_X1_star_2 = epsilon**2/kappa**3*coth + epsilon**2*h/2/kappa**2*csch_2 - epsilon**2*h**2/2/kappa*coth*csch_2 
    mu_X2_star = epsilon**2/4/kappa**2*(-2+2*x*coth)
    sigma_X2_star_2 = epsilon**4/2/kappa**4*(-2+x*coth+x**2*csch_2)

    sum_1 = np.sum(lambda_star/gamma)
    sum_2 = np.sum(0.5/gamma)
    sum_3 = np.sum(2*lambda_star/gamma**2)
    sum_4 = np.sum(0.5/gamma**2)

    mu_R = r * (mu_X1_star - sum_1) + (4*eta + delta) * (mu_X2_star - sum_2)
    sigma_R_2 = r * (sigma_X1_star_2 - sum_3) + (4*eta + delta) * (sigma_X2_star_2 - sum_4)

    sigma_LN_2 = np.log( sigma_R_2/mu_R**2 + 1 ) 
    mu_LN = np.log(mu_R) - 0.5 * sigma_LN_2
    sigma_LN = np.sqrt(sigma_LN_2)
    
    # results
    IntR[i] = gamma_sum + np.exp( mu_LN + sigma_LN * Z_LN[i] )
     
  return IntR

'''

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
    ro : float
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

def sim_1D(filename, Paras, Ns, h, random_seed = None):    
    """
    run Gillespie_CIR_1D with given parameters
       
    Parameters
    ----------
    filename : string
        file to store the output data:
            Parameter values
            End time
            Time vector
            Runtime
            Number of cells
            step size
            2D array of molecule counts in each cell at each time point (time index, cell index)
            2D array of simulated CIR process (time index, cell index)
    h : float
        time step threshold
    random_seed : int
        set numpy.random.seed
        
    Returns
    -------
    T : ndarray
        Time points  
    X : ndarray
        The value of molecules at time points in T
    R : ndarray
        The value of CIR process
    """
    meta = ('paras: Parameter values: alpha beta sigma_2 c gamma\t' + 'Te: End time\t' + 'T: Time vector\t' + 'runtime: Runtime (seconds)\t'
        + 'Ncell: Number of cells' + 'dt: integration step size'
        + 'X: 3D array of molecule counts in each cell at each time point (time index, cell index, molecules index)'
        + 'R: 2D array of simulated CIR process (time index, cell index)')
    
    # np.random.seed is global
    if isinstance(random_seed, int):
        np.random.seed(random_seed)
        
    Np=len(Paras)
    x0=0
    ts=0
    tes=[]
    dts=[]
    Xss=[];Tss=[];Rss=[]   
    trun = time.time()   
    for i in range(Np):
        args_CIR = Paras[i,0:3]
        gamma = Paras[i,3]
        T = Paras[i,4]
        alpha, beta, sigma_2 = args_CIR
        
        #initial value
        r0=np.random.gamma(2*alpha*beta/sigma_2, sigma_2/(2*alpha), size = Ns)      
            
        Xs=[];Ts=[];Rs=[]    
        te = T/min(alpha, gamma, sigma_2/(2*alpha))
        dt = te/500
        while dt > h:
            dt = dt/2
        tes.append(te)
        dts.append(dt)
        
        for j in range(Ns):
            T, X, Tr, R = Gillespie_CIR_1D(ts, te, dt, r0[j], x0, gamma, args_CIR)
            Ts.append(T)
            Xs.append(X)
            Rs.append(R)

        Tss.append(Ts)
        Xss.append(Xs)
        Rss.append(Rs)

    trun = time.time() - trun 
    print('runtime', trun, 's')
    mdict={'runtime': trun, 'Ncell': Ns, 'Te': tes, 'dt': dts, 'T': np.array(Tss), 'X': np.array(Xss), 'R': np.array(Rss), 'paras': Paras, 'metadata': meta}
    sio.savemat(filename, mdict)
    print('data saved to', filename)
    
    return np.array(Tss), np.array(Xss), np.array(Rss)

def sim_2D(filename, Paras, Ns, h, random_seed = None):  
    """
    run Gillespie_CIR_2D with given parameters
       
    Parameters
    ----------
    filename : string, must be a .mat file
        file to store the output data:
            metadata
            Parameter values
            End time
            Time vector
            Runtime
            Number of cells
            step size
            3D array of molecule counts in each cell at each time point (time index, cell index, molecules index)
            2D array of simulated CIR process (time index, cell index) 
            
    Paras: 2D array
        array storing parameters. Columns: arg_CIRs, c, gamma, T        
    Ns: int
        number of cells/simulations 
    h: float
        time step threshold
    random_seed : int
        set numpy.random.seed
        
    Returns
    -------
    T : ndarray
        Time points  
    X : ndarray
        The value of molecules at time points in T
    R : ndarray
        The value of CIR process
    
    """
    meta = ('paras: Parameter values: alpha beta sigma_2 c gamma\t' + 'Te: End time\t' + 'T: Time vector\t' + 'runtime: Runtime (seconds)\t'
            + 'Ncell: Number of cells' + 'dt: integration step size'
            + 'X: 3D array of molecule counts in each cell at each time point (time index, cell index, molecules index)'
            + 'R: 2D array of simulated CIR process (time index, cell index)')
    
    # np.random.seed is global
    if isinstance(random_seed, int):
        np.random.seed(random_seed)
    # Paras = alpha, beta, sigma, c, gamma 
    Np=len(Paras)
    x0=[0,0]
    ts=0
    tes=[]
    dts=[]    
    Xss=[];Tss=[];Rss=[]      
    trun = time.time()   
    for i in range(Np):
        args_CIR = Paras[i,0:3]
        c, gamma = Paras[i,3:5]
        T = Paras[i,5]
        alpha, beta, sigma_2 = args_CIR
        
        #initial value
        r0 = np.random.gamma(2*alpha*beta/sigma_2, sigma_2/(2*alpha), size = Ns)      
        te = T/min(alpha,  gamma, (2*alpha**2*beta)/sigma_2, sigma_2/(2*alpha))
        dt = te/500
        while dt > h:
            dt = dt/2
        tes.append(te)
        dts.append(dt)
        
        Xs=[];Ts=[];Rs=[]    
        for j in range(Ns):
            T, X, Tr, R = Gillespie_CIR_2D(ts, te, dt, r0[j], x0, c, gamma, args_CIR)
            Ts.append(T)
            Xs.append(X)
            Rs.append(R)

        Tss.append(Ts)
        Xss.append(Xs)
        Rss.append(Rs)
        
    trun = time.time() - trun 
    mdict={'runtime': trun, 'Ncell': Ns, 'Te': tes, 'dt': dts, 'T': np.array(Tss), 'X': np.array(Xss), 'R': np.array(Rss), 'paras': Paras, 'metadata': meta}
    sio.savemat(filename, mdict, do_compression = True)
    
    return np.array(Tss), np.array(Xss), np.array(Rss), trun, tes, dts

def Gillespie_CIR_gen(ts, te, dt, r0, x0, v, a, args_a, args_CIR):
    """
    For general reactions systems with one reation rate being CIR process
    Tried to avoid using np array in the middle...

    t0, te: float, time of start and end.
    dt: time step.
    x0: 1D list, initial system state.
    v: 2D list, the i th row is the stoichiometry vector for reaction i.
    a(r,x,t,args_a): propensity function at time t, dependent on r (CIR process) and state x. 
          Return a 1D list with length = # reactions.
    """

    def waiting_time(R,x,t0,dt,N,a,args_a):
        i=int(np.ceil(t0/dt))
        rv=np.random.uniform(0,1,1) 
        int_next=(i*dt-t0)*( sum(a(R[i],x,t,args_a)) + sum(a(R[i-1],x,t-dt,args_a)) )/2 #trapezoidal rule #trapezoidal rule
        int_sum=-log(rv)
        
        if int_sum<=int_next:
            return i, int_sum/int_next*(i*dt-t0)
        
        while int_sum>int_next:
            int_sum=int_sum-int_next
            i=i+1
            if i>N:
              return 0, dt*N
            int_next=dt*( sum(a(R[i],x,i*dt,args_a)) + sum(a(R[i-1],x,(i-1)*dt,args_a)) )/2 #trapezoidal rule
        
        u=int_sum/int_next*dt
        
        return i, (i-1)*dt-t0+u

    #np.random.seed()    
    N=int(np.ceil((te-ts)/dt))
    Tr, R = CIR(r0, ts, te, dt, args_CIR)
    t=ts 
    
    x=x0 #system state
    T=[]
    X=[]
  
    while t<te:
        T.append(t)
        X.append(x)

        ti,tau=waiting_time(R,x,t,dt,N,a,args_a)

        if ti==0: # next reaction happens after te
            t=te
            break  

        t=t+tau 
        r=np.random.uniform(0,1,1)
        a_cumsum=np.cumsum( (a(R[ti],x,ti*dt,args_a)+a(R[ti-1],x,(ti-1)*dt,args_a))/2 )
        a_normalized=a_cumsum/a_cumsum[-1]
        
        idx=0
        while r>a_normalized[idx]:
            idx=idx+1
            
        x=x+v[idx]

    T.append(t)
    X.append(x)

    return np.array(T), np.array(X), np.array(Tr), np.array(R)



if __name__ == '__main__':
    fig, ax = plt.subplots(4, 1, figsize = (6,18))
    
    ax[0].set_title('test CIR function')
    T, R = CIR(0,0,10,0.001,(10, 20, 0.5))
    ax[0].plot(T, R, 'r.', markersize = 0.1)
    
    ax[1].set_title('test CIR_Milstein function')
    T, R = CIR(0,0,10,0.001,(10, 20, 0.5))
    ax[1].plot(T, R, 'r.', markersize = 0.1)
    
    ax[2].set_title('test Gillespie_CIR_1D function')
    T, X, Tr, R = Gillespie_CIR_1D(0, 10, 0.001, 20, 0, 2, (10, 20, 0.5))
    ax[2].plot(T, X, 'r.-', markersize = 1)
    
    ax[3].set_title('test Gillespie_CIR_2D function')
    T, X, Tr, R = Gillespie_CIR_2D(0, 10, 0.001, 20, [0,0], 1, 2, (10, 20, 0.5))
    ax[3].plot(T, X[:,0], 'r.-', markersize = 1)
    ax[3].plot(T, X[:,1], 'b.-', markersize = 1)
