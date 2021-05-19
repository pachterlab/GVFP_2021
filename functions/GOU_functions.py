import numpy as np
from numpy import matlib

import scipy 
import scipy.integrate
from CIR_functions import *


def compute_Pss(par,topclip=np.inf):
    kappa,L,eta,splic,gamma = par
    alpha = L/kappa
    mu_k = alpha/eta
    m = np.array([mu_k/splic,mu_k/gamma])
    std = np.sqrt(np.array([m[0]*(1+1/(eta*(kappa+splic))), m[1]*(1+splic*(kappa+splic+gamma)/(eta*(kappa+splic)*(kappa+gamma)*(splic+gamma)))]))
    mx = np.clip(np.ceil(m+std*5),8,topclip)
    
    
    Pss_gou = np.squeeze(cme_integrator(L,1/eta/kappa,gamma,[kappa,splic],[1,int(mx[0]),int(mx[1])],np.inf))
    Pss_cir = np.squeeze(get_Pss_CIR_2sp_ODE([int(mx[0]),int(mx[1])], [splic,gamma,kappa,1/eta,alpha]))
    return Pss_gou,Pss_cir,mx

def get_1sp_sb(tau,params):
    gamma, beta, b = params
    prefactor = (gamma*beta*b)/((1+b)*beta+gamma)

    auto = np.exp(-gamma*tau) + prefactor*( (np.exp(-beta*tau) - np.exp(-gamma*tau) )/(gamma - beta)  )
    return auto

def cme_integrator(kini,b,g,beta,dims,t):
    l1 = np.arange(dims[0])
    l2 = np.arange(dims[1])
    l3 = np.arange(dims[2])
    u1_ = np.exp(-2j*np.pi*l1/dims[0])-1;
    u2_ = np.exp(-2j*np.pi*l2/dims[1])-1;
    u3_ = np.exp(-2j*np.pi*l3/dims[2])-1;
    U1,U2,U3 = np.meshgrid(u1_,u2_,u3_,indexing='ij')
    u=np.vstack((U1.flatten(),U2.flatten(),U3.flatten())).T
    
    rates = np.asarray([beta[0],beta[1],g])
    
    coeff = coeff_calc(rates,u)
    fun = lambda x: INTFUNC(x,b,coeff,rates)
    INT = scipy.integrate.quad_vec(fun,0,t)
    I = np.exp(INT[0]*kini)
    I = np.reshape(I,dims)
    return np.real(np.fft.ifftn(I))

def coeff_calc(rates,u):
    d = u.shape[0]
    Nsteps = len(rates)
    coeff = np.zeros((d,Nsteps),dtype=u.dtype)
    coeff[:,-1] = u[:,-1]
    for i in range(Nsteps-2,-1,-1):
        f = rates[i]/(rates[i]-rates[(i+1):])
        coeff[:,(i+1):] *= f
        coeff[:,i] = u[:,i] - np.sum(coeff[:,(i+1):],1)
    return coeff

def INTFUNC(x,b,coeff,rates):
    Ufun = np.sum(coeff*np.exp(-rates*x),1)
    Ufun = b*Ufun
    F = Ufun/(1-Ufun)
    return F


def divg(data,proposal,kind='kl'):
    EPS = 1e-16
    proposal[proposal<EPS]=EPS
    if kind=='kl':
        filt = data>0
        data = data[filt]
        proposal = proposal[filt]
        d=data*np.log(data/proposal)
        return np.sum(d)
    if kind=='ks':
        data_cdf = np.cumsum(data)
        proposal_cdf = np.cumsum(proposal)
        return np.max(np.abs(data_cdf-proposal_cdf))