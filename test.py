#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 22:15:04 2021

@author: fang
"""

import numpy as np
import scipy.stats as ss
import scipy.io as sio
import matplotlib.pyplot as plt
import os
from CIR_functions_ODE import get_Pss_CIR_2sp_ODE, get_Pss_CIR_1sp_ODE


#%% a e s t h e t i c s
w_mean = 2
w_one = 0.5
col_mean = [0,0,0]
col_one = [0.5]*3
col_hist = [0.7]*3
col_theory = [1,0,0]
w_theory = 1.5
fontsize = 12

#%%
titles = ('Intrinsic','Extrinsic','Poisson',)
labels = ('SDE driver','Nascent histogram','Mature histogram','Joint histogram')

names = ('1_intrinsic','2_extrinsic','3_poisson')
names = [os.path.join('output','20201223','CIR_'+i_+'.mat') for i_ in names]

N = len(names)
sz = (3,N)
figsize = (7,4)
fig, ax = plt.subplots(nrows=sz[0],ncols=sz[1],figsize=figsize)

Pss_exts = []
np.random.seed(42)
IND_ = np.random.randint(0,100,6)

for i,name in enumerate(names):
    data = sio.loadmat(name)
    X = data['X_s']
    R = data['SDE_t']
    tvec = data['tvec']
    r = R[IND_[i],:]
    R_mean = data['SDE_mean']
    
    beta = data['beta'][0,0]
    gamma = data['gamma'][0,0]
    kappa = data['kappa'][0,0]
    theta = 1/data['eta'][0,0]
    alpha = data['alpha'][0,0]
    mu = alpha*theta
    
    params_2sp = [beta, gamma, kappa, theta, alpha]     # parameters for 2 species problem
    params_1sp = [beta, kappa, theta,  alpha]             # parameters for 1 species problem (ignoring spliciing) 
    
    # Set up grid on which Pss evaluated
    x_nas = np.arange(np.amax(X[:,0])+5)
    x_mat = np.arange(np.amax(X[:,1])+5)
    X_,Y_ = np.meshgrid(x_nas,x_mat)
        
    Pss = get_Pss_CIR_2sp_ODE(X_, Y_, params_2sp)       # get Pss for 2 species case    
    Pss_1sp = get_Pss_CIR_1sp_ODE(x_nas, params_1sp)       # get Pss for 1 species case

    nat_Poiss_comp = ss.poisson.pmf(x_nas, mu)                 # Poisson limit for comparison
    #neg_binom_comp = ss.nbinom.pmf(x_nas, n=alpha, p=((theta/beta)/(1 + theta/beta))) # Neg binom limit for comparison

    
    ax[0,i].plot(tvec.flatten(), r.flatten(), color = col_one)
    ax[0,i].plot(tvec.flatten(), R_mean.flatten(), color = col_mean)
    ax[0,i].plot([0,data['Tmax']],[mu]*2,color=col_theory,linestyle=(0,(5,10)),linewidth=w_theory)
    

    ax[1,i].plot(x_nas , nat_Poiss_comp, '--', color='b', alpha = 0.75)
    #ax[1,i].plot(x_nas , neg_binom_comp,'--', color='black', alpha = 0.5)
    ax[1,i].plot(x_nas, Pss_1sp, '--', color='r', alpha = 0.75)
    ax[1,i].hist(X[:,0], bins = np.arange(min(X[:,0]), max(X[:,0])+2, 1) - 0.5, density=True, color=col_hist)
    ax[1,i].set_xlim([0,10])#np.amax(X[:,0])+1])
    #ax[1,i].set_yscale('log')

    mat_Poiss_comp = ss.poisson.pmf(x_mat, mu*beta/gamma)  
    ax[2,i].plot(x_mat , mat_Poiss_comp, '--', color='b', alpha = 0.75)
    ax[2,i].plot(x_mat, np.sum(Pss, axis=1), '--', color='r', alpha = 0.75)
    ax[2,i].hist(X[:,1], bins = np.arange(min(X[:,1]), max(X[:,1])+2, 1) - 0.5, density=True, color=col_hist)
    ax[2,i].set_xlim([0,10])#np.amax(X[:,1])+1])
    #ax[2,i].set_yscale('log')


fig.tight_layout()
plt.savefig('./figure/20201223-test.png',dpi=450)

 