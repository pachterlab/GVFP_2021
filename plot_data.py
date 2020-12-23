#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 11:13:23 2020

@author: fang
"""
import numpy as np
import scipy.stats as ss
import scipy.io as sio
import matplotlib.pyplot as plt
import os
#from CIR_functions_ODE import get_Pss_CIR_2sp_ODE, get_Pss_CIR_1sp_ODE


#%% a e s t h e t i c s
w_mean = 2
w_one = 0.5
col_mean = [1,0,0]
col_one = [0.5]*3
col_hist = [0.7]*3
scatter_col = [0.2]*3
scatter_size = 0.1
scatter_alpha = 0.3
col_theory = [1,0,0]
w_theory = 1.5
fontsize = 12

#%%
titles = ('Intrinsic','Extrinsic','Poisson','Fast-noise','Intermed. 1','Intermed. 2')
labels = ('SDE driver','Nascent histogram','Mature histogram','Joint histogram')

names = ('1_intrinsic','2_extrinsic','3_poisson','4_fastnoise','5_intermed','6_intermed')
names = [os.path.join('data','CIR_20201223','CIR_'+i_+'.mat') for i_ in names]

N = len(names)
sz = (4,N)
figsize = (14,8)
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
        
    #Pss = get_Pss_CIR_2sp_ODE(X_, Y_, params_2sp)       # get Pss for 2 species case    
    #Pss_1sp = get_Pss_CIR_1sp_ODE(x_nas, params_1sp)       # get Pss for 1 species case

    #Poiss_comp = ss.poisson.pmf(my_x, mu)                 # Poisson limit for comparison
    #neg_binom_comp = ss.nbinom.pmf(my_x, n=params[3], p=((theta/beta)/(1 + theta/beta))) # Neg binom limit for comparison

    
    ax[0,i].plot(tvec.flatten(), r.flatten(), '.', color = col_one)
    ax[0,i].plot(tvec.flatten(), R_mean.flatten(), color = col_mean)
    
'''
    #ax[1,i].plot(my_x.flatten(), Poiss_comp.flatten(), '--', color='b', alpha = 0.75)
    #ax[1,i].plot(my_x.flatten(), neg_binom_comp.flatten(),'--', color='black', alpha = 0.5)
    ax[1,i].plot(x_nas, np.sum(Pss, axis=0), '--', color='b', alpha = 0.75)
    ax[1,i].plot(x_nas, Pss_1sp, '--', color='r', alpha = 0.75)
    ax[1,i].hist(X[:,0], bins = np.arange(min(X[:,0]), max(X[:,0])+2, 1) - 0.5, density=True, color=col_hist)
    ax[1,i].set_yscale('log')
    ax[1,i].set_xlim([0,np.amax(X[:,0])+1])

    
    ax[2,i].plot(x_mat, np.sum(Pss, axis=1), '--', color='b', alpha = 0.75)
    ax[2,i].hist(X[:,1], bins = np.arange(min(X[:,1]), max(X[:,1])+2, 1) - 0.5, density=True, color=col_hist)
    ax[2,i].set_yscale('log')
    ax[2,i].set_xlim([0,np.amax(X[:,1])+1])
    
    ###
    nCells = X.shape[0]
    noise = 1+np.random.randn(nCells,2)/20

    nas_ = (X[:,0]+1)*noise[:,0]
    filt = np.logical_and(X[:,0]==0, nas_<1)
    nas_[filt] = 2-nas_[filt]

    mat_ = (X[:,1]+1)*noise[:,1]
    filt = np.logical_and(X[:,1]==0, mat_<1)
    mat_[filt] = 2-mat_[filt]

    
    ax[3,i].pcolormesh(X_+0.5, Y_+0.5, Pss, cmap='summer')  
    ax[3,i].scatter(nas_,mat_,color=scatter_col,s=scatter_size,alpha=scatter_alpha,edgecolors=None)
    ax[3,i].set_xscale('log')
    ax[3,i].set_yscale('log')
    ax[3,i].set_xlim([1,np.amax(X[:,0])+1])
    ax[3,i].set_ylim([1,np.amax(X[:,1])+1])
    
for j_ in range(4):
    ax[j_,0].set_ylabel(labels[j_],fontsize=fontsize)
for j_ in range(N):
    ax[0,j_].set_title(titles[j_],fontsize=fontsize)

for a in ax:
    for b in a:
        # b.axis('off')
        b.set_xticks([])
        b.set_yticks([])
        b.set_xticks([],minor=True)
        b.set_yticks([],minor=True)
'''
fig.tight_layout()
plt.savefig('./figure/CIR_20201211.png',dpi=450)

 