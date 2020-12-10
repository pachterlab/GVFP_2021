#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 13:53:34 2020

@author: fang
"""

import pandas as pd
import os.path
import numpy as np
from CIR_Gillespie_functions import Gillespie_CIR_2D
import time
import scipy.io as sio
from multiprocessing import Pool
#import argparse


# multiprocessing wrapper
def Gillespie_CIR_2D_p(args):
    ts, te, dt, r0_, x0, beta, gamma, args_CIR = args
    T, X, Tr, R = Gillespie_CIR_2D(*args)
    idx = np.round(np.linspace(0, te, nT)).astype(int)
    SDE = R[idx]
    return X[-1], SDE

#%%
nT = 500
h = 0.001
nCell = 1000
n_threads = 4
x0 = [0,0]
ts=0  
total_t = 0
meta = ('alpha, eta: shape and rate parameters of gamma distribution' + 'beta, gamma: splicing and degradation rate\t' 
            + 'kappa: mean-reversion rate' + 'T_: simulation timescale end. Tmax = T/min([kappa, gamma, alpha*kappa, eta])' 
            + 'nCell: Number of cells' + 'dt: integration step size' + 'runtime: Runtime (seconds)\t'
            + 'nT: number of time points\t' + 'Tmax: End time\t' + 'tvec: Time vector of SDE\t' 
            + 'X_s: 2D array of molecule counts in each cell at Tmax (nCell, 2)'
            + 'SDE_t: 100 samples of simulated CIR process (100, nT)' + 'SDE_mean: mean of all CIR processes (not SDE_t)')
#%%
if __name__ == "__main__":
    # use another file for command line
    #parser = argparse.ArgumentParser(add_help=True, description='Run CIR 2D simulation')
    #parser.add_argument('--nthreads', '-t', default=4, metavar='n_threads', type=int, action='store_true', help='n_threads')
    #parser.add_argument('--nCell', '-n', default=10000, metavar='nCell', type=int, action='store_true', help='nCell')
    #parser.add_argument('--outdir', '-0', default='data', metavar='outdir', type=str, action='store_true', help='mat data output dir')
    #args = parser.parse_args()

    paras = pd.read_excel('../Gennady/gg_201206_cir_param_vals 2.xlsx', index_col=0, header = 0)
    names = ('1_intrinsic','2_extrinsic','3_poisson','4_fastnoise','5_intermed','6_intermed')
    names = [os.path.join('data','CIR_20201210','CIR_'+i_+'.mat') for i_ in names]
    #%%        
    for i in range(1,7):
        trun = time.time()
        # load parameters
        beta = paras.at[i,'beta'] 
        gamma = paras.at[i,'gamma']
        kappa = paras.at[i,'kappa']  
        eta = paras.at[i,'eta']
        alpha = paras.at[i,'alpha']
        args_CIR = [kappa, alpha/eta, 2*kappa/eta]
        T_ = paras.at[i,'T']
   
        #initial value
        r0 = np.random.gamma(alpha, 1/eta, size = nCell)      
        te = T_/min(kappa, gamma, alpha*kappa, eta)
        tvec = np.linspace(0, te, nT)
        dt = te/(nT-1)
        while dt > h:
            dt = dt/2
            
        # Pool
        input_args = [(ts, te, dt, r0_, x0, beta, gamma, args_CIR) for r0_ in r0]       
        with Pool(n_threads) as pool:      
            X_s, SDE = zip(*pool.map(Gillespie_CIR_2D_p, input_args))
        
        # write data    
        SDE_mean = np.mean(SDE, axis = 0)            
        trun = time.time() - trun 
        total_t = total_t + trun
        mdict={'runtime': trun, 'Ncell': nCell, 'Tmax': te, 'dt': dt, 'T_': T_, 'tvec': tvec,
               'X_s': np.array(X_s), 'SDE_t': np.array(SDE[0:100]), 'SDE_mean': SDE_mean,
               'alpha': alpha, 'eta': eta, 'kappa': kappa, 'beta': beta, 'gamma': gamma,
               'metadata': meta}
        sio.savemat(names[i-1], mdict, do_compression = True)
        print('number of simulations:', nCell)
        print('time:', trun)
    print('total time (min):', total_t/60)
        
