#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 13:53:34 2020

@author: fang
"""

import pandas as pd
import os
import numpy as np
from CIR_Gillespie_functions import Gillespie_CIR_2D_tau
import time
import scipy.io as sio
from multiprocessing import Pool
import argparse


nT = 500
h = 0.001
x0 = [0,0]
Ts=[20,200,60,7.14285714285714,7.39098300073910,7.14285714285714]
ts=0  
total_t = 0
meta = ('alpha, eta: shape and rate parameters of gamma distribution\t' + 'beta, gamma: splicing and degradation rate\n' 
            + 'kappa: mean-reversion rate\t' + 'T_: simulation timescale end. Tmax = T/min([kappa, gamma, alpha*kappa, eta])\n' 
            + 'nCell: Number of cells\t' + 'dt: integration step size\t' + 'runtime: Runtime (seconds)\n'
            + 'nT: number of time points\t' + 'Tmax: End time\t' + 'tvec: Time vector of SDE\n' 
            + 'X_s: 2D array of molecule counts in each cell at Tmax (nCell, 2)\n'
            + 'SDE_t: 100 samples of simulated CIR process (100, nT)\t' + 'SDE_mean: mean of all CIR processes (not SDE_t)')
#%%
if __name__ == "__main__":   
    parser = argparse.ArgumentParser(add_help=True, description='Run CIR 2D simulation')
    parser.add_argument("-t", "--n_threads", default=4, type=int, action='store', help='n_threads')
    parser.add_argument("-n", "--nCell",  default=1000, type=int, action='store', help='nCell')
    parser.add_argument("-o", "--outdir", default="output", type=str, action='store', help='output dir')
    args = parser.parse_args()

    paras = pd.read_csv('data/param_vals.txt', sep='\t', index_col=0, header = 0)
    names = ('1_intrinsic','2_extrinsic','3_poisson')#,'4_fastnoise','5_intermed','6_intermed')
    names = [os.path.join(args.outdir,'CIR_'+i_+'.mat') for i_ in names]
    
    if not os.path.isdir(args.outdir):
        os.mkdir(args.outdir)
    #%%        
    for i in range(1,4):
        trun = time.time()
        # load parameters
        beta = paras.at[i,'beta'] 
        gamma = paras.at[i,'gamma']
        kappa = paras.at[i,'kappa']  
        eta = paras.at[i,'eta']
        alpha = paras.at[i,'alpha']
        args_CIR = [kappa, alpha/eta, 2*kappa/eta]
        T = Ts[i-1]#T = paras.at[i,'T']
   
        #initial value
        r0 = np.random.gamma(alpha, 1/eta, size = args.nCell)      
        te = T+10
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
        with Pool(args.n_threads) as pool:      
            X_s, SDE = zip(*pool.map(Gillespie_CIR_2D_tau, input_args))
        
        # write data    
        SDE_mean = np.mean(SDE, axis = 0)            
        trun = time.time() - trun 
        total_t = total_t + trun
        mdict={'runtime': trun, 'Ncell': args.nCell, 'Tmax': te, 'dt': dt, 'T': T, 'tvec_mol': tvec_mol, 'tvec_sde': tvec_sde,
               'X_s': np.array(X_s), 'SDE_t': np.array(SDE[0:100]), 'SDE_mean': SDE_mean,
               'alpha': alpha, 'eta': eta, 'kappa': kappa, 'beta': beta, 'gamma': gamma,
               'metadata': meta}
        sio.savemat(names[i-1], mdict, do_compression = True)
        print(names[i-1],' time(s): ', trun)
    print('total time (min):', total_t/60)
        
