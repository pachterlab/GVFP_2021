#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 09:05:16 2021

@author: fang
"""

from CIR_Gillespie_functions import Gillespie_CIR_2D_data



kappa,L,eta,beta,gamma = 0.60440829, 0.24280198, 0.17960711, 2.44193867, 0.2120487
alpha = L/kappa
T = 23.579489051335848
lag = 10
nCell = 10000
n_threads = 40
filename = "output/20210122/CIR_7_.mat"

if __name__ == "__main__":      
    trun = Gillespie_CIR_2D_data(beta, gamma, kappa, alpha, eta, T, lag, nCell, n_threads, filename)
    print(trun)
        
