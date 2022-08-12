[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6363751.svg)](https://doi.org/10.5281/zenodo.6363751)

# Overview
This repository contains all of the simulation and analysis code for the manuscript ["Interpretable and tractable models of transcriptional noise for the rational design of single-molecule quantification experiments"](https://www.biorxiv.org/content/10.1101/2021.09.06.459173v4), a discussion of the actionable differences induced by choice of stochastic continuous transcriptional model. We investigate a two-species model with a time-dependent transcription rate, splicing, and degradation. The time-dependent transcription rate is described by the Cox-Ingersoll-Ross (CIR) or Gamma Ornstein-Uhlenbeck (Γ-OU) models.

# Repository Contents

* `figure_2_notebook.ipynb`, `figure_3_notebook.ipynb` and `figure_4_notebook.ipynb`: Code to reproduce the demonstrations of limiting cases and inferential performance.

* `figure_2_notebook_colab.ipynb`, `figure_3_notebook_colab.ipynb` and `figure_4_notebook_colab.ipynb`: Code to reproduce the demonstrations of limiting cases and inferential performance, adapted for Google Colaboratory.

* `figure_4_data_4pMC.ipynb` and `figure_4_data_grid.ipynb`: Code to generate the data used in figure 4.

* `gg210824_gou_cir.ipynb`: Code to reproduce the figures and results in the supplement. 

* `gg220316_sim_demo.ipynb`: Code to demonstrate the simulation procedure at runtime.

* `data/`: All pre-computed simulated data and parameters used for simulation.
  * `CIR_X_NAME.mat`: results for CIR model simulations.
  * `gou_X_NAME.mat`: results for Γ-OU model simulations.

* `fig/`: Figures generated by the notebooks.

* `functions/`: All functions used to generate and analyze the data.
  * `CIR_Gillespie_functions.py`: CIR model simulation.
  * `gg_210207_gillespie_gou_oct_1.m`: Γ-OU model simulation.
  * `autocorr_functions.py`: analytical solutions for model autocorrelation functions.
  * `CIR_functions.py`: analytical solutions for CIR model distributions.
  * `GOU_functions.py`: analytical solutions for Γ-OU model distributions.

# Software and Hardware Requirements

The analysis does not require any specific software or hardware, aside from a modern browser: all components have been tested on Google Colaboratory. This environment provides 2 CPUs (2.30GHz), 25 GB disk space, and 12 GB RAM. 

# Installation Guide

The analysis does not require installation: all notebooks can be immediately opened and run in Google Colaboratory. The notebook preamble automatically imports all necessary custom code and packages. The notebooks have been most recently tested with the following package versions:
```
python 3.7.12
scipy 1.4.1
numpy 1.21.5
oct2py 5.4.3
octave-kernel 0.34.1
matplotlib 3.2.2
multiprocess 0.70.12.2
parfor 2022.3.0
Theano-PyMC 1.1.2
arviz 0.11.4
pymc3 3.11.4
```
Installing these dependencies typically takes under one minute.

# Use demonstrations

The expected outputs for all notebooks are contained in the compiled and pre-run notebooks contained in the directory. The Figure 3 and simulation notebooks typically have small differences, as they use stochastic procedures for MCMC sampling and simulation.

## Figure 2

To reproduce the distributions and transcription rate time-series demonstrated in Figure 2, open `figure_2_notebook_colab.ipynb`, click "Open in Colab", and select "Runtime &rarr; Run all". This typically takes under 20 seconds and should exactly reproduce Figure 2. 

## Figure 3

To reproduce the distributions, parameter posteriors, Bayes factor landscapes, and identifiability trends demonstrated in Figure 3, open `figure_3_notebook_colab.ipynb`, click "Open in Colab", and select "Runtime &rarr; Run all". Note that this notebook is not deterministic, as the model sampling and MCMC chains are intrinsically random. This notebook typically takes under 25 minutes to complete. 

## Supplementary figures

To reproduce the stationary distributions and autocorrelation functions reported in the supplement, as well as several exploratory analyses, open `gg210824_gou_cir.ipynb`, click "Open in Colab", navigate to "Finding the maximally divergent parameter set", and select "Runtime &rarr; Run before". Then, navigate to "Analyzing the maximally divergent parameter set" and select "Runtime &rarr; Run after". This notebook typically takes under 4 minutes to complete and should exactly reproduce the supplementary figures. The procedure omits the time-consuming, stochastic gradient optimization procedure used to calculate the illustrative "maximally divergent" parameter values in Figure 3d. However, it is straightforward to run a limited version of this search by setting the `nsearch` variable to 2 (typically taking 2 minutes). In the demonstration, we omit the simulation of this distribution.

## Simulation
Simulations used in the manuscript typically use 10,000 independent cells. To facilitate inspection of the simulation procedure and results, we have set up a small notebook environment that generates new data for 1,000 cells, for arbitrary parameters. To use it, open `gg220316_sim_demo.ipynb`, click "Open in Colab", and select "Runtime &rarr; Run all". The parameters are defined by the variables `kappa`, `L`, `eta` (SDE parameters corresponding to κ, α κ, and 1/θ), `beta`, `gamma` (CME parameters corresponding to β and γ), and `T`, `lag` (simulation durations up to equilibrium and past it, corresponding to T_ss and T_R). The procedure outputs result files `gou_8_sample.mat` and `CIR_8_sample.mat`, containing data structures analogous to the pre-computed files provided in the `data/` directory. This notebook typically takes approximately 7 minutes to complete. 
