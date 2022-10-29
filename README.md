[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7262328.svg)](https://doi.org/10.5281/zenodo.7262328)

# Overview
This repository contains all of the simulation and analysis code for the manuscript ["Interpretable and tractable models of transcriptional noise for the rational design of single-molecule quantification experiments"](https://www.biorxiv.org/content/10.1101/2021.09.06.459173v4), a discussion of the actionable differences induced by choice of stochastic continuous transcriptional model. We investigate a two-species model with a time-dependent transcription rate, splicing, and degradation. The time-dependent transcription rate is described by the Cox-Ingersoll-Ross (CIR) or Gamma Ornstein-Uhlenbeck (Γ-OU) models.

# Repository Contents

* `figure_2_notebook.ipynb`, `figure_3_notebook.ipynb` and `figure_4_notebook.ipynb`: Code to reproduce the demonstrations of limiting cases and inferential performance.

* `figure_2_notebook_colab.ipynb`, `figure_3_notebook_colab.ipynb` and `figure_4_notebook_colab.ipynb`: Code to reproduce the demonstrations of limiting cases and inferential performance, adapted for Google Colaboratory.

* `figure_4_data_4pMC.ipynb` and `figure_4_BF.ipynb`: Code to generate the Bayes factors displayed in figure 4.

* `figure_SI_extfrac_notebook.ipynb` and `figure_SI_extfrac_notebook_colab.ipynb`: Code to generate the extrinsic noise fraction figure from the SI.

* `figure_3_alt_notebook.ipynb` and `figure_3_alt_notebook_colab.ipynb`: Code to generate the alternative versions of Figure 3a from the SI.

* `gg210824_gou_cir.ipynb`: Code to reproduce the other figures and results in the supplement. 

* `gg220316_sim_demo.ipynb`: Code to demonstrate the simulation procedure at runtime.

* `data/`: All pre-computed simulated data and parameters used for simulation.
  * `CIR_X_NAME.mat`: results for CIR model simulations.
  * `gou_X_NAME.mat`: results for Γ-OU model simulations.

* `fig/`: Figures generated by the notebooks.

* `fits/`: Outputs of the *Monod* package, as well as the likelihood ratio computation procedure. The most up-to-date version of *Monod* is available at <https://github.com/pachterlab/monod>.

* `functions/`: All functions that may be used to generate and analyze the data. Note that individual notebooks may have slight variations in the way the CIR and Γ-OU models are calculated, which correspond to slightly different quadrature rules for differing computational requirements.
  * `CIR_Gillespie_functions.py`: CIR model simulation (Python).
  * `gg_210207_gillespie_gou_oct_1.m`: Γ-OU model simulation (MATLAB, may be used in Python through an Octave interface).
  * `autocorr_functions.py`: analytical solutions for model autocorrelation functions.
  * `CIR_functions.py`: analytical solutions for CIR model distributions.
  * `GOU_functions.py`: analytical solutions for Γ-OU model distributions.

* `preprocessing/`: All scripts to generate kallisto|bustools references and construct count matrices. We mirror the cluster annotation metadata in the current respository.

* `loom/`: loom files used for gradient descent fits and MCMC fits, subsetted to 80 genes of interest. These files were generated by processing the [raw data](http://data.nemoarchive.org/biccn/grant/u19_zeng/zeng/transcriptome/scell/10x_v3/mouse/raw/MOp/) with [kallisto|bustools](https://www.kallistobus.tools/) (with the `--lamanno` setting for the `kb ref` and `kb count` commands), then extracting barcodes corresponding to [annotated clusters](http://data.nemoarchive.org/biccn/grant/u19_zeng/zeng/transcriptome/scell/10x_v3/mouse/processed/analysis/10X_cells_v3_AIBS/). The cell type loom files (`allen_LIBRARY_Glutamatergic`) comprise all glutamatergic cells from the corresponding datasets, less those with fewer than 10,000 UMIs. The full count data are located in the Zenodo package 10.5281/zenodo.7262328.

* `smc_results/`:  Outputs of Sequential Monte Carlo sampling (pyMC). The pickle files storing traces have the name convention `allen_LIBRARY_Glutamatergic_GENE_MODEL_trace.pickle`.

# Software and Hardware Requirements

The simulated data analyses in Figures 2-3 do not require any specific software or hardware, aside from a modern browser: all components have been tested on Google Colaboratory. This environment provides 2 CPUs (2.30GHz), 25 GB disk space, and 12 GB RAM. 

The real data analysis in Figure 4 was performed using up to 35 CPUs (3.7GHz each) on a dedicated server. 

# Installation Guide

The analysis does not require installation: all notebooks can be immediately opened and run in Google Colaboratory. The notebook preamble automatically imports all necessary custom code and packages. The notebooks have been most recently tested with the following package versions:
```
python 3.7.12
scipy 1.4.1
numpy 1.21.5
oct2py 5.4.3
octave-kernel 0.34.1
matplotlib 3.2.2
parfor 2022.3.0
Theano-PyMC 1.1.2
arviz 0.11.4
pymc3 3.11.4
loompy 3.0.7
numdifftools 0.9.40
anndata 0.8.0
monod 0.2.4.0
```
Installing these dependencies typically takes under one minute.

# Use demonstrations

The expected outputs for all notebooks are contained in the compiled and pre-run notebooks contained in the directory. The Figure 3 and simulation notebooks typically have small differences, as they use stochastic procedures for MCMC sampling and simulation.

## Figure 2

To reproduce the distributions and transcription rate time-series demonstrated in Figure 2, open `figure_2_notebook_colab.ipynb`, click "Open in Colab", and select "Runtime &rarr; Run all". This typically takes under 20 seconds and should exactly reproduce Figure 2. 

## Figure 3

To reproduce the distributions, parameter posteriors, Bayes factor landscapes, and identifiability trends demonstrated in Figure 3, open `figure_3_notebook_colab.ipynb`, click "Open in Colab", and select "Runtime &rarr; Run all". Note that this notebook is not deterministic, as the model sampling and MCMC chains are intrinsically random. This notebook typically takes under 25 minutes to complete. 

## Figure 4

Figure 4 requires fairly computationally intensive analyses, and has been split so parts can be reproduced separately.

To reproduce the count matrices, run the scripts under `preprocessing/ref/` to download and build the genome reference. Then, download the datasets from NeMO and run the `count_allen_X.sh` scripts to quantify the unspliced and spliced counts. This step may take several hours.

To prepare the main notebook, open `figure_4_notebook.ipynb` and run all cells above "Predictive filtering: preprocessing." This has to be done before any other downstream analysis.

To reproduce the primary *Monod* workflow, which fits a set of glutamatergic clusters to the limiting regimes of the CIR and Γ-OU models, run all cells above "Predictive filtering: analysis of *Monod* results and AIC computation". This step should take under 3 hours. Lower-fidelity fits with satisfactory quality can be performed by reducing the `max_iterations` and `num_restarts` parameters.

To reproduce the secondary *Monod* workflow, which uses those fits and Akaike weights to generate candidate genes and Figure 4a, run all cells between "Predictive filtering: analysis of Monod results and AIC computation" and "Out-of-sample analysis". This step should take under 3 minutes.

To reproduce the full model gradient estimation and likelihood ratio computation, run all cells between "Out-of-sample analysis" and "SDE goodness of fit". This step should take under 32 hours using 15 restarts and a maximum of 20 iterations on 35 cores. Lower-fidelity fits with largely satisfactory quality can be performed by reducing the number of restarts and iterations (the integers in the final two arguments of the `zip` call in this section). However, low-fidelity fits will occasionally fail and converge to suboptimal, uninterpretable parameter values.

To reproduce the *post hoc* goodness-of-fit testing and generate the rest of Figure 4, run all cells after "SDE goodness of fit". The generation of Figure 4b will automatically load in the Bayes factor results. This step should take under 4 minutes. 

To reproduce the MCMC fits to 12 genes of interest, open `figure_4_data_4pMC.ipynb` and run all cells above "BF plots". This step typically takes several hours, up to a day. 

To compute the Bayes factors from the MCMC fits, open `figure_4_BF.ipynb` and run the notebook. This step should take under one minute.

## Supplementary figures

To reproduce the simulated stationary distributions and autocorrelation functions reported in the supplement, as well as several exploratory analyses, open `gg210824_gou_cir.ipynb`, click "Open in Colab", navigate to "Finding the maximally divergent parameter set", and select "Runtime &rarr; Run before". Then, navigate to "Analyzing the maximally divergent parameter set" and select "Runtime &rarr; Run after". This notebook typically takes under 4 minutes to complete and should exactly reproduce the supplementary figures. The procedure omits the time-consuming, stochastic gradient optimization procedure used to calculate the illustrative "maximally divergent" parameter values in Figure 3d. However, it is straightforward to run a limited version of this search by setting the `nsearch` variable to 2 (typically taking 2 minutes). In the demonstration, we omit the simulation of this distribution. To reproduce the extrinsic noise fraction behaviors, open `figure_SI_extfrac_notebook_colab.ipynb` and run the notebook. To reproduce the alternative versions of Figure 3a from the SI, open `figure_3_alt_notebook_colab.ipynb` and run the notebook.

## Simulation
Simulations used in the manuscript typically use 10,000 independent cells. To facilitate inspection of the simulation procedure and results, we have set up a small notebook environment that generates new data for 1,000 cells, for arbitrary parameters. To use it, open `gg220316_sim_demo.ipynb`, click "Open in Colab", and select "Runtime &rarr; Run all". The parameters are defined by the variables `kappa`, `L`, `eta` (SDE parameters corresponding to κ, α κ, and 1/θ), `beta`, `gamma` (CME parameters corresponding to β and γ), and `T`, `lag` (simulation durations up to equilibrium and past it, corresponding to T_ss and T_R). The procedure outputs result files `gou_8_sample.mat` and `CIR_8_sample.mat`, containing data structures analogous to the pre-computed files provided in the `data/` directory. This notebook typically takes approximately 7 minutes to complete. 
