[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6361041.svg)](https://doi.org/10.5281/zenodo.6361041)

# GVFP_2021
SDE comparison preprint: A discussion of the differences induced by choice of stochastic continuous noise model for transcription. Includes simulation code and validation notebooks.


## Content

### figure_2_notebook.ipynb and figure_3_notebook.ipynb
Code to reproduce the demonstrations of limiting cases and inferential performance.

### gg210824_gou_cir.ipynb
Code to reproduce the figures and results in the supplement. 

### data
Includes simulation parameters and all simulation data.

### functions
``CIR_functions.py`` includes functions that compute steady state distribution of 2 species CIR based on analytical results.

``GOU_functions.py`` includes functions that compute steady state distribution of 2 species Γ-OU and divergence between two models.

``autocorr_functions.py`` computes the autocorrelation function.

``CIR_Gillespie_functions.py`` implements the Gillespie algorithm for 2 species CIR.

