### Late Universe Likelihoods
This code computes the total χ² from a variety of late-universe cosmological datasets. In particular, it includes cosmic chronometers, megamasers from the Megamaser Cosmology Project, BAO from DESI DR2, and Type Ia Supernovae from Pantheon+, Union3, or the Dark Energy Survey (DES).

In addition, it includes the Planck 2018 distance priors (arXiv:1808.05724), implemented following the formulation of the shift parameter R, the acoustic scale, and the baryon density \Omega_b h^2. These priors can be toggled on or off using the Planck_distance_priors flag in `bin_range.py`.

The code allows binning the data within a specified redshift range (z_min, z_max), thereby computing the chi^2 for the data contained in that bin. This reproduces the methodology and results presented in arXiv:2411.00095.

The active parameters in each run are automatically selected based on which datasets fall within the chosen redshift bin (i.e., datasets with no data in that range contribute chi^2 = 0). Nevertheless, the complete cosmological model remains that of CPL (Chevallier–Polarski–Linder)

To produce confidence regions for cosmological parameters, the likelihood functions can be combined with a sampler such as emcee. A complete, ready-to-run example of this procedure is provided in the `emcee.ipynb` notebook, which explains in detail how to build and sample the posterior automatically using the dynamic parameter configuration from `likelihood.build.ipyb`



### Citation

If you are going to use this code, please make sure to reference the work `arXiv:2411.00095`.


### Usage
- Install the python modules: scipy, numpy, os, pandas, astropy
- Download the data
`./data_download.sh`
- Evaluate the likelihoods. For example with
`python evaluation.py`
- To perform MCMC sampling and visualize confidence contours, open and run: 
`emcee.ipynb`


### `data_download.sh`
Use this file to download the SNe Ia data: Pantheon+, Union3, and DES; as well as the BAO-DESI DR2 data, from official repositories.


### `bin_range.py`

Defines the redshift bin by specifying `zmin < z < zmax`. 
Also includes the logical flag `Planck_distance_priors` to toggle inclusion of the Planck 2018 distance priors.


### `cosmology.py`
Contains the cosmological background equations and distance measures implemented within the CPL parametrization of dark energy.


### `likelihood.py`
Main likelihood functions module. Defines individual chi^2 functions for each dataset (cosmic chronometers, megamasers, SNe Ia, BAO, Planck distance priors) and the corresponding combined chi^2 for Base+SNe. The code automatically assigns a null chi^2 to datasets with no data inside the current bin, ensuring dynamic parameter selection. Setting the full redshift range (e.g., 0.01 < z < 2.33) recovers the full, unbinned likelihood.



### Data

Cosmic chronometers: 
Data obtained from the official page of Michele Moresco: https://cluster.difa.unibo.it/astro/CC_data/ . The covariance matrix was formed following arXiv:2003.07362 with the next gitlab repository https://gitlab.com/mmoresco/CCcovariance .


Megamasers:
Taken from Table 1 of arXiv:2001.09213

