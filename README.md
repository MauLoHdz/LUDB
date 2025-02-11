### Late Universe Likelihoods
This code computes the Chi2 from a set of different cosmological data of the late universe. In particular, 	cosmic chronometers, megamasers from the megamaser cosmology project, BAO from DESI Y1, SNeIa from either Pantheon+, Union3 or Dark Energy Survey (DES). The code allows binning the data within a specified range, thereby obtaining the likelihood (\chi^2). It recreates the results obtained in arXiv:2411.00095.

In order to produce the confidence regions of cosmological parameters, the provided functions need to be used together with a sampler code, for example `emcee`. An example of this is provided in the `emcee.ipynb` notebook.


### Citation

If you are going to use this code, please make sure to reference the work `arXiv:2411.00095`.


### Usage
- Install the python modules: scipy, numpy, os, pandas, astropy
- Download the data
`./data_download.sh`
- Now you can evaluate the Likelihoods. For example with
`python likelihood.py`


### data_download.sh
Use this file to download the SNe Ia data: Pantheon+, Union3, and DES; as well as the BAO-DESI data, from official repositories.


### bin_range.py
File used to specify the redshift range used in the bin by providing the values of z_min and z_max : `zmin < z < zmax`. 


### cosmology.py
This file defines the cosmological functions to be fitted within the context of the CPL model.


### functions.py
This file defines the chi squares of all data used: Cosmic Chronometers, Megamasers, SNe Ia, BAO; as well as the total chi2 for BASE+SNe

The code is designed to assign a null value to the chi2s that are not within a certain bin. It also allows obtaining the full likelihood without binning, using the entire dataset, when specifying the redshift range as `0.01<z<2.33`


### Likelihood.py
Likelihood functions. chi2_Base_Pantheon, chi2_Base_Union3, and chi2_Base_DES are the chi-squared values when using data from Pantheon+, Union3, and DES, respectively.


### Data
Cosmic chronometers: 
Data obtained from the official page of Michele Moresco: https://cluster.difa.unibo.it/astro/CC_data/ . The covariance matrix was formed following arXiv:2003.07362 with the next gitlab repository https://gitlab.com/mmoresco/CCcovariance .

Megamasers:
Taken from Table 1 of arXiv:2001.09213

