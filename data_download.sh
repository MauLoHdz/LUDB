#!/bin/sh

cd Data

# DESI data from the Cobaya repository
git clone https://github.com/CobayaSampler/bao_data.git
mv bao_data BAO_data


# Union3 data from a repository by David Rubin
git clone https://github.com/rubind/union3_release



# Pantheon+ data from a repository by the Pantheon+SH0ES collaboration. 
# The download also includes SH0ES 2022 data. https://arxiv.org/pdf/2112.04510
git clone https://github.com/PantheonPlusSH0ES/DataRelease.git
mv DataRelease PantheonPlusSH0ES



# DES-SNe data from the DES-science repository
git clone https://github.com/des-science/DES-SN5YR
cd DES-SN5YR/4_DISTANCES_COVMAT
gunzip STAT+SYS.txt.gz
