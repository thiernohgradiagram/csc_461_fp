# csc_461_fp
This repo is dedicated for our csc 461 final project / Fall 2024 / University Of Rhode Island

## Setup
- clone this repository: git clone https://github.com/thiernohgradiagram/csc_461_fp.git
- Download, install and verify miniconda installation: https://docs.anaconda.com/miniconda/miniconda-install/
- Create the conda environment associated with this project: 
    - conda env create --file environment.yml
- control or command + shit + p and select base conda as the python interpreter for vs code
- activate the environment associated with this project: 
    - conda activate csc-461-fp

## Installing a package in the environment
- Method 1: 
    - conda activate csc-461-fp
    - conda install packageName
    - conda env export --from-history > environment.yml

- Method 2: 
    - add the package under the dependencies section of environment.yaml
    - conda env update --file environment.yaml --prune

## SSH SETUP
- ls -al ~/.ssh | ls -al /home/zeus/.ssh
- ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
- ls -al /home/zeus/.ssh
- eval "$(ssh-agent -s)"
