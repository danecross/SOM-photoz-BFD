# SOM-photoz-BFD


## Install Instructions

1. Add `path/to/SOM-photoz-BFD/src` to your PYTHONPATH
2. pip install progressbar

## Note on Data Preprocessing

This pipeline does not have a direct pre-processing module, so the data files must have the correct column names for seamless integration. Below are the input data requirements. 

### Wide Photometric Data
- each flux band has the name: `Mf_[g,r,i,z,etc]` (e.g. for the g band, `Mf_g`)
- each flux covariance band has the name `cov_Mf_[g,r,etc]` (e.g. for the g band `cov_Mf_g`)

### Deep Photometric Data 
- each flux band has the name: `Mf_[g,r,i,z,etc]` (e.g. for the g band, `Mf_g`)
- each flux covariance band has the name `cov_Mf_[g,r,etc]` (e.g. for the g band `cov_Mf_g`)
- the redshift column should be names `COSMOS_PHOTZ`

You can check data formatting using the DataValidation module.
 
