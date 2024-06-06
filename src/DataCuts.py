
from bfd import TierCollection

import healsparse
from astropy.io import fits
import numpy as np

NOISETIER_FILE = '/global/cfs/cdirs/des/mgatti/BFD_targets_newrun/noisetiers_WF.fits'

def get_noisetiers(covariances):
	'''
	Assigns noise tiers to any galaxy with a full set of covariances. 

	Args: 
		- covariances (numpy ndarray): 15xn numpy array of the BFD moments covariance matrix

	Returns: 
		numpy array of length n of the assigned noise tiers
	'''

	tc = TierCollection.load(NOISETIER_FILE)

	nt = tc.assign(covariances)
	noisetier_targets = -1*np.ones(covariances.shape[0], dtype=int)
	for key in nt.keys():
		noisetier_targets[nt[key]] = key

	return noisetier_targets

def get_sn_mask(Mf, covariances=None, preassigned_noisetiers=None, 
					 sn_min=10, sn_max=200, flux_min=1500, flux_max=90000):
	'''
	Determines the mask for noise tier cuts.

	Args:
		- *covariances (numpy ndarray): 15xn numpy array for the covarainces for all n galaxies **
		- *preassigned_noisetiers (numpy array): numpy array of noise tiers assigned to all n galaxies **
		- *sn_min (float): Signal-to-noise lower limit
		- *sn_max (float): Signal-to-noise upper limit

	Returns:
		Noise tier mask for each covariance given

	** either `covariances` or `preassigned_nosietier` can be not None. Defaults to assigning 
		covariances manually. 
	'''

	if covariances is not None:
		noisetier_targets = get_noisetiers(covariances)
	else:
		noisetier_targets = preassigned_noisetiers 
		assert(noisetier_targets is not None)

	noisetier = fits.open(NOISETIER_FILE)

	mask = np.zeros_like(noisetier_targets, dtype=bool)
	for tier in range(23):
		mask_nt = (noisetier_targets == tier)

		means_cov_mf = noisetier[1+tier].data['COVARIANCE'][0,0]
		min_flux = sn_min * np.sqrt(means_cov_mf)
		max_flux = sn_max * np.sqrt(means_cov_mf)
		mask[mask_nt] = ((Mf[mask_nt] > min_flux) & (Mf[mask_nt] < max_flux))

	mask = (mask & (Mf > flux_min) & (Mf < flux_max))

	return mask

def get_size_mask(Mf, Mr):
	'''Determines the mask for the BFD size cut.'''
	return ((Mr/Mf > 2.2) & (Mr/Mf < 3.5))


def get_footprint_mask(RA, DEC):
	'''Determines the position mask for the official DES footprint.'''
	
	fname = '/global/cfs/cdirs/des/y6-shear-catalogs/'+\
			  'y6-combined-hleda-gaiafull-des-stars-hsmap131k-mdet-v2.hsp'
	hmap = healsparse.HealSparseMap.read(fname)
	in_footprint = hmap.get_values_pos(RA,DEC, valid_mask=True)

	return in_footprint


