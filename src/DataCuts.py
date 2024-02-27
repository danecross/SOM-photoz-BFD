
from bfd import TierCollection

import healsparse
from astropy.io import fits
import numpy as np

NOISETIER_FILE = '/global/cfs/cdirs/des/mgatti/BFD_targets_newrun/noisetiers.fits'

def get_noisetiers(covariances):

	tc = TierCollection.load(NOISETIER_FILE)

	nt = tc.assign(covariances)
	noisetier_targets = -1*np.ones(covariances.shape[0], dtype=int)
	for key in nt.keys():
		noisetier_targets[nt[key]] = key

	return noisetier_targets

def get_sn_mask(Mf, covariances, sn_min=7, sn_max=200):

	noisetier_targets = get_noisetiers(covariances)
	noisetier = fits.open(NOISETIER_FILE)

	mask = np.zeros_like(noisetier_targets, dtype=bool)
	for tier in range(23):
		mask_nt = (noisetier_targets == tier)

		means_cov_mf = noisetier[1+tier].data['COVARIANCE'][0,0]
		min_flux = sn_min * np.sqrt(means_cov_mf)
		max_flux = sn_max * np.sqrt(means_cov_mf)
		mask[mask_nt] = ((Mf[mask_nt] > min_flux) & (Mf[mask_nt] < max_flux))

	return mask

def get_size_mask(Mf, Mr):
	return ((Mr/Mf > 2.2) & (Mr/Mf < 3.5))


def get_footprint_mask(RA, DEC):
	
	fname = '/global/cfs/cdirs/des/y6-shear-catalogs/'+\
			  'y6-combined-hleda-gaiafull-des-stars-hsmap16384-nomdet-v3.fits'
	hmap = healsparse.HealSparseMap.read(fname)
	in_footprint = hmap.get_values_pos(RA,DEC, valid_mask=True)

	return in_footprint

