'''Class definition for an object that generates covariance maps for gaussian simulations. '''

import tqdm
import multiprocess as mp
from astropy.table import Table
import healpy as hp
import numpy as np
from scipy.linalg import sqrtm

from DataCuts import get_noisetiers

class CovMaps(object):

	def __init__(self, cat_path, nside, rafmt='RA', decfmt='DEC'):
		'''
		Initializer for CovMaps object. 

		Args:
			- cat_path (str): path to the catalog from which to make the maps
			- nside (int): healpy nside (~resolution) for the resulting maps
			- rafmt (str, opt): format of the RA column of the catalog
			- decfmt (str, opt): format of the DEC column of the catalog

		'''
		self.cat = Table.read(cat_path)
		self.ra=self.cat[rafmt] ; self.dec=self.cat[decfmt]

		self.nside = nside ; self.npix = hp.nside2npix(nside)
		self.population = self._calculate_population_map()

	def _calculate_population_map(self):

		ncounts = np.zeros(self.npix)
		pix = self._bin_cat()
		unique_pix, idx_rep = np.unique(pix, return_inverse=True)
		ncounts[unique_pix] += np.bincount(idx_rep)
	
		return ncounts	

	def _bin_cat(self):

		pix = hp.ang2pix(self.nside, self.ra, self.dec, lonlat=True)
		self.cat['PIX'] = pix
		return pix

	def apply_mask(self, mask):
		'''Safely applies a mask to the catalog object.'''

		setattr(self, 'cat', self.cat[mask])
		setattr(self, 'ra', self.ra[mask])
		setattr(self, 'dec', self.dec[mask])

		new_population = self._calculate_population_map()
		setattr(self, 'population', new_population)

	def make_map(self, col):
		'''
		Makes a pixel averaged map of the given column. 
		
		Args: 
			- col (str or array-like object): either the name of the column to aggregate in 
						the catalog or a list of values to aggregate, ordered with the RA, DEC 
						of the original catalog.

		Returns: 
			- a pixel-averaged healpix map of the column given in the arguments. 
		'''
		npix = hp.nside2npix(self.nside)
		covariance_map = np.zeros(npix)
		data = col if type(col) != str else self.cat[col]

		pix = self.cat['PIX']
		unique_pix, idx_rep = np.unique(pix, return_inverse=True)
		covariance_map[unique_pix] += np.bincount(idx_rep, weights=data)
		covariance_map[self.population!=0] /= self.population[self.population!=0]

		return covariance_map

	def make_single_band_maps(self, perband_col='cov_Mf_per_band'):
		'''
		Makes a map of the covariances of the single bands.

		Args: 
			- perband_col (str or array-like object, opt): the name of the column or the column 
									containing all of the single band covariance information. 
									(shape: (ngals, nbands)

		Returns: 
			- a (npix, nbands) numpy array of pixel-averaged covariances
		'''
		mf_per_band = perband_col if type(perband_col) is not str else self.cat[perband_col]
		nbands = mf_per_band.shape[1]

		Mf_band_covs = np.zeros((self.npix, nbands))
		for i in range(nbands):
			band_cov = mf_per_band[:,i]
			Mf_band_covs[:,i] = self.make_map(band_cov)

		return Mf_band_covs

	def make_full_covariance_map(self, cov_col='covariance'):
		'''
		Makes a map of the full covariance measurements from BFD. 
		
		Args: 
			- cov_col (str, opt): name of the column or the column itself containing the full BFD
										 covariance. (shape: ~(ngals, 15))
		'''
		full_covs = self.cat[cov_col] if type(cov_col)==str else cov_col
		covariance_maps = np.zeros((self.npix, full_covs.shape[1]))
		for i in range(full_covs.shape[1]):
			covi = full_covs[:,i]
			covariance_maps[:,i] = self.make_map(covi)

		return covariance_maps

	def make_noisetier_map(self, full_covariance_maps):
		'''
		Makes a map of the noisetiers for each pixel in the healpy map. 

		Args: 
			- full_covariance_maps (ndarray): pixel-averaged values of the full BFD covariance (this 
														 can be obtained from a call to `make_full_covariance_map`)

		Returns: 
			- Noise tier map
		'''
		nonempty_pix = (self.population!=0)
		nts = -1*np.ones(self.npix)
		nts[nonempty_pix] = get_noisetiers(full_covariance_maps[nonempty_pix,:])
		
		return nts

