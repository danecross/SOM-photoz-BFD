'''
Class definition for Gaussian Simulation generation. 

Written by Dane
'''

import pickle
import numpy as np
import os
import multiprocess as mp
import tqdm
import time
from random import sample
from astropy.table import Table, vstack, Column, MaskedColumn

from gauss_sim_tools import gen_fluxes, apply_cuts 
from SOM import load_SOM

class GaussianSims(object):

	def __init__(self, outpath, MfMr_covpth, Mf_band_covpth, noisetier_pth,
					 full_catalog, wide_SOM_bands, 
					 sim_save=None, n_realizations=100,
					 deep_flux_col_fmt="Mf_", Mf_fmt='Mf', Mr_fmt='Mr', id_fmt='ID', 
					 use_covariances=True):

		'''
		Object for generating Gaussian flux simulations. 

		Args:
			- outpath (str): path for default outputs
			- MfMr_covpth (str): path to healpix map of average MfMr covariance over the footprint*
			- Mf_band_covpth (str): list of paths to healpix maps of average Mf per band error*
			- noisetier_pth (str): path the the healpix map of the noisetier associated with each pixel*
			- full_catalog (str): path to the deep field catalog to be used
			- wide_SOM_bands (str): wide bands to simulate
			- sim_save (str, opt): alternate path to save the simulations (scratch, this file can be very large)
			- n_realizations (int, opt): number of wide realizations for each deep field galaxy
			- deep_flux_col_fmt (str, opt): starting string for the fluxes to be used for simulation
			- Mf_fmt (str, opt): starting string for the mean flux moment to be used for simulation
			- Mr_fmt (str, opt): starting string for the mean size moment to be used for simulation
			- use_covariances (bool, opt): flag to use covariances (square of the error)

		*the sky covariance maps should be loadable by numpy.load and have the structure of a healpix map

		'''

		
		self.n_realizations = n_realizations
		self.deep_flux_col_fmt = deep_flux_col_fmt
		self.id_fmt = id_fmt
		self.use_covariances = use_covariances
		self.wide_SOM_bands = wide_SOM_bands ; self.Mf_fmt = Mf_fmt ; self.Mr_fmt = Mr_fmt

		self.save_path = os.path.abspath(outpath)
		self.sim_save = os.path.abspath(sim_save) if sim_save is not None \
								else self.save_path

		self.full_catalog_pth = full_catalog
		self.full_catalog = Table.read(full_catalog, memmap=True)

		self.err_maps = self._load_covariances(MfMr_covpth, Mf_band_covpth, use_covariances)
		self.noisetier_map = np.load(noisetier_pth)

	def _load_covariances(self, MfMr_covpth, Mf_band_covpth, use_covariances):

		err_maps = {'MfMr': np.load(MfMr_covpth)}
		
		covs_band = np.load(Mf_band_covpth)
		err_maps = err_maps | {'Mf_'+b: covs_band[:,i] 
										for i,b in enumerate(self.wide_SOM_bands)}

		if use_covariances:
			for k in err_maps:
				if '_' not in k: continue
				err_maps[k] = np.sqrt(err_maps[k])

		setattr(self, 'nonzero_covpix', np.where(err_maps['MfMr']!=0)[0])
		return err_maps

	def generate_realizations(self, mask_SN=(7,200), n_procs=20, save_uncut=False):
		'''
		Get/generate the wide map realizations for making the transfer function.

		args:
			- *mask_sn (tuple of floats): Signal-to-Noise cuts for simulations
			- *n_procs (int): number of processes to run (multiprocess.Pool)
			- *save_uncut (bool): save all generated simulations, not just the version after cuts
		output:
			Saves the simulated catalog to the outpath in an astropy table format called "simulations.fits"
			If `save_uncut` is true, saves uncut table to "sims_uncut.fits"
		'''

		with mp.Pool(n_procs) as p:

			# generate fluxes in parallel
			args = [(gts, self.nonzero_covpix, self.err_maps, self.noisetier_map, self.n_realizations, 
							self.wide_SOM_bands, self.deep_flux_col_fmt, self.Mf_fmt, self.Mr_fmt, self.id_fmt)
						  for gts in self.full_catalog]
			sims = p.starmap(gen_fluxes, tqdm.tqdm(args, total=len(self.full_catalog)))

			simulations = vstack(sims)		
		
			sim_ID = np.arange(len(simulations), dtype=int)
			simulations['sim_ID'] = sim_ID

			if save_uncut: simulations.write(os.path.join(self.sim_save, "sims_uncut.fits"), overwrite=True)

			simulations = apply_cuts(simulations) # noise tier and size cuts
			simulations.write(os.path.join(self.sim_save, "simulations.fits"), overwrite=True)
			setattr(self, "simulations", simulations)

			return simulations

	def load_realizations(self, alternate_save_path=None):
		'''
		Loads the wide-field realizations that were generated (gaussian xfer) or simulated (balrog).

		Args:
			- *alternate_save_path (str): if not None, will load the simulations table specified. If	
					left to default value, will load the `simulations.fits` file in the save path.
		'''

		if alternate_save_path is not None:
			savepath = alternate_save_path
		else:
			savepath = os.path.join(self.simsave, "simulations.fits")

		setattr(self, "simulations", Table.read(savepath))





