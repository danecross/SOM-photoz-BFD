
import pickle
import numpy as np
import os
import multiprocess as mp
import tqdm

from random import sample
from astropy.table import Table, vstack, Column, MaskedColumn

from pipeline_tools import *
from SOM import load_SOM
from XferFn import Simulations

class GaussianSims(Simulations):

	def __init__(self, outpath, sky_cov, full_catalog, wide_SOM_bands, 
					 sim_save=None, n_realizations=100,
					 deep_flux_col_fmt="Mf_", use_covariances=True):

		'''
		Object for generating Gaussian flux simulations. 

		Args:
			- outpath (str): path for default outputs
			- sky_cov (str): path to healpix map of average error over the footprint*
			- full_catalog (str): path to the deep field catalog to be used
			- wide_SOM_bands (str): wide bands to simulate
			- sim_save (str, opt): alternate path to save the simulations (scratch, this file can be very large)
			- n_realizations (int, opt): number of wide realizations for each deep field galaxy
			- deep_flux_col_fmt (str, opt): starting string for the fluxes to be used for simulation
			- use_covariances (bool, opt): flag to use covariances (square of the error)

		*the sky covariance map should be a pickled healpix map

		'''

		
		self.n_realizations = n_realizations
		self.deep_flux_col_fmt = deep_flux_col_fmt
		self.use_covariances = use_covariances
		self.wide_SOM_bands = wide_SOM_bands

		self.save_path = os.path.abspath(outpath)
		self.sim_save = os.path.abspath(sim_save) if sim_save is not None \
								else self.save_path

		self.sky_cov_path = sky_cov
		with open(sky_cov, 'rb') as f:
			self.coverrs = pickle.load(f)

		self.full_catalog_pth = full_catalog
		self.full_catalog = Table.read(full_catalog, memmap=True)

		nonzero_mask = np.array(self.coverrs[:,0].nonzero()[0])
		try:
			self.noise_options = self.coverrs[nonzero_mask,:].A
		except AttributeError:
			self.noise_options = self.coverrs[nonzero_mask,:]

		if use_covariances:
			self.noise_options = np.sqrt(self.noise_options)

		self.load_fn = load_GaussianSims

	def generate_realizations(self, mask_SN=(7,200)):
		'''
		Get/generate the wide map realizations for making the transfer function.

		args:
			- *mask_sn (tuple of floats): Signal-to-Noise cuts for simulations
		output:
			Saves the simulated catalog to the outpath in an astropy table format called "simulations.fits"
		'''

		# generates n_realizations fluxes for single deep field galaxy
		def _gen_fluxes(gal_to_sim, noise_options, n_realizations, bands, deep_col_fmt):
			template_fluxes = [gal_to_sim[deep_col_fmt+s] for s in bands]

			idcs = np.array(sample(range(noise_options.shape[0]), n_realizations))
			background_truths = np.array([noise_options[idcs, i]
													for i in range(noise_options.shape[1])]).T

			shape = (n_realizations, len(template_fluxes))
			background_noise = np.zeros(shape) ; poisson_noise = np.zeros(shape)
			wide_fluxes = np.zeros(shape) ; wide_fluxes_err = np.zeros(shape)
			for i,tf in enumerate(template_fluxes):
				background_noise[:,i] = np.random.normal(0, background_truths[:,i])
				poisson_noise[:,i] =  np.zeros_like(background_noise[:,i]) #TODO: use shot noise

				wide_fluxes[:,i] = np.array([tf]*n_realizations) + poisson_noise[:,i] + background_noise[:,i]

				wide_fluxes_err[:,i] = background_truths[:,i] 

			return wide_fluxes, wide_fluxes_err

		# removes fluxes with "bad" S/N
		def _mask_SN(wide_fluxes, wide_fluxes_err):

			min_SN = mask_SN[0] ; max_SN=mask_SN[1]
			SN_ = SN(wide_fluxes,wide_fluxes_err)
			mask = (SN_>min_SN) & (SN_<max_SN)

			return wide_fluxes[mask], wide_fluxes_err[mask]

		def _make_tables(gal_to_sim, wide_fluxes, wide_fluxes_err, bands):

			tlen = wide_fluxes.shape[0]
			if tlen==0: return Table()

			t = Table([[np.ma.masked]*tlen]*2 + [[gal_to_sim['ID']]*tlen] ,
						 names=['DC', 'WC', 'ID'])

			for j,b in enumerate(bands):
				t['Mf_%s'%b] = wide_fluxes[:,j]
				t['err_Mf_%s'%b] = wide_fluxes_err[:,j]

			return t

		# multiprocess pool for parallelizing the flux generation process
		with mp.Pool(100) as p:

			gals_to_sim = [row for row in self.full_catalog]
			num_inds = len(gals_to_sim)

			# generate fluxes in parallel
			args = [(gts, self.noise_options, self.n_realizations, 
						self.wide_SOM_bands, self.deep_flux_col_fmt,)
						for gts in gals_to_sim]
			generated_fluxes = p.starmap(_gen_fluxes, tqdm.tqdm(args, total=num_inds))

			# mask fluxes by Signal to Noise
			if mask_SN is not None:
				masked_fluxes = p.starmap(_mask_SN, tqdm.tqdm(generated_fluxes, total=num_inds))
			else:
				masked_fluxes = generated_fluxes

			# save results to file
			args = [(gts, wf, wfe, self.wide_SOM_bands) 
							for (wf, wfe), gts in zip(masked_fluxes, gals_to_sim)]
			simulations_to_stack = p.starmap(_make_tables, tqdm.tqdm(args, total=len(args)))
		
		simulations = vstack(simulations_to_stack)		
		simulations.write(os.path.join(self.sim_save, "simulations.fits"), format='fits', overwrite=True)
		setattr(self, "simulations", simulations)

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

	def save(self, savepath=None):

		savepath = savepath if savepath is not None else os.path.join(self.save_path, 'xferfn.pkl')

		ivars_to_save = ['n_realizations', 'sky_cov_path', 'deep_flux_col_fmt',
								'sky_cov_path', 'save_path', 'use_covariances', 
								'full_catalog_pth']
		d = {ivar: getattr(self, ivar) for ivar in ivars_to_save}

		savepath = savepath if savepath is not None else self.save_path
		with open(savepath, 'wb') as f:
			pickle.dump(d, f)

def load_GaussianSims(savepath):
	'''
	Loads a saved GaussianXfer object.
	'''

	with open(savepath, 'rb') as f:
		d = pickle.load(f)

	gxfr = GaussianSims(d['save_path'], d['sky_cov_path'], 
								d['full_catalog_pth'],
								n_realizations=d['n_realizations'],
								deep_flux_col_fmt=d['deep_flux_col_fmt'], 
								use_covariances=d['use_covariances'])



	return gxfr







