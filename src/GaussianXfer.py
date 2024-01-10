
import pickle
import numpy as np
import os
import multiprocess as mp
import tqdm

from random import sample
from astropy.table import Table, vstack

from pipeline_tools import *
from PZC import PZC, load_PZC
from SOM import load_SOM

class GaussianXfer(PZC):

	def __init__(self, wide_SOM, deep_SOM, outpath, sky_cov, n_realizations=100,
						deep_flux_col_fmt="Mf_", use_covariances=True):
		
		self.n_realizations = n_realizations
		self.deep_flux_col_fmt = deep_flux_col_fmt
		self.use_covariances = use_covariances

		super().__init__(wide_SOM, deep_SOM, outpath)
		
		self.sky_cov_path = sky_cov
		with open(sky_cov, 'rb') as f:
			self.coverrs = pickle.load(f)

		nonzero_mask = np.array(self.coverrs[:,0].nonzero()[0])
		try:
			self.noise_options = self.coverrs[nonzero_mask,:].A
		except AttributeError:
			self.noise_options = self.coverrs[nonzero_mask,:]

		if use_covariances:
			self.noise_options = np.sqrt(self.noise_options)

		self.ivars_to_save = ['sky_cov_path', 'use_covariances',]

	def generate_realizations(self, mask_SN=(7,200)):
		'''
		Get/generate the wide map realizations for making the transfer function.

		args:
			- *mask_sn (tuple of floats): Signal-to-Noise cuts for simulations
		output:
			Saves the simulated catalog to the outpath in an astropy table format called "simulations.fits"
		'''

		# generates n_realizations fluxes for single deep field galaxy
		def _gen_fluxes(gal_to_sim, noise_options, n_realizations, bands):
			template_fluxes = [gal_to_sim[self.deep_flux_col_fmt+s] for s in bands]

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

				wide_fluxes_err[:,i] = background_truths[:,i] #TODO: figure out if this is okay

			return wide_fluxes, wide_fluxes_err

		# removes fluxes with "bad" S/N
		def _mask_SN(wide_fluxes, wide_fluxes_err):

			min_SN = mask_SN[0] ; max_SN=mask_SN[1]
			SN_ = SN(wide_fluxes,wide_fluxes_err)
			mask = (SN_>min_SN) & (SN_<max_SN)

			return wide_fluxes[mask], wide_fluxes_err[mask]

		# classifies simulated wide fluxes into the wide SOM
		def _classify(wide_fluxes, wide_fluxes_err, wide_SOM, gal_to_sim):

			cells, _ = wide_SOM.SOM.classify(wide_fluxes, wide_fluxes_err)

			t = Table([[gal_to_sim['CA']]*len(cells), cells, [gal_to_sim['COSMOS_PHOTZ']]*len(cells)],
						 names=['DC', 'WC', 'Z'])
			for j,b in enumerate(wide_SOM.bands):
				t['Mf_%s'%b] = wide_fluxes[:,j]
				t['err_Mf_%s'%b] = wide_fluxes_err[:,j]

			return t

		# multiprocess pool for parallelizing the flux generation process
		with mp.Pool(100) as p:

			gals_to_sim = [row for row in self.deep_SOM.validate_sample]
			num_inds = len(gals_to_sim)

			# generate fluxes in parallel
			args = [(gts, self.noise_options, self.n_realizations, self.wide_SOM.bands,)
						for gts in gals_to_sim]
			generated_fluxes = p.starmap(_gen_fluxes, tqdm.tqdm(args, total=num_inds))

			# mask fluxes by Signal to Noise
			if mask_SN is not None:
				masked_fluxes = p.starmap(_mask_SN, tqdm.tqdm(generated_fluxes, total=num_inds))
			else:
				masked_fluxes = generated_fluxes

			# classify wide fluxes and make final "catalogs"
			args = [(wf, wfe, self.wide_SOM, gts,) for (wf, wfe), gts in zip(masked_fluxes, gals_to_sim)]
			results = p.starmap(_classify, tqdm.tqdm(args, total=num_inds))

		simulations = vstack(results)
		simulations.write(os.path.join(self.save_path, "simulations.fits"), format='fits', overwrite=True)
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
			savepath = os.path.join(self.save_path, "simulations.fits")

		setattr(self, "simulations", Table.read(savepath))

def load_GaussianXfer(savepath):
   '''
   Loads a saved PZC object.
   '''

   with open(os.path.join(savepath, 'PZC.pkl'), 'rb') as f:
      sd = pickle.load(f)

   wide_SOM = load_SOM(os.path.join(savepath, 'wide.pkl'))
   deep_SOM = load_SOM(os.path.join(savepath, 'deep.pkl'))

   pzc = GaussianXfer(wide_SOM, deep_SOM, sd['save_path'],
            			 sky_cov=sd.get('sky_cov_path',None),
            			 n_realizations=sd.get('n_realizations', None),
            			 use_covariances=sd.get('use_covariances'))

   for ivar in sd:
      setattr(pzc, ivar, sd[ivar])

   return pzc












