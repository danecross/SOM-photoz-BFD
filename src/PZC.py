
import time

import multiprocess as mp
import numpy as np
import pickle
import os
import tqdm

from random import sample
from astropy.table import Table, vstack

from pipeline_tools import *
from SOM import load_SOM

class PZC(object):

	def __init__(self, wide_SOM, deep_SOM, outpath, sky_cov=None, n_realizations=100, 
					 deep_flux_col_fmt="Mf_", use_covariances=True):
		
		'''
		Constructor for PZC class. 

		args:
			- wide_SOM (SOM): trained wide SOM object
			- deep_SOM (SOM): trained deep SOM object
			- outpath (str): the directory to which we save all results
			- *sky_cov (str): path to the sky covariances used to generate the gaussian xfer function
			- *n_realizations (int): how many wide realizations for each deep field galaxy
			- *deep_flux_col_fmt (str): the format of the flux column to be used for wide field 
					simulations. `deep_SOM.validation_sample` should have columns that begin with this 
					argument.
			- *use_covariances (bool): if True, the covariance map is indeed made of covariances, if 
					False, just pure errors

		Note on covariance input: should be a pickled, a numpy matrix with dimensions (npix,4)

		'''
		self.wide_SOM = wide_SOM
		self.deep_SOM = deep_SOM
		self.save_path = os.path.abspath(outpath)
		self.deep_flux_col_fmt = deep_flux_col_fmt

		self.n_realizations = n_realizations
		self.use_covariances = use_covariances

		if sky_cov is not None:
			self.sky_cov_path = os.path.abspath(sky_cov)
			with open(sky_cov, 'rb') as f:
				self.coverrs = pickle.load(f)

			nonzero_mask = np.array(self.coverrs[:,0].nonzero()[0])
			try:
				self.noise_options = self.coverrs[nonzero_mask,:].A
			except AttributeError:
				self.noise_options = self.coverrs[nonzero_mask,:]

			if use_covariances:
				self.noise_options = np.sqrt(self.noise_options)

		else:
			raise NotImplementedError

	def generate_realizations(self, gaussian=True, mask_SN=True):
		'''
		Get/generate the wide map realizations for making the transfer function.

		args:
			- *gaussian (bool): flag for generating gaussian realizations or pulling from simulations
		output:
			Saves the simulated catalog to the outpath in an astropy table format called "simulations.fits"
		'''
		if gaussian and self.coverrs is None: 
			raise ValueError("For gaussian xfer function must provide sky covariances")
		elif not gaussian:
			raise NotImplementedError

		# generates n_realizations fluxes for single deep field galaxy
		def _gen_fluxes(gal_to_sim, noise_options, n_realizations, bands):
			template_fluxes = [gal_to_sim[self.deep_flux_col_fmt+s] for s in bands]

			idcs = np.array(sample(range(noise_options.shape[0]), n_realizations)) 
			wide_fluxes_err = np.array([noise_options[idcs, i] 
												for i in range(noise_options.shape[1])]).T

			wide_fluxes = np.array([[np.random.normal(f, ferr[j]) for j,f in enumerate(template_fluxes)]
                                 for ferr in wide_fluxes_err])

			return wide_fluxes, wide_fluxes_err

		# removes fluxes with "bad" S/N
		def _mask_SN(wide_fluxes, wide_fluxes_err):

			min_SN = 7 ; max_SN=200
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

			num_inds = len(self.deep_SOM.validate_sample)
			gals_to_sim = [row for row in self.deep_SOM.validate_sample]

			# generate fluxes in parallel
			args = [(gts, self.noise_options, self.n_realizations, self.wide_SOM.bands,) 
						for gts in gals_to_sim]
			generated_fluxes = p.starmap(_gen_fluxes, tqdm.tqdm(args, total=num_inds))

			# mask fluxes by Signal to Noise
			if mask_SN:
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
		Loads the wide-field realizations that were previously generated.

		Args:
			- *alternate_save_path (str): if not None, will load the simulations table specified. If 	
					left to default value, will load the `simulations.fits` file in the save path.
		'''
		if alternate_save_path is not None:
			savepath = alternate_save_path
		else:
			savepath = os.path.join(self.save_path, "simulations.fits")

		setattr(self, "simulations", Table.read(savepath))
	
	def make_redshift_map(self, weights=None, zmax=4, fill_zeros=False):
		'''
		Calculates p(z|chat) (after the simulations have been created). 

		Args:
			- *weights (): Not Implemented Yet. 
			- *zmax (float): the maximum redshift cutoff to use
		'''
		try:
			setattr(self, "pcchat", self._get_p_c_chat(weights))
			setattr(self, "pzc", self._get_p_z_c(zmax, fill_zeros=fill_zeros))
			setattr(self, "pzchat", self._get_p_z_chat())

		except AttributeError as e:
			print(e)
			raise AttributeError("If the above error says that PZC does not have a 'simulations' attribute,"+\
										"run 'load_realizations' on your PZC object to see if this fixes the problem")

		peak_probs = np.array([E(self.redshifts, pzc) for pzc in self.pzchat])
		redshift_map = peak_probs.reshape(self.wide_SOM.somres,self.wide_SOM.somres)

		return redshift_map

	def _get_p_c_chat(self, weights):

		ncells_deep, ncells_wide = self.deep_SOM.somres**2,self.wide_SOM.somres**2
		pcchat = np.zeros((ncells_deep,ncells_wide))

		for dc,wc in zip(self.simulations['DC'], self.simulations['WC']):
			pcchat[int(dc),:] += np.histogram(wc, bins=ncells_wide, range=(-0.5,ncells_wide))[0]

		# normalize
		empty_count = 0
		for i,_ in enumerate(pcchat.T):
			if sum(pcchat[:,i]) > 0:
				pcchat[:,i] = pcchat[:,i]/np.sum(pcchat[:,i])
			else:
				empty_count += 1

		if weights is not None:
			# apply weights
			pcchat = pcchat/weights
   
		pcchat = np.nan_to_num(pcchat)

		return pcchat

	def _get_p_z_c(self, zmax, fill_zeros=False):
		
		ncells_deep, ncells_wide = self.deep_SOM.somres**2,self.wide_SOM.somres**2
		
		zmax = np.max(self.simulations['Z']) if zmax is None else zmax
		zrange = (0,zmax) ; step_size = 0.01 

		redshifts = np.arange(zrange[0], zrange[1]+step_size, step_size) 
		setattr(self, 'redshifts', redshifts)

		cz = [np.array([])]*ncells_deep 
		for row in self.simulations:
			cz[int(row['DC'])] = np.append(cz[int(row['DC'])], [row['Z']])

		zero = 0
		pzc_unnormed = np.zeros((len(redshifts), ncells_deep)) 
		pzc = np.zeros((len(redshifts), ncells_deep)) 

		for i in range(ncells_deep):
			if len(cz[i])>0:
				pzc_unnormed[:,i] = np.histogram(cz[i], len(redshifts), range=zrange)[0]
				pzc[:,i] = pzc_unnormed[:,i]/np.sum(pzc_unnormed[:,i])
			elif fill_zeros:
				zero += 1
				pzc[:,i] = self.pzc.pzc[:,i]

		pzc = np.nan_to_num(pzc)
		return pzc

	def _get_p_z_chat(self):
		pzchat = np.einsum('zt,td->dz', self.pzc, self.pcchat)

		for i in range(self.wide_SOM.somres**2):
			if np.sum(pzchat[i]) > 0:
				pzchat[i] = pzchat[i]/np.sum(pzchat[i])

		return pzchat

	def save(self, savepath='.'):
		'''
		Saves the PZC path.

		*Note: saves wide/deep SOM 
		'''
		if not os.path.isdir(savepath):
			raise ValueError("Save path must be a directory (3 files to be created)")

		PZC_path = os.path.join(savepath, 'PZC.pkl')
		wideSOM_path = os.path.join(savepath, 'wide.pkl')
		deepSOM_path = os.path.join(savepath, 'deep.pkl')

		self.wide_SOM.save(wideSOM_path)
		self.deep_SOM.save(deepSOM_path)
		
		to_save = {} ; ivars = ['pzchat', 'pzc', 'pcchat', 'save_path', 'redshifts', 
										'sky_cov_path', 'use_covariances']
		for ivar in ivars:
			to_save[ivar] = getattr(self, ivar, None)

		with open(PZC_path, 'wb') as f:
			pickle.dump(to_save, f)


def load_PZC(savepath):
	'''
	Loads a saved PZC object.
	'''
	
	with open(os.path.join(savepath, 'PZC.pkl'), 'rb') as f:
		sd = pickle.load(f)
	
	wide_SOM = load_SOM(os.path.join(savepath, 'wide.pkl'))
	deep_SOM = load_SOM(os.path.join(savepath, 'deep.pkl'))

	pzc = PZC(wide_SOM, deep_SOM, sd['save_path'], 
				sky_cov=sd.get('sky_cov_path',None),
				n_realizations=sd.get('n_realizations', None), 
				use_covariances=sd.get('use_covariances'))

	for ivar in sd:
		setattr(pzc, ivar, sd[ivar])

	return pzc





