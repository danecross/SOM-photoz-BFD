
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

	def __init__(self, wide_SOM, deep_SOM, outpath, sky_cov=None, n_realizations=100):
		
		'''
		Constructor for PZC class. 

		args:
			- wide_SOM (SOM): trained wide SOM object
			- deep_SOM (SOM): trained deep SOM object
			- outpath (str): the directory to which we save all results
			- *sky_cov (str): path to the sky covariances used to generate the gaussian xfer function
			- *n_realizations (int): how many wide realizations for each deep field galaxy
		'''
		self.wide_SOM = wide_SOM
		self.deep_SOM = deep_SOM
		self.save_path = os.path.abspath(outpath)

		self.n_realizations = n_realizations

		if sky_cov is not None:
			self.sky_cov_path = os.path.abspath(sky_cov)
			with open(sky_cov, 'rb') as f:
				self.covariances = pickle.load(f)

			nonzero_mask = np.array(self.covariances.nonzero()[1])
			try:
				self.noise_options = self.covariances.A[:,nonzero_mask]
			except AttributeError:
				self.noise_options = self.covariances[:,nonzero_mask]
			self.noise_options_idcs = np.where(nonzero_mask)
		else:
			raise NotImplementedError

	def generate_realizations(self, gaussian=True):
		'''
		Get/generate the wide map realizations for making the transfer function.

		args:
			- *gaussian (bool): flag for generating gaussian realizations or pulling from simulations
		output:
			Saves the simulated catalog to the outpath in an astropy table format called "simulations.fits"
		'''
		if gaussian and self.covariances is None: 
			raise ValueError("For gaussian xfer function must provide sky covariances")
		elif not gaussian:
			raise NotImplementedError

		def _gen_fluxes(gal_to_sim, noise_options, n_realizations, bands):
			template_fluxes = [gal_to_sim['Mf_%s'%s] for s in bands]
			idcs = np.array(sample(range(noise_options.shape[1]), n_realizations)) 
			wide_fluxes_err = np.array([np.sqrt(noise_options[i, idcs]) 
												for i in range(noise_options.shape[0])]).T
			wide_fluxes = np.array([[np.random.normal(f, ferr[j]) for j,f in enumerate(template_fluxes)]
                                 for ferr in wide_fluxes_err])

			return wide_fluxes, wide_fluxes_err

		def _mask_SN(wide_fluxes, wide_fluxes_err):

			min_SN = 7 ; max_SN=200
			SN_ = SN(wide_fluxes,wide_fluxes_err)
			mask = (SN_>min_SN) & (SN_<max_SN)

			return wide_fluxes[mask], wide_fluxes_err[mask]

		def _classify(wide_fluxes, wide_fluxes_err, wide_SOM, gal_to_sim):
			
			cells, _ = wide_SOM.SOM.classify(wide_fluxes, wide_fluxes_err)

			t = Table([[gal_to_sim['CA']]*len(cells), cells, [gal_to_sim['COSMOS_PHOTZ']]*len(cells)], 
						 names=['DC', 'WC', 'Z'])
			for j,b in enumerate(wide_SOM.bands):
				t['Mf_%s'%b] = wide_fluxes[:,j]
				t['cov_Mf_%s'%b] = wide_fluxes_err[:,j]

			return t


		with mp.Pool(100) as p:

			num_inds = len(self.deep_SOM.validate_sample)
			gals_to_sim = [row for row in self.deep_SOM.validate_sample]

			# generate fluxes in parallel
			args = [(gts, self.noise_options, self.n_realizations, self.wide_SOM.bands,) 
						for gts in gals_to_sim]
			generated_fluxes = p.starmap(_gen_fluxes, tqdm.tqdm(args, total=num_inds))

			# mask fluxes by Signal to Noise
			masked_fluxes = p.starmap(_mask_SN, tqdm.tqdm(generated_fluxes, total=num_inds))

			# classify wide fluxes and make final "catalogs"
			args = [(wf, wfe, self.wide_SOM, gts,) for (wf, wfe), gts in zip(masked_fluxes, gals_to_sim)]
			results = p.starmap(_classify, tqdm.tqdm(args, total=num_inds))
			
		simulations = vstack(results)
		simulations.write(os.path.join(self.save_path, "simulations.fits"), format='fits', overwrite=True)
		setattr(self, "simulations", simulations)

	def load_realizations(self, alternate_save_path=None):
		if alternate_save_path is not None:
			savepath = alternate_save_path
		else:
			savepath = os.path.join(self.save_path, "simulations.fits")

		setattr(self, "simulations", Table.read(savepath))
	
	def make_redshift_map(self, weighted=False, zmax=6):

		setattr(self, "pcchat", self._get_p_c_chat(weighted))
		setattr(self, "pzc", self._get_p_z_c(zmax))
		setattr(self, "pzchat", self._get_p_z_chat())

		peak_probs = np.array([E(self.redshifts, pzc) for pzc in self.pzchat])
		redshift_map = peak_probs.reshape(self.wide_SOM.somres,self.wide_SOM.somres)

		return redshift_map

	def _get_p_c_chat(self, weighted=False):

		ncells_deep, ncells_wide = self.deep_SOM.somres**2,self.wide_SOM.somres**2
		pcchat = np.zeros((ncells_deep,ncells_wide))
		for dc,wc in zip(self.simulations['DC'], self.simulations['WC']):
			pcchat[int(dc),:] += np.histogram(wc, bins=ncells_wide, range=(0,ncells_wide))[0]

		if weighted:
			# apply weights
			pcchat = pcchat/self.WC_weights
    
		# normalize
		empty_count = 0
		for i,_ in enumerate(pcchat.T):
			if sum(pcchat[:,i]) > 0:
				pcchat[:,i] = pcchat[:, i]/np.sum(pcchat[:,i])
			else:
				empty_count += 1

		pcchat = np.nan_to_num(pcchat)
		return pcchat

	def _get_p_z_c(self, zmax):
		
		ncells_deep, ncells_wide = self.deep_SOM.somres**2,self.wide_SOM.somres**2
		
		zmax = np.max(self.simulations['Z']) if zmax is None else zmax
		zrange = (0,np.max(self.simulations['Z']))
		step_size = 0.01 

		redshifts = np.arange(zrange[0], zrange[1], step_size) ; num_zbins = len(redshifts)
		setattr(self, 'redshifts', redshifts)

		cz = [np.array([])]*ncells_deep 
		for row in self.simulations:
			cz[int(row['DC'])] = np.append(cz[int(row['DC'])], [row['Z']])
    
		pzc_unnormed = np.zeros((len(redshifts), ncells_deep)) 
		pzc = np.zeros((len(redshifts), ncells_deep)) 
		for i in range(ncells_deep):
			if len(cz[i])>0:
				pzc[:,i] = np.histogram(cz[i], num_zbins, range=zrange)[0]/len(cz[i])
				pzc_unnormed[:,i] = np.histogram(cz[i], num_zbins, range=zrange)[0]

		pzc = np.nan_to_num(pzc)
		return pzc

	def _get_p_z_chat(self):
		pzchat = np.einsum('zt,td->dz', self.pzc, self.pcchat)
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
		
		to_save = {} ; ivars = ['pzchat', 'save_path', 'redshifts', 'sky_cov_path']
		for ivar in ivars:
			to_save[ivar] = getattr(self, ivar)

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
				n_realizations=sd.get('n_realizations', None))

	for ivar in sd:
		setattr(pzc, ivar, sd[ivar])

	return pzc





