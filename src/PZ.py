
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

class PZ(object):

	def __init__(self, wide_SOM, deep_SOM, outpath, sky_cov=None, n_realizations=100):
		
		'''
		Constructor for PZ class. 

		args:
			- wide_SOM (SOM): trained wide SOM object
			- deep_SOM (SOM): trained deep SOM object
			- outpath (str): the directory to which we save all results
			- *sky_cov (str): path to the sky covariances used to generate the gaussian xfer function
			- *n_realizations (int): how many wide realizations for each deep field galaxy
		'''
		self.wide_SOM = wide_SOM
		self.deep_SOM = deep_SOM
		self.save_path = outpath

		self.sky_cov_path = sky_cov
		self.n_realizations = n_realizations

		if sky_cov is not None:
			with open(sky_cov, 'rb') as f:
				self.covariances = pickle.load(f)

			nonzero_mask = np.array(self.covariances.nonzero()[1])
			self.noise_options = self.covariances.A[:,nonzero_mask]
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


		with mp.Pool(4) as p:
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
		simulations.write(os.path.join(self.save_path, "simulations.fits"), format='fits')
		setattr(self, "simulations", simulations)

	def load_realizations(self, alternate_save_path=None):
		if alternate_save_path is not None:
			savepath = alternate_save_path
		else:
			savepath = os.path.join(self.save_path, "simulations.fits")

		setattr(self, "simulations", Table.read(savepath))
	
	def make_redshift_map(self, weighted=True):

		setattr(self, "pcchat", self._get_p_c_chat(weighted))
		setattr(self, "pzc", self._get_p_z_c())
		setattr(self, "pzchat", self._get_p_z_chat())

		def E(v, pv):
			return np.array([np.average(v, weights=p) if np.sum(p)>0 else np.nan for p in pv])

		peak_probs = E(self.redshifts, self.pzchat)
		redshift_map = peak_probs.reshape(self.wide_SOM.somres,self.wide_SOM.somres)

		setattr(self, "redshift_map", redshift_map)
		
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
			pcchat[:,i] = pcchat[:, i]/np.sum(pcchat[:,i])

		pcchat = np.nan_to_num(pcchat)
		return pcchat

	def _get_p_z_c(self):
		
		ncells_deep, ncells_wide = self.deep_SOM.somres**2,self.wide_SOM.somres**2
		num_zbins = 150 ; zrange = (0,np.max(self.simulations['Z']))

		redshifts = np.linspace(zrange[0], zrange[1], num_zbins)
		setattr(self, 'redshifts', redshifts)

		cz = [np.array([])]*ncells_deep 
		for row in self.simulations:
			cz[int(row['DC'])] = np.append(cz[int(row['DC'])], [row['Z']])
    
		pzc_unnormed = np.zeros((num_zbins, ncells_deep)) 
		pzc = np.zeros((num_zbins, ncells_deep)) 
		for i in range(ncells_deep):
			pzc[:,i] = np.histogram(cz[i], num_zbins, range=zrange)[0]/len(cz[i])
			pzc_unnormed[:,i] = np.histogram(cz[i], num_zbins, range=zrange)[0]
    
		pzc = np.nan_to_num(pzc)
		return pzc

	def _get_p_z_chat(self):
		pzchat = np.einsum('zt,td->dz', self.pzc, self.pcchat)
		return pzchat

	def make_bins(self, num_tomo_bins, **kwargs):
		'''
		Function to generate the tomographic bins
		
		args:
			- num_tomo_bins (int): number of tomographic bins
			- *weights (bool): use cell weights
			- *compost_bin (float): percent of the galaxies to compost
			- *
		'''
		
		if 'weights' in kwargs and kwargs['weights']:
			raise NotImplementedError

		available_bins, occupation = self._get_bin_populations()
		if 'compost_bin' in kwargs:
			compost_bin_WCs, available_bins = self._assign_to_compost(available_bins, 
																						 kwargs['compost_bin'], 
																						 occupation)
		else:
			compost_bin_WCs = []

		binned_WCs = self._assign_to_tomo_bins(available_bins, num_tomo_bins, occupation)
	
		return Result(self.pzchat, binned_WCs, compost_bin_WCs, self.redshifts)
		

	def _get_bin_populations(self, BSOM=None, overwrite_assignments=False):
		'''
		Gets the bin population for the sample from the wide SOM sample.

		args: 
			- *BSOM (BSOM): to get the population for the whole DES catalog, use a 
								 BSOM (Big data SOM). See BSOM.py for more information.
		'''

		som = self.wide_SOM if BSOM is None else BSOM
		available_bins = [int(i) for i in list(set(som.grouped_by_cell['CA']))]
		som.validate(overwrite_assignments)
		return available_bins, som.get('occupation').flatten()

	def _assign_to_compost(self, available_bins, percent_to_compost, occupation):
		
		# order bins on standard deviation of pzchat
		stddev = [D(self.redshifts, self.pzchat[i]) for i in available_bins]
		ordered_bins = [wc for _,wc in reversed(sorted(zip(stddev, available_bins)))]

		# remove bins one by one until percent removed is percent_to_compost
		ng_in_compost = 0 ; total_ng = np.sum(occupation)
		compost_wcs = []
		while ng_in_compost/total_ng <= percent_to_compost:
			compost_wcs += [ordered_bins.pop(0)]
			ng_in_compost += occupation[compost_wcs[-1]]
	
		return compost_wcs, ordered_bins

	def _assign_to_tomo_bins(self, available_bins, num_tomo_bins, occupation):

		medians = [E(self.redshifts, self.pzchat[i]) for i in available_bins]
		ordered_bins = [wc for m,wc in sorted(zip(medians, available_bins)) if m>0]

		total_ng = np.sum(occupation[available_bins])
		bin_wcs = []
		for i in range(num_tomo_bins):
			wcs = [] ; ng_in_bin = 0
			while ng_in_bin/total_ng < 1/num_tomo_bins and len(ordered_bins)>0:
				wcs += [ordered_bins.pop(0)]
				ng_in_bin += occupation[wcs[-1]]

			bin_wcs += [wcs]

		return bin_wcs

	def save(self, savepath='.'):
		'''
		Saves the PZ path.

		*Note: saves wide/deep SOM 
		'''
		if not os.path.isdir(savepath):
			raise ValueError("Save path must be a directory (3 files to be created)")

		PZ_path = os.path.join(savepath, 'PZ.pkl')
		wideSOM_path = os.path.join(savepath, 'wide.pkl')
		deepSOM_path = os.path.join(savepath, 'deep.pkl')

		self.wide_SOM.save(wideSOM_path)
		self.deep_SOM.save(deepSOM_path)
		
		to_save = {} ; variables_to_skip = []
		for ivar in self.__dict__.keys():
			if ivar in variables_to_skip: continue
			to_save[ivar] = getattr(self, ivar)

		with open(PZ_path, 'wb') as f:
			pickle.dump(to_save, f)


def load_PZ(savepath):
	'''
	Loads a saved PZ object.
	'''
	wide_SOM = load_SOM(os.path.join(savepath, 'wide.pkl'))
	deep_SOM = load_SOM(os.path.join(savepath, 'deep.pkl'))
	
	with open(os.path.join(savepath, 'PZ.pkl'), 'rb') as f:
		sd = pickle.load(f)
	
	pz = PZ(wide_SOM, deep_SOM, sd['save_path'], 
				sky_cov=sd['sky_cov_path'],
				n_realizations=sd['n_realizations'])

	for ivar in sd:
		setattr(pz, ivar, sd[ivar])

	return pz

class Result(object):
	
	def __init__(self, p_z_chat, binned_WCs, compost_WCs, redshifts):

		self.pzchat = p_z_chat
		self.binned_WCs = binned_WCs
		self.compost_WCs = compost_WCs
		self.z = redshifts
	
	def calculate_Nz(self):
		
		Nz = {}
		if len(self.compost_WCs) > 0:
			Nz[-1] = np.sum(self.pzchat[self.compost_WCs], axis=0)/len(self.compost_WCs)
		for i,tbin in enumerate(self.binned_WCs):
			Nz[i] = np.sum(self.pzchat[tbin], axis=0)/len(tbin)

		return Nz

	

