
import time

import multiprocess as mp
import numpy as np
import pickle
import os
import tqdm

from random import sample
from astropy.table import Table, vstack
import pandas as pd

from pipeline_tools import *

class PZC(object):

	def __init__(self, simulations_path, redshift_catalog, outpath, wideSOM_res, deepSOM_res, 
						redshift_colname='Z', deepcell='DC', widecell='WC'):
		
		'''
		Initializer for PZC class. 

		args:
			- simulations_path (str): path to the classified simulations
			- redshift_catalog (str): path to a classified catalog of wide field 
											  galaxies with redshift information
			- outpath (str): the directory to which we save all results
			- wideSOM_res (int): resolution of the wide SOM
			- deepSOM_res (int): resolution of the deep SOM
			- redshift_colname (str, opt): name of the redshift column in the above catalog
		'''

		self.deepcell = deepcell
		self.widecell = widecell

		self.sims_pth = simulations_path
		self.simulations = Table.read(simulations_path)
		self.simulations.rename_columns([deepcell, widecell], ['DC','WC'])

		self.z_cat_pth = redshift_catalog
		self.z_cat = Table.read(redshift_catalog)
		self.z_cat.rename_columns([deepcell], ['DC'])

		self.wideSOM_res = wideSOM_res
		self.deepSOM_res = deepSOM_res
		self.save_path = outpath
		
		self.zcol = redshift_colname


	def make_redshift_map(self, weights=None, zmax=5, fill_zeros=False):
		'''
		Calculates p(z|chat) (after the simulations have been created). 

		Args:
			- *weights (): Not Implemented Yet. 
			- *zmax (float): the maximum redshift cutoff to use
		'''

		# TODO: calculate p(chat) for weights
		
		setattr(self, "pcchat", self._get_p_c_chat(weights))
		setattr(self, "pzc", self._get_p_z_c(zmax, fill_zeros=fill_zeros))
		setattr(self, "pzchat", self._get_p_z_chat())

		peak_probs = np.array([E(self.redshifts, pzc) for pzc in self.pzchat])
		redshift_map = peak_probs.reshape(self.wideSOM_res,self.wideSOM_res)

		return redshift_map

	def _get_p_c_chat(self, weights):

		ncells_deep, ncells_wide = self.deepSOM_res**2,self.wideSOM_res**2
		pcchat = np.zeros((ncells_deep,ncells_wide))

		#TODO: replace 1 with weighted value of each DF galaxy
		if weights is None: weights = np.ones(self.simulations['DC'].shape)
		np.add.at(pcchat, (self.simulations['DC'].data.astype(int), 
								 self.simulations['WC'].data.astype(int)), weights) 

		# normalize
		empty_count = 0
		for i,_ in enumerate(pcchat.T):
			if sum(pcchat[:,i]) > 0:
				pcchat[:,i] = pcchat[:,i]/np.sum(pcchat[:,i])
			else:
				empty_count += 1

		pcchat = np.nan_to_num(pcchat)

		return pcchat

	def _get_p_z_c(self, zmax, fill_zeros=False):
		
		ncells_deep, ncells_wide = self.deepSOM_res**2,self.wideSOM_res**2
		
		zmax = np.max(self.z_cat[self.zcol]) if zmax is None else zmax
		zmin = 0.01
		zrange = (zmin,zmax) ; step_size = 0.05 
		zbins = np.arange(zrange[0], zrange[1]+step_size, step_size) 
		redshifts = zbins[:-1] + (zbins[1] - zbins[0])/2.
		setattr(self, 'redshifts', redshifts)

		self.z_cat = self.z_cat[(self.z_cat['Z']>zmin) & (self.z_cat['Z']<zmax)]
		zidx = np.digitize(self.z_cat['Z'], redshifts) - 1
		
		pzc = np.zeros((len(redshifts), ncells_deep))
		weights = 1 if 'overlap_weight' not in self.z_cat.colnames \
						else self.z_cat['overlap_weight']
		np.add.at(pzc, (zidx, self.z_cat['DC']), weights)

		pzc = pzc / np.sum(pzc, axis=0)[None,:]

		pzc = np.nan_to_num(pzc)
		return pzc

	def _get_p_z_chat(self):
		pzchat = np.einsum('zt,td->dz', self.pzc, self.pcchat)

		for i in range(self.wideSOM_res**2):
			if np.sum(pzchat[i]) > 0:
				pzchat[i] = pzchat[i]/np.sum(pzchat[i])

		return pzchat

	def save(self, savepath='.'):
		'''
		Saves the PZC path.

		*Note: saves wide/deep SOM 
		'''
		if not os.path.isdir(savepath):
			raise ValueError("Save path must be a directory")

		to_save = {} 
		ivars = ['sims_pth', 'z_cat_pth', 'save_path', 'pzchat', 'pcchat', 'pzc',
						'wideSOM_res', 'deepSOM_res', 'z_col', 'redshifts',
						'deepcell', 'widecell']
		for ivar in ivars:
			to_save[ivar] = getattr(self, ivar, None)

		PZC_path = os.path.join(savepath, 'PZC.pkl')
		with open(PZC_path, 'wb') as f:
			pickle.dump(to_save, f)


def load_PZC(savepath):
	'''
	Loads a saved PZC object.
	'''
	
	with open(os.path.join(savepath, 'PZC.pkl'), 'rb') as f:
		sd = pickle.load(f)

	pzc = PZC(sd['sims_pth'], sd['z_cat_pth'], sd['save_path'],
					sd['wideSOM_res'], sd['deepSOM_res'], sd.get('z_col', 'Z'), 
					sd.get('deepcell', 'DC'), sd.get('widecell', 'WC'))

	for ivar in sd:
		setattr(pzc, ivar, sd[ivar])

	return pzc



