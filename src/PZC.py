
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
from SOM import load_SOM

class PZC(object):

	def __init__(self, wide_SOM, deep_SOM, outpath):
		
		'''
		Initializer for PZC class. 

		args:
			- wide_SOM (SOM): trained wide SOM object
			- deep_SOM (SOM): trained deep SOM object
			- outpath (str): the directory to which we save all results

		'''
		self.wide_SOM = wide_SOM
		self.deep_SOM = deep_SOM
		self.save_path = os.path.abspath(outpath)

	def load_realizations(self, alternate_save_path=None):
		'''
		Loads the wide-field realizations. 

		Note: To make subclass implementation for this class, the only requirement is that
				there is an instance variable established called "simulations" which points to an
				astropy Table of deep field galaxies simulated onto the wide field. This table must have 
				columns "DC" and "WC" for the deep cell and wide cell assignments respectively

		Args:
			- *alternate_save_path (str): if not None, will load the simulations table specified. If 	
					left to default value, will load the `simulations.fits` file in the save path.
		'''
		raise NotImplementedError("This must be implemented in the child class")

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
		
		to_save = {} 
		ivars = ['pzchat', 'pzc', 'pcchat', 'save_path', 'redshifts'] 
		ivars += getattr(self, "ivars_to_save", [])
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

	pzc = PZC(wide_SOM, deep_SOM, sd['save_path'],) 

	for ivar in sd:
		setattr(pzc, ivar, sd[ivar])

	return pzc





