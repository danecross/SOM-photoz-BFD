
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

	def __init__(self, xfer_fn, redshift_catalog, outpath):
		
		'''
		Initializer for PZC class. 

		args:
			- xfer_fn (obj): the trasnfer function being used to make the redshift map
			- redshift_catalog (str): path to a classified catalog of wide field 
											  galaxies with redshift information
			- outpath (str): the directory to which we save all results

		'''

		self.xfer_fn = xfer_fn
		
		self.z_cat_pth = redshift_catalog
		self.z_cat = Table.read(redshift_catalog, memmap=True)

		self.wide_SOM = self.xfer_fn.wide_SOM
		self.deep_SOM = self.xfer_fn.deep_SOM
		self.save_path = outpath


	def make_redshift_map(self, weights=None, zmax=4, fill_zeros=False):
		'''
		Calculates p(z|chat) (after the simulations have been created). 

		Args:
			- *weights (): Not Implemented Yet. 
			- *zmax (float): the maximum redshift cutoff to use
		'''
		
		setattr(self, "pcchat", self._get_p_c_chat(weights))
		setattr(self, "pzc", self._get_p_z_c(zmax, fill_zeros=fill_zeros))
		setattr(self, "pzchat", self._get_p_z_chat())

		peak_probs = np.array([E(self.redshifts, pzc) for pzc in self.pzchat])
		redshift_map = peak_probs.reshape(self.wide_SOM.somres,self.wide_SOM.somres)

		return redshift_map

	def _get_p_c_chat(self, weights):

		ncells_deep, ncells_wide = self.deep_SOM.somres**2,self.wide_SOM.somres**2
		pcchat = np.zeros((ncells_deep,ncells_wide))

		for dc,wc in zip(self.xfer_fn.simulations['DC'], self.xfer_fn.simulations['WC']):
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
		
		zmax = np.max(self.xfer_fn.simulations['Z']) if zmax is None else zmax
		zrange = (0,zmax) ; step_size = 0.01 

		redshifts = np.arange(zrange[0], zrange[1]+step_size, step_size) 
		setattr(self, 'redshifts', redshifts)

		cz = [np.array([])]*ncells_deep 
		for row in self.xfer_fn.simulations:
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
			raise ValueError("Save path must be a directory")

		to_save = {} 
		ivars = ['z_cat_pth', 'pzchat', 'pzc', 'pcchat', 'save_path', 'redshifts'] 
		for ivar in ivars:
			to_save[ivar] = getattr(self, ivar, None)

		xferfn_path = os.path.join(self.xfer_fn.save_path, 'xferfn.pkl')
		self.xfer_fn.save(xferfn_path)
		to_save['xfer_fn_path'] = xferfn_path
		to_save['xfer_load_fn'] = self.xfer_fn.load_fn

		PZC_path = os.path.join(savepath, 'PZC.pkl')
		with open(PZC_path, 'wb') as f:
			pickle.dump(to_save, f)


def load_PZC(savepath):
	'''
	Loads a saved PZC object.
	'''
	
	with open(os.path.join(savepath, 'PZC.pkl'), 'rb') as f:
		sd = pickle.load(f)

	xfer = sd['xfer_load_fn'](sd['xfer_fn_path'])
	pzc = PZC(xfer, sd['z_cat_pth'], sd['save_path'],) 

	for ivar in sd:
		setattr(pzc, ivar, sd[ivar])

	return pzc



