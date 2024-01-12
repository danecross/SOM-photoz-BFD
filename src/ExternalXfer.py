
import os
import numpy as np
import pickle

import pandas as pd
from astropy.table import Table

from XferFn import XferFn
from SOM import load_SOM

class ExternalXfer(XferFn):

	def __init__(self, wide_SOM, deep_SOM, outpath,
					 balrog_rlzns=None,
					 use_covariances=False, 
					 deep_flux_fmt=None, deep_cov_fmt=None, deep_err_fmt=None,
					 wide_flux_fmt=None, wide_cov_fmt=None, wide_err_fmt=None,):

		
		self.wide_SOM = wide_SOM
		self.deep_SOM = deep_SOM
		self.save_path = outpath
		self.balrog_path = balrog_rlzns
		self.use_covariances = use_covariances

		self.deep_flux_fmt = deep_flux_fmt
		self.deep_cov_fmt = deep_cov_fmt ; self.deep_err_fmt = deep_err_fmt
		self.wide_flux_fmt = wide_flux_fmt
		self.wide_cov_fmt = wide_cov_fmt ; self.wide_err_fmt = wide_err_fmt

		self.load_fn = load_ExternalXfer

	def generate_realizations(self, balrog_path=None, override=False):
		'''Takes the deep-to-wide table and classifies them into SOMs.'''

		if not hasattr(self, 'simulations'): self.load_realizations()

		svpth = os.path.join(self.save_path, '')
		deep_assignments = self._get_assignments(svpth+"deep_assignments.pkl", self.deep_SOM, 
																flux_fmt=self.deep_flux_fmt,
																err_fmt = self.deep_err_fmt, 
																cov_fmt = self.deep_cov_fmt, override=override)

		wide_assignments = self._get_assignments(svpth+"wide_assignments.pkl", self.wide_SOM, 
																flux_fmt= self.wide_flux_fmt,
																err_fmt = self.wide_err_fmt, 
																cov_fmt = self.wide_cov_fmt, override=override)
	
		self.simulations['DC'] = deep_assignments
		self.simulations['WC'] = wide_assignments

	def _get_assignments(self, savepath, som, 
								flux_fmt=None, err_fmt=None, cov_fmt=None, 
								override=False):

		if not os.path.exists(savepath) or override:
			assignments = som.classify(self.simulations, savepth=savepath, 
												flux_fmt=flux_fmt, err_fmt=err_fmt, cov_fmt=cov_fmt)
		else:
			with open(savepath, 'rb') as f:
				assignments = pickle.load(f)

		return assignments

	def load_realizations(self, alternate_save_path=None):
		'''Loads classified tables generated in `generate_realizations`.'''

		brog_pth = self.balrog_path if alternate_save_path is None else alternate_save_path
		pd_simulations = pd.read_pickle(brog_pth)

		simulations = Table()
		for coln in pd_simulations.columns:
			simulations[coln] = pd_simulations[coln]

		setattr(self, "simulations", simulations)
	
	def save(self, savepath=None):

		savepath = savepath if savepath is not None else os.path.join(self.save_path, 'xferfn.pkl')

		ivars_to_save = ['balrog_path', 'use_covariances', 'deep_flux_fmt', 
								'deep_cov_fmt', 'deep_err_fmt', 'wide_flux_fmt', 
								'wide_cov_fmt', 'wide_err_fmt', 'save_path']
							
		d = {ivar: getattr(self, ivar) for ivar in ivars_to_save}

		d['wide_SOM_savepath'] = os.path.join(self.wide_SOM.save_path, "NoiseSOM.pkl")
		d['deep_SOM_savepath'] = os.path.join(self.deep_SOM.save_path, "NoiseSOM.pkl")

		self.wide_SOM.save(d['wide_SOM_savepath'])
		self.deep_SOM.save(d['deep_SOM_savepath'])

		savepath = savepath if savepath is not None else self.save_path
		with open(savepath, 'wb') as f:
			pickle.dump(d, f)


def load_ExternalXfer(savepath):
	'''
	Loads a saved ExternalXfer object.
	'''

	with open(savepath, 'rb') as f:
		d = pickle.load(f)

	wide_SOM = load_SOM(d['wide_SOM_savepath'])
	deep_SOM = load_SOM(d['deep_SOM_savepath'])

	exfr = ExternalXfer(wide_SOM, deep_SOM, d['save_path'],
								balrog_rlzns=d['balrog_path'],
                			use_covariances=d['use_covariances'],
                			deep_flux_fmt=d['deep_flux_fmt'], 
								deep_cov_fmt=d['deep_cov_fmt'], 
								deep_err_fmt=d['deep_err_fmt'],
                			wide_flux_fmt=d['wide_flux_fmt'], 	
								wide_cov_fmt=d['wide_cov_fmt'], 
								wide_err_fmt=d['wide_err_fmt'],)


	return exfr

