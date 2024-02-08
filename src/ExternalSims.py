
import os
import numpy as np
import pickle

import pandas as pd
from astropy.table import Table

from XferFn import Simulations
from SOM import load_SOM

class ExternalSims(Simulations):

	def __init__(self, outpath,
					 balrog_rlzns=None,
					 use_covariances=False, 
					 deep_flux_fmt=None, deep_cov_fmt=None, deep_err_fmt=None,
					 wide_flux_fmt=None, wide_cov_fmt=None, wide_err_fmt=None,):

		
		self.save_path = outpath
		self.balrog_path = balrog_rlzns
		self.use_covariances = use_covariances

		self.deep_flux_fmt = deep_flux_fmt
		self.deep_cov_fmt = deep_cov_fmt ; self.deep_err_fmt = deep_err_fmt
		self.wide_flux_fmt = wide_flux_fmt
		self.wide_cov_fmt = wide_cov_fmt ; self.wide_err_fmt = wide_err_fmt

		self.load_fn = load_ExternalXfer

	def generate_realizations(self, balrog_path=None, override=False):
		raise NotImplementedError("This class should be used with simulations that are made externally.")

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

		savepath = savepath if savepath is not None else self.save_path
		with open(savepath, 'wb') as f:
			pickle.dump(d, f)


def load_ExternalSims(savepath):
	'''
	Loads a saved ExternalXfer object.
	'''

	with open(savepath, 'rb') as f:
		d = pickle.load(f)

	exfr = ExternalSims(d['save_path'],
								balrog_rlzns=d['balrog_path'],
                			use_covariances=d['use_covariances'],
                			deep_flux_fmt=d['deep_flux_fmt'], 
								deep_cov_fmt=d['deep_cov_fmt'], 
								deep_err_fmt=d['deep_err_fmt'],
                			wide_flux_fmt=d['wide_flux_fmt'], 	
								wide_cov_fmt=d['wide_cov_fmt'], 
								wide_err_fmt=d['wide_err_fmt'],)


	return exfr

