
import pickle
import numpy as np
from astropy.table import Table
import multiprocess as mp
import tqdm
import os

from NoiseSOMPZ import *
from SOMRunner import ClassifyRunner

class SOM(object):

	def __init__(self, resolution, train_cat=None, pretrained_weights=None, 
						reinit_SOM_args={}, analysis_output_path='./',
						col_fmt="Mf_", cov_fmt="cov_Mf_", err_fmt=None):
		'''
		Initialize a SOM object (wraps Gary's NoiseSOM)

		args: 
			- resolution (int): the square side size of the SOM
			- train_cat (str): path to the randomly selected catalog for training the SOM
			- pretrained_weights (numpy ndarray): a resolutionxresolution numpy array of 
															  pre-trained weights
			- reinit_SOM_args (dict): arguments for the NoiseSOM metric
			- *analysis_output_path (str): directory to save analysis ouptuts
			- *col_fmt (str): format for the flux columns the catalog
			- *cov_fmt (str): format for the flux covariance columns in the catalog (if errors 
					not covariances, set this value to None)
			- *err_fmt (str): format for the flux error columns in the catalog (default None, 
					if flux errors, set this value appropriately)

		Note on flux column names: the [col/cov/err]_fmt arguments should denote the beginning 
			of the flux column names. e.g. if the col_fmt = "EX_FLUX_" then the column for the	
			g-band should be "EX_FLUX_G" or EX_FLUX_g""

		'''

		self.save_path = os.path.abspath(analysis_output_path)
		self.somres = resolution
			
		if train_cat is not None:
			self._get_catalogs(train_cat, col_fmt, cov_fmt, err_fmt)
			self.train_cat_path = os.path.abspath(train_cat)
			self.SOM = self._initialize_SOM(self.train_fluxes, self.train_err)
			self.trained = False
		elif pretrained_weights is not None:
			self.SOM = self._reinitialize_SOM(pretrained_weights, **reinit_SOM_args)
			self.trained = True
		else:
			raise ValueError("Either train_cat or pretrained_weights must be not None")

		self.col_fmt = col_fmt
		self.cov_fmt = cov_fmt
		self.err_fmt = err_fmt

	def _initialize_SOM(self, fluxes, fluxes_err):

		if len(self.train_sample) == 0: return
		
		nTrain=fluxes.shape[0]
		hh = hFunc(nTrain,sigma=(30,1))
		metric = AsinhMetric(lnScaleSigma=0.4,lnScaleStep=0.03)
		indices = np.random.choice(fluxes.shape[0],size=nTrain,replace=False)
		som = NoiseSOM(metric,fluxes[indices,:],fluxes_err[indices,:], \
									 learning=hh, \
									 shape=(self.somres,self.somres), \
									 wrap=False,logF=True, \
									 initialize='sample', \
									 minError=0.02)

		return som

	def _reinitialize_SOM(self, som_weights, hFunc_sigma = (30, 1), lnScaleSigma = 0.4,  
									lnScaleStep = 0.03, minError = 0.02):

		metric  = AsinhMetric(lnScaleSigma = lnScaleSigma, lnScaleStep = lnScaleStep)
		som     = NoiseSOM(metric, None, None, learning = None, shape = som_weights.shape[:-1],
                       	 wrap=False, logF=True, initialize=som_weights, minError = minError)
	
		return som


	def _get_catalogs(self, train_path, col_fmt, cov_fmt, err_fmt):

		for path, cattype in [(train_path, 'train')]:
			t = Table.read(train_path, format='fits')
			self._get_available_bands(t, col_fmt, cov_fmt, err_fmt) 

			fluxes = self._get_fluxes(t) 
			if self.use_covariances:
				fluxes_cov = self._get_covs(t) 
				fluxes_err = np.sqrt(fluxes_cov)
			else:
				fluxes_err = self._get_errs(t)

			setattr(self, '%s_sample'%cattype, t)
			setattr(self, '%s_fluxes'%cattype, fluxes)
			setattr(self, '%s_err'%cattype, fluxes_err)

	def _get_columns(self, t, colnames):
		return np.array([t[fcn].data for fcn in colnames]).squeeze().T

	def _get_fluxes(self, t):
		return np.array([t[fcn].data for fcn in self.flux_cols]).squeeze().T

	def _get_covs(self, t):
		return np.array([t[ccn].data for ccn in self.cov_cols]).squeeze().T

	def _get_errs(self, t):
		return np.array([t[ecn].data for ecn in self.err_cols]).squeeze().T

	def _get_available_bands(self, t, col_fmt, cov_fmt, err_fmt):
		flxcols, coverrcols = self._get_column_names(t, col_fmt, cov_fmt, err_fmt)
		bands = [cn[len(col_fmt):] for cn in flxcols]

		setattr(self, "bands", bands)
		setattr(self, "flux_cols", flxcols)
		if self.use_covariances: setattr(self, "cov_cols", coverrcols)
		else: setattr(self, "err_cols", coverrcols)

	def _get_column_names(self, t, col_fmt, cov_fmt, err_fmt):
		flux_cols = [cn for cn in t.colnames if cn.startswith(col_fmt)
														 and len(col_fmt)==len(cn)-1]
		if err_fmt is not None:
			coverr_cols = [cn for cn in t.colnames if cn.startswith(err_fmt) 
																and len(err_fmt)==len(cn)-1]
			setattr(self, "use_covariances", False)
		elif cov_fmt is not None:
			coverr_cols = [cn for cn in t.colnames if cn.startswith(cov_fmt)
																and len(cov_fmt)==len(cn)-1]
			setattr(self, "use_covariances", True)
		else:
			raise ValueError("Must set either cov_fmt or err_fmt to non-None value")

		if len(flux_cols) == 0:
			raise ValueError("Table does not have columns starting with " + col_fmt)
		elif len(coverr_cols) == 0 and err_fmt is not None:
			raise ValueError("Table does not have columns starting with " + err_fmt)
		elif len(coverr_cols) == 0 and cov_fmt is not None:
			raise ValueError("Table does not have columns starting with " + cov_fmt)

		return flux_cols, coverr_cols

	def train(self):
		self.SOM.train()
		if self.save_path is not None:
			self.SOM.save(path=self.save_path)
		
		setattr(self, "trained", True)

	def load(self):
		fname = os.path.join(self.save_path, 'NoiseSOM.pkl')
		try:
			som = load_NoiseSOM(fname)	
		except KeyError:
			full_som = load_SOM(fname)
			som = full_som.SOM
		setattr(self, 'SOM', som)


	def _load_assignments(self):
		'''Load previously calculated assignments.'''
		fpath = os.path.join(self.save_path, "assignments.pkl")
		with open(fpath, 'rb') as f:
			assignments = pickle.load(f)
		return assignments

	def classify(self, table, num_threads=150, num_inds=100000, savepth=None,
					 flux_fmt=None, err_fmt=None, cov_fmt=None, ID_fmt='ID'):

		flx_cn, fer_cn = self._get_column_names(table, flux_fmt, cov_fmt, err_fmt)
		fluxes = self._get_columns(table, flx_cn)
		if err_fmt is not None: errs = self._get_columns(table, fer_cn)
		else: errs = np.sqrt(self._get_columns(table, fer_cn))

		return ClassifyRunner(fluxes, errs, table[ID_fmt], self.SOM.weights)


	def save(self, savepath='.'):
		
		if os.path.isdir(savepath): 
			raise ValueError("SOM save path must include the name of the file to be saved")

		to_save = {} 
		ivars = ['save_path', 'somres', 'train_cat_path', 
					'bands', 'SOM', 'col_fmt', 'cov_fmt', 'err_fmt']
		for ivar in ivars:
			to_save[ivar] = getattr(self, ivar)

		with open(savepath, 'wb') as f:
			pickle.dump(to_save, f)

def load_SOM(savepath):

	with open(savepath, 'rb') as f:
		sd = pickle.load(f)

	som = SOM(sd['somres'], sd['train_cat_path'],  
						analysis_output_path=sd['save_path'], 
						col_fmt=sd['col_fmt'], cov_fmt=sd['cov_fmt'], err_fmt=sd['err_fmt'])

	ivar_to_skip = []
	for ivar in sd:
		if ivar in ivar_to_skip: continue
		setattr(som, ivar, sd[ivar])

	return som

def masked_nanmean(col):
	return np.nanmean(col[(col<np.inf)&(col>-0.9e10)])

def masked_nanstd(col):
	return np.nanstd(col[(col<np.inf)&(col>-np.inf)])








