
import pickle
import numpy as np
from astropy.table import Table
import multiprocess as mp
import tqdm

from NoiseSOM import *

class SOM(object):

	def __init__(self, *args, **kwargs):
		'''
		Initialize a SOM object (wraps Gary's NoiseSOM)

		args: 
			- resolution (int): the square side size of the SOM
			- train_cat (str): path to the randomly selected catalog for training the SOM
			- validate_cat (str): path to the randomly selected catalog for validating the SOM
			- *analysis_output_path (str): directory to save analysis ouptuts

		Note: the input catalogs must have the following column name conventions:
			- each flux band has the name: "Mf_[g,r,i,z,etc]" (e.g. for the g band, "Mf_g")
			- each flux covariance band has the name "cov_Mf_[g,r,etc]" (e.g. for the g band "cov_Mf_g")

		'''

		if len(args)==3 and len(kwargs)==1:
	
			resolution, train_cat, validate_cat = args 
			analysis_output_path = kwargs.get('analysis_output_path', '.')

			self.save_path = os.path.abspath(analysis_output_path)
			self.somres = resolution
			self.train_cat_path = os.path.abspath(train_cat)
			self.validate_cat_path = os.path.abspath(validate_cat)
			
			self._get_catalogs(train_cat, validate_cat)
			self.SOM = self._initialize_SOM(self.train_fluxes, self.train_err)
			self.trained = False
		elif len(args)==0 and len(kwargs)==0:
			pass
		else:
			raise TypeError("SOM.__init__() takes 3 arguments and "+\
								 "1 kwarg (%i, %i given)"%(len(args), len(kwargs)))


	def _get_catalogs(self, train_path, validate_path):

		for path, cattype in [(train_path, 'train'), (validate_path, 'validate')]:
			t = Table.read(train_path, format='fits') ; self._get_available_bands(t) 
			fluxes = self._get_fluxes(t) #np.array([t['Mf_%s'%s] for s in self.bands]).T
			fluxes_cov = self._get_covs(t) #np.array([t['cov_Mf_%s'%s] for s in self.bands]).T
			fluxes_err = np.sqrt(fluxes_cov)

			setattr(self, '%s_sample'%cattype, t)
			setattr(self, '%s_fluxes'%cattype, fluxes)
			setattr(self, '%s_err'%cattype, fluxes_err)

	def _get_fluxes(self, t):
		return np.array([t['Mf_%s'%s] for s in self.bands]).T

	def _get_covs(self, t):
		return np.array([t['cov_Mf_%s'%s] for s in self.bands]).T

	def _get_available_bands(self, t):
		bands = [cn[len('Mf_'):] for cn in t.colnames if cn.startswith("Mf_")]
		setattr(self, "bands", bands)

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


	def validate(self, overwrite=False, num_inds=1000):
		'''Classify the validation data to examine efficacy of training.'''

		if len(self.validate_sample) == 0: return

		if not overwrite and self.save_path is not None and \
			 os.path.exists(os.path.join(self.save_path, "assignments.pkl")):
			assignments = self._load_assignments()
		else:
			assignments = self._run_assignments(num_inds)
		

		self.validate_sample['CA'] = assignments

		grouped_by_cell = self.validate_sample.group_by('CA')
		setattr(self, 'grouped_by_cell', grouped_by_cell)

	def _load_assignments(self):
		'''Load previously calculated assignments.'''
		fpath = os.path.join(self.save_path, "assignments.pkl")
		with open(fpath, 'rb') as f:
			assignments = pickle.load(f)
		return assignments

	def _run_assignments(self, table=None, num_inds=1000, save=True):

		fluxes = self.validate_fluxes if table is None else self._get_fluxes(table)
		errs = self.validate_err if table is None else np.sqrt(self._get_covs(table))

		def assign_som(index):
			cells, _ = self.SOM.classify(fluxes[inds[index]], 
                                      errs[inds[index]])
			return cells
    
		num_inds = min(len(fluxes), num_inds)
		inds = np.array_split(np.arange(len(fluxes)),num_inds)
		with mp.Pool(30) as p: 
			results = list(tqdm.tqdm(p.imap(assign_som, range(num_inds)), total=num_inds))
        
		assignments = []
		for res in results:
			assignments = np.append(assignments,res)
      
		if save:
			fpath = os.path.join(self.save_path, "assignments.pkl")
			with open(fpath, 'wb') as f:
				pickle.dump(assignments, f)
				f.close()

		return assignments

	def classify(self, table):
		
		assignments = self._run_assignments(table=table, save=False)
		return assignments
		


	def get(self, statistic, colname=None):
		if len(self.validate_sample.groups) == 1: 
			grouped_by_cell = self.validate_sample.group_by('CA')
			setattr(self, 'grouped_by_cell', grouped_by_cell)

		if statistic == 'occupation': # return somres x somres array of number of galaxies per cell
			occupation=np.histogram(self.validate_sample['CA'], 
                        				bins=self.somres*self.somres, 
                        				range=(0,self.somres*self.somres))[0]
			return occupation.reshape((self.somres, self.somres))
		elif colname is None:
			raise ValueError("Must specify column to aggregate")
		else: # return somres x somres array of aggregated statistic
			if statistic == 'mean': 
				agg = self.grouped_by_cell.groups.aggregate(masked_nanmean)
			elif statistic == 'std':
				agg = self.grouped_by_cell.groups.aggregate(masked_nanstd)
			else:
				raise NotImplementedError("Must define aggregate operation %s in SOM class"%statistic)

			stat = np.zeros(self.somres**2)
			stat[np.array(agg['CA'], dtype=int)] = agg[colname]
			return stat.reshape((self.somres, self.somres))

	def save(self, savepath='.'):
		
		if os.path.isdir(savepath): 
			raise ValueError("SOM save path must include the name of the file to be saved")

		to_save = {} 
		ivars = ['save_path', 'somres', 'train_cat_path', 'validate_cat_path',
					'bands', 'SOM']
		for ivar in ivars:
			to_save[ivar] = getattr(self, ivar)

		with open(savepath, 'wb') as f:
			pickle.dump(to_save, f)

def load_SOM(savepath):

	with open(savepath, 'rb') as f:
		sd = pickle.load(f)

	som = SOM(sd['somres'], sd['train_cat_path'], sd['validate_cat_path'], 
						analysis_output_path=sd['save_path'])

	ivar_to_skip = []
	for ivar in sd:
		if ivar in ivar_to_skip: continue
		setattr(som, ivar, sd[ivar])

	som.validate()
	return som

def masked_nanmean(col):
	return np.nanmean(col[(col<np.inf)&(col>-0.9e10)])

def masked_nanstd(col):
	return np.nanstd(col[(col<np.inf)&(col>-np.inf)])








