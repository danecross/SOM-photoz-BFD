
import multiprocess as mp
import pickle
import os

from SOM import *
from PZC import *
from pipeline_tools import *

class TomographicBins(object):

	def __init__(self, pzc, outpath, large_data_path=None, overwrite=False):
		
		self.pzc = pzc
		self.outpath = outpath

		if large_data_path is not None:
			self.large_data_path = large_data_path

		self.classified = False

	def classify(self, num_inds=1000, overwrite=False):

		ws = self.pzc.wide_SOM
		som_copy = SOM(ws.somres, 
							self.large_data_path,
							self.large_data_path,		
							analysis_output_path=ws.save_path) 

		som_copy.load()
		som_copy.validate(overwrite=overwrite, num_inds=num_inds)

		table_outpath = os.path.join(self.outpath, "assigned_table.fits")
		som_copy.grouped_by_cell.write(table_outpath, overwrite=True)


	def make_bins(self, num_tomo_bins, **kwargs):
		'''
		Function to generate the tomographic bins
		
		args:
			- num_tomo_bins (int): number of tomographic bins
			- *compost_sigma (float): redshift sigma maximum for compost
			- *ng_per_bin (np.array): array of pre-counted ng_per_bin 
			- *weights (np.array): array of wide cell weights
		'''

		if kwargs.get('weights', None) is not None:
			raise NotImplementedError
		
		if kwargs.get('ng_per_bin', None) is None:
			available_bins, occupation = self._get_bin_populations()
		else:
			occupation = kwargs.get('ng_per_bin')
			available_bins = [wc for wc,occ in enumerate(occupation) if occ>0]

		compost_bin_WCs, available_bins, pct_in_compost = \
			self._assign_to_compost(available_bins,
											kwargs.get('compost_sigma',None),
											occupation)

		binned_WCs = self._assign_to_tomo_bins(available_bins, num_tomo_bins, occupation)

		return Result(self.pzc, binned_WCs, compost_bin_WCs)


	def _get_bin_populations(self, overwrite_assignments=False):
		'''
		Gets the bin population for the sample from the wide SOM sample.

		args: 
			- *BSOM (BSOM): to get the population for the whole DES catalog, use a 
								 BSOM (Big data SOM). See BSOM.py for more information.
		'''

		som = self.pzc.wide_SOM 
		available_bins = [int(i) for i in list(set(som.grouped_by_cell['CA']))]
		som.validate(overwrite_assignments)
		return available_bins, som.get('occupation').flatten()

	def _assign_to_compost(self, available_bins, compost_sigma, occupation):

		if compost_sigma is None: return [], available_bins, 0

		# order bins on standard deviation of pzchat
		stddev = [D(self.pzc.redshifts, self.pzc.pzchat[i]) for i in available_bins]
		ordered_bins = [wc for _,wc in reversed(sorted(zip(stddev, available_bins)))]
		stddev = [std for std,_ in reversed(sorted(zip(stddev, available_bins)))]

		# remove bins one by one until the sigma cross specified threshold
		compost_wcs = [] ; ng_in_compost = 0
		while stddev.pop(0) > compost_sigma:
			compost_wcs += [ordered_bins.pop(0)]
			ng_in_compost += occupation[compost_wcs[-1]]

		return compost_wcs, ordered_bins, ng_in_compost/np.sum(occupation)


	def _assign_to_tomo_bins(self, available_bins, num_tomo_bins, occupation):

		medians = [E(self.pzc.redshifts, self.pzc.pzchat[i]) for i in available_bins]
		ordered_bins = [wc for m,wc in sorted(zip(medians, available_bins)) if m>0]
		if len(ordered_bins) == 0: 
			raise ValueError("no bins to assign (compost bin may be too big)")

		total_ng = np.sum(occupation[ordered_bins]) 
		bin_wcs = []
		for i in range(num_tomo_bins):
			wcs = [] ; ng_in_bin = 0
			while ng_in_bin/total_ng < 1/num_tomo_bins and len(ordered_bins)>0:
				wcs += [ordered_bins.pop(0)]
				ng_in_bin += occupation[wcs[-1]]
			
			bin_wcs += [wcs]

		return bin_wcs


	def save(self):
		raise NotImplementedError

class PZCB(PZC):

	def __init__(self, pzc, subcatalog):

		self.pzc = pzc
		self.simulations = subcatalog

		self.deep_SOM = pzc.deep_SOM
		self.wide_SOM = pzc.wide_SOM

	#TODO: generate_realizations will be different for p(z|c,chat)
	

class Result(object):

	def __init__(self, pzc, binned_WCs, compost_WCs):

		self.pzchat = pzc.pzchat
		self.binned_WCs = binned_WCs
		self.compost_WCs = compost_WCs
		self.z = pzc.redshifts

		# for selection effects
		self.pzc = pzc
		self.pzc.load_realizations()

	def calculate_Nz(self, apply_bin_cond=True, weights=None, zmax=6, fill_zeros=False):

		if not apply_bin_cond:
			Nz = {}
			if len(self.compost_WCs) > 0:
				Nz[-1] = np.sum(self.pzchat[self.compost_WCs], axis=0)/len(self.compost_WCs)
			for i,tbin in enumerate(self.binned_WCs):
				Nz[i] = np.sum(self.pzchat[tbin], axis=0)/len(tbin)

		else:
			pzchats, trash_pzchats = self._apply_bin_cond(weights=None, zmax=6, fill_zeros=fill_zeros)
			Nz = {}
			for i,pzchat in enumerate(pzchats):
				Nz[i] = np.sum(pzchat, axis=0)/np.sum(pzchat)
			Nz[-1] = np.sum(trash_pzchats, axis=0)/np.sum(pzchat)

		return Nz

	def _apply_bin_cond(self, weights=None, zmax=6, fill_zeros=False):
		
		compost, grouped_by_bin = self._group_sims_by_tomobin()
		pzcbs = []
		for tomobin_subsample in grouped_by_bin.groups:
			pzcb = PZCB(self.pzc, tomobin_subsample)
			pzcb.make_redshift_map(weights=weights, zmax=zmax, fill_zeros=fill_zeros)
			pzchat = pzcb.pzchat
			
			pzcbs += [pzchat]

		pzcb_t = PZCB(self.pzc, compost)
		pzcb_t.make_redshift_map(weights=weights, zmax=zmax)
		trash_pzcb = pzcb_t.pzchat

		return pzcbs, trash_pzcb

	def _group_sims_by_tomobin(self, table=None):
		# assign tomo bins to each wide simulation
		simulated_widegals = self.pzc.simulations if table is None else table
		bin_assignment = (-1)*np.ones(len(simulated_widegals)) #default to compost
		for i,row in enumerate(simulated_widegals):
			for j,tbin in enumerate(self.binned_WCs):
				if row['WC'] in tbin: bin_assignment[i] = j

		simulated_widegals['tomo_bin'] = bin_assignment

		compost_mask = (simulated_widegals['tomo_bin']==-1)
		compost = simulated_widegals[compost_mask]
		simulated_widegals = simulated_widegals[~compost_mask]

		grouped_by_tbin = simulated_widegals.group_by('tomo_bin')

		return compost, grouped_by_tbin

	def bin_sample(self, table):
		if not 'WC' in table.colnames:
			assignments = self.pzc.wide_SOM.classify(table)
			table['WC'] = assignments
		else:
			assignments = table['WC']
		return assignments, self._group_sims_by_tomobin(table)
	

	def save(self, output_path):
		
		if os.path.isdir(output_path): 
			output_path = os.path.join(output_path, "Nz.pkl")

		dts = {"pzc": self.pzc, "tomographic_bins": self.binned_WCs,
				 "compost": self.compost_WCs}
		with open(output_path, 'wb') as f:
			pickle.dump(dts, f)


def load_result(path):
	
	with open(path, 'rb') as f:
		dtr = pickle.load(f)

	return Result(dtr['pzc'], dtr['tomographic_bins'], dtr['compost'])
			


