
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
			- *weights (bool): use cell weights
			- *compost_bin (float): percent of the galaxies to compost
			- *
		'''

		if kwargs.get('weights', False):
			raise NotImplementedError

		available_bins, occupation = self._get_bin_populations()
		if 'compost_bin' in kwargs:
			compost_bin_WCs, available_bins = self._assign_to_compost(available_bins,
																						 kwargs['compost_bin'],
																						 occupation)
		else:
			compost_bin_WCs = []

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

	def _assign_to_compost(self, available_bins, percent_to_compost, occupation):

		# order bins on standard deviation of pzchat
		stddev = [D(self.pzc.redshifts, self.pzc.pzchat[i]) for i in available_bins]
		ordered_bins = [wc for _,wc in reversed(sorted(zip(stddev, available_bins)))]

		# remove bins one by one until percent removed is percent_to_compost
		ng_in_compost = 0 ; total_ng = np.sum(occupation)
		compost_wcs = []
		while ng_in_compost/total_ng <= percent_to_compost:
			compost_wcs += [ordered_bins.pop(0)]
			ng_in_compost += occupation[compost_wcs[-1]]

		return compost_wcs, ordered_bins


	def _assign_to_tomo_bins(self, available_bins, num_tomo_bins, occupation):

		medians = [E(self.pzc.redshifts, self.pzc.pzchat[i]) for i in available_bins]
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

	def calculate_Nz(self, apply_selection_effects=False, weighted=False, zmax=6):

		if not apply_selection_effects:
			Nz = {}
			if len(self.compost_WCs) > 0:
				Nz[-1] = np.sum(self.pzchat[self.compost_WCs], axis=0)/len(self.compost_WCs)
			for i,tbin in enumerate(self.binned_WCs):
				Nz[i] = np.sum(self.pzchat[tbin], axis=0)/len(tbin)

		else:
			pzchats = self._apply_selection_effects(weighted=False, zmax=6)
			Nz = []
			for pzchat in pzchats:
				Nz += [np.sum(pzchat, axis=0)/np.sum(pzchat)]

		return Nz

	def _apply_selection_effects(self, weighted=False, zmax=6):
		
		grouped_by_bin = self._group_sims_by_tomobin()
		pzcbs = []
		for tomobin_subsample in grouped_by_bin.groups:
			pzcb = PZCB(self.pzc, tomobin_subsample)
			pzcb.make_redshift_map(weighted=False, zmax=zmax)
			
			pzcbs += [pzcb.pzchat]

		return pzcbs

	def _group_sims_by_tomobin(self):
		# assign tomo bins to each wide simulation
		simulated_widegals = self.pzc.simulations
		bin_assignment = (-1)*np.ones(len(simulated_widegals)) #default to compost
		for i,row in enumerate(simulated_widegals):
			for j,tbin in enumerate(self.binned_WCs):
				if row['WC'] in tbin: bin_assignment[i] = j-1

		simulated_widegals['tomo_bin'] = bin_assignment
		grouped_by_tbin = simulated_widegals.group_by('tomo_bin')

		return grouped_by_tbin

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
			


