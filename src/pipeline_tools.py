
import os
from mpi4py import MPI
from matplotlib import pyplot as plt
import numpy as np
import pickle
from scipy.signal import savgol_filter

from astropy.table import Table

#######################################
## Large-Scale Classification Scheme ##
#######################################

def run_classification(tfname, som, metasave_path, rank, size,
								classify_kwargs={}):
	'''
	Runs classifications in an MPI environment.

	Args:
		- tfname (str): filename of the table to classify
		- som (SOM): SOM instance to use to classify the data
		- rank (int): rank of the process being run (modelled after MPI)
		- size (int): total number of processes to run 
							(or how many sections to split the input table into)
		- classify_kwargs (dict): keyword arguments for the SOM classify function
	'''

	rank = rank if rank is not None else comm.Get_rank() 
	size = size if size is not None else comm.Get_size()
	svpth = os.path.join(metasave_path, f"assignments_{rank}_{size}.pkl")
	if os.path.exists(svpth):
		return

	t = Table.read(tfname, memmap=True)

	this_cat = split_table(t, rank, size)	
	if len(this_cat) == 0: return
	assignments = som.classify(this_cat, **classify_kwargs)

	with open(svpth, 'wb') as f:
		pickle.dump(assignments, f)

# NOTE: meant to loop from 0 to size **inclusive** i.e. size+1 iterations
def split_table(t, rank, size):
	'''Splits an astropy table given a thread number and total number of threads.'''
	chunk_size = len(t)//size
	srt, end = chunk_size * (rank), chunk_size * (rank+1)

	if rank == size: 
		return t[srt:]
	else:
		return t[srt:end]


############################
## Other Useful Functions ##
############################

# calculates the colors and magnitudes for a given table
def get_cms(t, colfmt='Mf_', bands='griz', decorator=''):

	mask = np.ones(len(t), dtype=bool)
	for b in bands:
		mag = flux_to_mag(t[colfmt+b])
		t[decorator+'mag_'+b] = mag
		mask = (mask & ~np.isnan(mag))
		  
	for b1,b2 in zip(bands, bands[1:]):
		color = t[decorator+'mag_'+b1] - t[decorator+'mag_'+b2]
		t[decorator+f'{b1}-{b2}'] = color
	 
	return t[mask]

# calculates the weights for a whole catalog
def calculate_weights(somres, classified_table, out_path):
	"""Calculates weights from ___ of ___."""
	if type(classified_table) == str:
		t = Table.read(classified_table_path)
	else:
		t = classified_table
	grouped_by_wc = t.group_by("WC")
	
	norm = np.sum(t['nz_R_weight'])
	weights = np.zeros(somres**2)
	for g in grouped_by_wc.groups:
		wc = int(g[0]['WC'])
		weight = np.average(g['nz_R_weight'])/norm
		weights[wc] = weight

	with open(out_path, 'wb') as f:
		pickle.dump(weights, f)
	
	return weights

# converts flux to magnitude
def flux_to_mag(f, const=30):
	 """Converts flux to magnitudes."""
	 return -2.5 * np.log10(f) + const

# calculates total signal to noise
def SN(fluxes, fluxes_err, ):
	 """Calcualtes the Signal-to-Noise using Y3 weights."""
	 g = fluxes[:,0] ; r = fluxes[:,1]
	 i = fluxes[:,2] ; z = fluxes[:,3]
	 
	 ge = fluxes_err[:,0] ; re = fluxes_err[:,1]
	 ie = fluxes_err[:,2] ; ze = fluxes_err[:,3]		

	 signal = 0.7 * r + 0.2 * i + 0.1 * z
	 noise = np.sqrt((0.7 * re)**2 + (0.2 * ie)**2 + (0.1 * ze)**2)
	 return signal/noise

# calculates total signal to noise
def SN_s_n(fluxes, fluxes_err, ):
	 g = fluxes[:,0] ; r = fluxes[:,1]
	 i = fluxes[:,2] ; z = fluxes[:,3]
	 
	 ge = fluxes_err[:,0] ; re = fluxes_err[:,1]
	 ie = fluxes_err[:,2] ; ze = fluxes_err[:,3]		

	 signal = 0.7 * r + 0.2 * i + 0.1 * z
	 noise = np.sqrt((0.7 * re)**2 + (0.2 * ie)**2 + (0.1 * ze)**2)
	 return signal/noise, signal, noise


def E(v, pv):
	'''Calcuates the expectation value of a given distribution.'''
	if np.sum(pv)==0: return 0
	return np.average(v, weights=pv) 

def D(v, pv):
	'''Calculates the dispersion of a given distribution.'''
	return np.sqrt(E(v**2, pv) - E(v,pv)**2)



