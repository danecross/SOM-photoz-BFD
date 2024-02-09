
from mpi4py import MPI
from matplotlib import pyplot as plt
import numpy as np
import pickle
from scipy.signal import savgol_filter

from astropy.table import Table

#######################################
## Large-Scale Classification Scheme ##
#######################################

def run_classification(tfname, comm, som, col_name, classify_kwargs={}):
	'''
	Runs classifications in an MPI environment.

	Args:
		- tfname (str): filename of the table to classify
		- comm (MPI COMM object): the communication object for the MPI process
		- som (SOM): SOM instance to use to classify the data
		- col_name (str): name of the classified column (usually 'WC'/'DC' for wide/deep classification)
		- classify_kwargs (dict): keyword arguments for the SOM classify function
	'''

	rank = comm.Get_rank() ; size = comm.Get_size()
	t = Table.read(tfname, memmap=True)
	
	if rank > 0:
		this_cat = split_table(t, rank, size)
		assignments = som.classify(this_cat, **classify_kwargs)
		comm.send(assignments, dest=0, tag=11)

	if rank == 0:
		assignments = np.array([])
		for i in range(1,size):
			assignments = np.append(assignments,comm.recv(source=i, tag=11))

		t[col_name] = assignments

		return t

def split_table(t, rank, size):
	'''Splits an astropy table given a thread number and total number of threads.'''
	srt, end = int(((rank-1)/size)*len(t)), int(((rank)/size)*len(t))
	if rank == size-1: this_cat = t[srt:]
	else: this_cat = t[srt:end]

	return this_cat



############################
## Other Useful Functions ##
############################

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



