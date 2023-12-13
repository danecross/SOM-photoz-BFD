
from matplotlib import pyplot as plt
import numpy as np
import pickle
from scipy.signal import savgol_filter

from astropy.table import Table

# calculates the weights for a whole catalog
def calculate_weights(somres, classified_table, out_path):

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
    return -2.5 * np.log10(f) + const

# calculates total signal to noise
def SN(fluxes, fluxes_err, ):
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


# calculate the expectation value of a distribution
def E(v, pv):
	if np.sum(pv)==0: return 0
	return np.average(v, weights=pv) 

# calculate the dispersion of a distribution
def D(v, pv):
	return np.sqrt(E(v**2, pv) - E(v,pv)**2)



