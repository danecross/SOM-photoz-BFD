'''Functions for making Gaussian Simulations.'''

from DataCuts import *
from astropy.table import Table
import numpy as np
import random
import time

# generates n_realizations fluxes for single deep field galaxy
def gen_fluxes(gal_to_sim, nonz_pixs, err_maps, nt_map, n_realizations, bands,
						deep_col_fmt, Mf_fmt, Mr_fmt, id_fmt):

	inj_pix = random.choices(nonz_pixs, k=n_realizations)
	inj_covs = {k: err_maps[k][inj_pix] for k in err_maps}
	inj_means = {'Mf_'+b: gal_to_sim[deep_col_fmt+b] for b in bands}
	inj_means['Mf'] = gal_to_sim[Mf_fmt] ; inj_means['Mr'] = gal_to_sim[Mr_fmt]

	wide_realizations, wide_rlzns_err = make_flux_realizations(inj_means, inj_covs, bands, n_realizations)
	wide_realizations = wide_realizations | make_Mr_realizations(inj_means, inj_covs, wide_realizations['Mf'])
	wide_realizations['inj_pix'] = inj_pix
	wide_realizations['NOISETIER'] = nt_map[inj_pix]

	d2t = wide_realizations | wide_rlzns_err

	t = Table(data=[d2t[k] for k in d2t], names=[k for k in d2t])
	t['ID'] = [gal_to_sim[id_fmt]] * n_realizations

	return t

def make_flux_realizations(inj_means, inj_covs, bands, n_realizations, band_weights=[0,0.6,0.3,0.1]):

	#TODO: include shot noise
	wide_realizations = {'Mf': np.zeros(n_realizations)} ; wide_rlzns_err = {}
	for i,b in enumerate(bands):
		k = 'Mf_'+b
		wide_realizations[k] = np.random.normal(inj_means[k], inj_covs[k])
		wide_rlzns_err['cov_'+k] = inj_covs[k]

		wide_realizations['Mf'] += band_weights[i]*wide_realizations[k]

	return wide_realizations, wide_rlzns_err


def make_Mr_realizations(inj_means, inj_covs, Mf):
	
	Mr = mulvar_rand(Mf, [inj_means['Mf'], inj_means['Mr']], inj_covs['MfMr'])

	return {'Mr': Mr}

def mulvar_rand(x, mean_vector, cov_matrix):

	mean_given_x = mean_vector[1] + cov_matrix[:,1,0] / cov_matrix[:,0,0] * (x - mean_vector[0])
	var_given_x = cov_matrix[:,1,1] - cov_matrix[:,1,0] / cov_matrix[:,0,0] * cov_matrix[:,0,1]

	random_value = np.random.normal(mean_given_x, np.sqrt(var_given_x))

	return random_value

def apply_cuts(t, return_separate_masks=False):

	size_mask = get_size_mask(t['Mf'], t['Mr'])
	mask_sn = get_sn_mask(t['Mf'], preassigned_noisetiers=t['NOISETIER'])

	full_mask = (size_mask & mask_sn)
	if not return_separate_masks:
		return t[full_mask]
	else:
		return t[full_mask], size_mask, mask_sn


