
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

# converts flux to magnitude
def flux_to_mag(f, const=30):
    return -2.5 * np.log10(f) + const

# calculates total signal to noise
def SN(fluxes, fluxes_err):
    g = fluxes[:,0] ; r = fluxes[:,1]
    i = fluxes[:,2] ; z = fluxes[:,3]
    
    ge = fluxes_err[:,0] ; re = fluxes_err[:,1]
    ie = fluxes_err[:,2] ; ze = fluxes_err[:,3]
    
    signal = np.sqrt(0.7 * r**2 + 0.2 * i**2 + 0.1 * z**2)
    noise = np.sqrt(0.7 * re**2 + 0.2 * ie**2 + 0.1 * ze**2)
    return signal/noise

# calculate the expectation value of a distribution
def E(v, pv):
	if np.sum(pv)==0: return 0
	return np.average(v, weights=pv) 

# calculate the dispersion of a distribution
def D(v, pv):
	return np.sqrt(E(v**2, pv) - E(v,pv)**2)

####################
## SOM processing ##
####################

def make_occupation_plot(SOM, outpath):

	occupation = SOM.get('occupation')
	plt.imshow(occupation)
	plt.title("Cell Occupation")
	if outpath is not None:
		plt.savefig(outpath+'occupation.png')

def make_color_plot(SOM, outpath):
	
	bands = ['g','r','i','z']
	means = [SOM.get('mean', 'Mf_%s'%b) for b in bands]
	stds = [SOM.get('std', 'Mf_%s'%b) for b in bands]

	fig, axs = plt.subplots(2,3, figsize=(30,10))
	for i, (ax, b1, b2) in enumerate(zip(axs.T, bands, bands[1:])):
		ax[0].imshow(flux_to_mag(means[i])-flux_to_mag(means[i+1]))
		ax[1].imshow(flux_to_mag(stds[i])-flux_to_mag(stds[i+1]))

		ax[0].set_title('%s-%s'%(b1,b2))

	if outpath is not None:
		plt.savefig(outpath+'occupation.png')

def tomographic_bin_plot(result, outpath='tomobin.png'):
	
	distributions = result.calculate_Nz()
	z = result.z

	colors = [plt.cm.tab20(i*2) for i in range(5)]
	fig, ax = plt.subplots(2, figsize=(25,17), height_ratios=[6,1])
	for i in distributions:
		if i == -1: continue
		nz = distributions[i]
		ax[0].plot(z,nz, color=colors[i])
		ax[0].fill_between([0]+z, savgol_filter([0]+nz, 5, 1), alpha=0.3, color=colors[i], label="Bin %i"%i)
 
		av = E(z,nz) 
		ax[0].axvline(av, color=colors[i], label='bin average = %.02f'%av)
    

	nz = distributions[-1]
	ax[1].plot(z, nz, color=colors[-1],)
	ax[1].fill_between([0]+z, savgol_filter([0]+nz, 5, 1), alpha=0.3, color=colors[-1], label='Trash Bin')
	av = E(z,nz) 
	ax[1].axvline(av, color=colors[-1], label='bin average = %.02f'%av)

	ax[0].legend(fontsize=25)
	ax[1].legend(fontsize=25)
	ax[0].tick_params(axis='both', which='major', labelsize=25)
	ax[1].tick_params(axis='both', which='major', labelsize=25)

	ax[1].set_xlabel('Redshift', fontsize=25)
	if outpath is not None:
		plt.savefig(outpath+'occupation.png')



