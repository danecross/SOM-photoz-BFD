
import os
from matplotlib import pyplot as plt
import multiprocess as mp

from SOM import SOM
from PZC import PZC, load_PZC
from TomographicBins import *
from pipeline_tools import *

import warnings
warnings.filterwarnings("ignore")

base_output = '../outputs/basic/'
if not os.path.exists(base_output): os.mkdir(base_output)

##################
## SOM TRAINING ##
##################
if __name__=='__main__':
	wide_SOM_path = base_output + 'wide/'
	if not os.path.exists(wide_SOM_path): os.mkdir(wide_SOM_path)
	wide_SOM = SOM(32, '../data/wide_field_data/TRAIN_CAT_1E+05.fits',
						 	 '../data/wide_field_data/VALIDATION_CAT_1E+06.fits', 
						 	 analysis_output_path=wide_SOM_path)

	if not os.path.exists(wide_SOM_path+"SOM.pkl"):
		print("training wide SOM...")
		wide_SOM.train()
	else:
		wide_SOM.load()

	print("wide SOM ready")

	wide_SOM.validate(overwrite=False)

	deep_SOM_path = base_output+'deep/'
	if not os.path.exists(deep_SOM_path): os.mkdir(deep_SOM_path)
	deep_SOM = SOM(64, '../data/deep_field_data/BFD/TRAIN_CAT_1E+05.fits',
						 	 '../data/deep_field_data/BFD/VALIDATE_CAT_1E+05.fits',
						 	 analysis_output_path=deep_SOM_path)

	if not os.path.exists(deep_SOM_path+"SOM.pkl"):
		print("training deep SOM...")
		deep_SOM.train()
	else:
		deep_SOM.load()
	print("deep SOM ready")

	deep_SOM.validate(overwrite=False)

##################
## MAKE P(Z|C^) ##
##################

if __name__=='__main__':

	if not os.path.exists(base_output+'PZC.pkl'):

		### Make Transfer Function ###
		if not os.path.exists(base_output): os.mkdir(base_output)
		pzc = PZC(wide_SOM, deep_SOM, base_output,
			  		'/pscratch/sd/d/dncross/SOM-photoz-BFD/data/covariances/BFD_covariances_masked.pkl')

		# make gaussian wide simulations
		print("getting deep SOM realizations onto the wide SOM...")
		if not os.path.exists(base_output+"simulations.fits"):
			pzc.generate_realizations()
		else:
			pzc.load_realizations()
		print("deep to wide simulations ready")

		# calculate p(z|chat)
		pzc.make_redshift_map(weighted=False)
		print("Redshift map ready")

		pzc.save(base_output)

	else: pzc = load_PZC(base_output)

	print("PZ loaded")

###########################
## MAKE TOMOGRAPHIC BINS ##
###########################

if __name__ == '__main__':	

	TB = TomographicBins(pzc, base_output)

	# for this example, let's do 5% in the compost bin
	five_percent = TB.make_bins(4, compost_bin=0.05)

	tomobin_path = os.path.join(base_output, "nzs")
	if not os.path.exists(tomobin_path): os.mkdir(tomobin_path)
	five_percent.save(tomobin_path)




