
import os
from matplotlib import pyplot as plt
import multiprocess as mp

from SOM import SOM
from PZ import PZ, load_PZ
from pipeline_tools import *

import warnings
warnings.filterwarnings("ignore")

base_output = '../outputs/basic/'
if not os.path.exists(base_output): os.mkdir(base_output)

images_path = base_output+"images/"
if not os.path.exists(images_path): os.mkdir(images_path)

##################
## SOM Training ##
##################
if __name__=='__main__':
	wide_SOM_path = base_output + 'BFD/'
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
	#make_occupation_plot(wide_SOM, images_path)
	#make_color_plot(wide_SOM, images_path)

	deep_SOM_path = base_output+'BFD_deep/'
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
	#make_occupation_plot(deep_SOM, deep_SOM_path)
	#make_color_plot(deep_SOM, deep_SOM_path)

#################
## PZ Pipeline ##
#################

if __name__=='__main__':

	pz_outpath = '../outputs/basic/PZ_pipeline/'
	if not os.path.exists(pz_outpath+'PZ.pkl'):

		### Make Transfer Function ###
		xfer_output = base_output+'mock_assignments/'
		if not os.path.exists(xfer_output): os.mkdir(xfer_output)
		pz = PZ(wide_SOM, deep_SOM, xfer_output,
			  		'/pscratch/sd/d/dncross/SOM-photoz-BFD/data/covariances/BFD_covariances_masked.pkl')

		# make gaussian wide simulations
		mp.set_start_method('spawn')
		print("getting deep SOM realizations onto the wide SOM...")
		if not os.path.exists(xfer_output+"simulations.fits"):
			pz.generate_realizations()
		else:
			pz.load_realizations()
		print("deep to wide simulations ready")

		# calculate p(z|chat)
		pz.make_redshift_map(weighted=False)
		plt.imshow(pz.redshift_map)
		plt.savefig(images_path+'redshift_map.png')
		print("Redshift map ready")

		if not os.path.exists(pz_outpath): os.mkdir(pz_outpath)
		pz.save('../outputs/basic/PZ_pipeline/')

	else: pz = load_PZ(pz_outpath)

	print("PZ loaded")

	### Make Tomographic Bins ###
	# for this example, let's do 5% in the compost bin
	five_percent = pz.make_bins(4, compost_bin=0.05)
	tomographic_bin_plot(five_percent, images_path+"tomobin.png")

