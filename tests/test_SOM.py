
import pickle
import os
import shutil

from SOM import *
import pytest

import warnings
warnings.filterwarnings("ignore")

catpath = 'fixtures/FIXTURE_CAT_1E+04.fits'

class TestSOM:

	## CONSTRUCTOR TESTS ##

	# fixtures
	
	@pytest.fixture
	def som(self):
		return SOM()

	@pytest.fixture
	def preread_table(self, som):
		som._get_catalogs(catpath, catpath) 
		return som.train_sample

	@pytest.fixture
	def output_path(self):
		opath = "fixtures/som_tests/"
		if not os.path.exists(opath): os.mkdir(opath)

		yield opath
		#if os.path.exists(opath): shutil.rmtree(opath)

	# TESTS

	def test__get_catalogs(self, som):
		som._get_catalogs(catpath, catpath)

		assert(hasattr(som, "train_sample"))
		assert(hasattr(som, "validate_sample"))
		assert(hasattr(som, "train_fluxes"))
		assert(hasattr(som, "validate_fluxes"))
		assert(hasattr(som, "train_err"))
		assert(hasattr(som, "validate_err"))

	def test__get_available_bands(self, preread_table, som):
		som._get_available_bands(preread_table)
		assert(som.bands == list('griz'))

	def test__initialize_SOM(self, som):
		som._get_catalogs(catpath, catpath)

		setattr(som, 'somres', 10)
		som._initialize_SOM(som.train_fluxes, som.train_err)

	def test_full_init(self, output_path):
		som = SOM(10, catpath, catpath, **{'analysis_output_path':output_path})


	## TRAIN TEST ##
	
	# additional fixtures

	@pytest.fixture
	def full_som(self, output_path):
		return SOM(10, catpath, catpath, **{'analysis_output_path':output_path})

	# TEST

	def test_train(self, full_som, output_path):
		full_som.train()
		assert(full_som.trained)
		assert(os.path.exists(output_path + 'SOM.pkl'))
	
	## LOAD TEST ##

	def test_load(self, full_som):
		full_som.load()

	
	## VALIDATE TEST ##

	def test__run_assignments(self, full_som):
		full_som._run_assignments()

	def test__load_assignments(self, full_som):
		full_som._load_assignments()
	
	def test_validate(self, full_som):
		full_som.load()
		full_som.validate()

	## SAVE/LOAD TESTS ##

	@pytest.fixture
	def outpath(self, output_path):	
		return os.path.join(output_path, "SOM.pkl")

	def test_save_SOM(self, full_som, outpath):
		
		full_som.save(outpath)
		assert(os.path.exists(outpath))
		
		with open(outpath, 'rb') as f:
			saved = pickle.load(f)

		assert(saved.get('somres', None) is not None)
		assert(saved.get('train_cat_path', None) is not None)
		assert(saved.get('validate_cat_path', None) is not None)
		assert(saved.get('save_path', None) is not None)
		assert(saved.get('bands', None) is not None)
		
	def test_load_SOM(self, outpath):

		load_SOM(outpath)



	
