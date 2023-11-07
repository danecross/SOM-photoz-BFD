
import os
import pytest 

from SOM import *
from PZC import *

som_path = "fixtures/loading/SOM.pkl"
covariance_path = "fixtures/loading/covariance.pkl"

class TestPZC:

	@pytest.fixture
	def wide_som(self):
		return load_SOM(som_path)

	@pytest.fixture
	def deep_som(self):
		return load_SOM(som_path)

	@pytest.fixture
	def outpath(self):
		output_path = "fixtures/pzc_tests/"
		if not os.path.exists(output_path): 
			os.mkdir(output_path)
		return output_path

	## test initialization configurations

	def test_gaussian_init(self, wide_som, deep_som, outpath):
		pzc = PZC(wide_som, deep_som, outpath, covariance_path)

	def test_nongaussian_init(self, wide_som, deep_som, outpath):

		with pytest.raises(NotImplementedError):
			pzc = PZC(wide_som, deep_som, outpath)

